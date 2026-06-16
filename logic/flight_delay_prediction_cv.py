#!/usr/bin/env python3
"""
flight_delay_prediction_cv.py
DATAbility "Ready for Takeoff" - Flight Delay Prediction Pipeline

Adds time-aware cross-validation and a stronger decision setup:
  1. A binary model estimates P(delay >= 15 min).
  2. A bucket model estimates the likely delay-size bucket.
  3. TimeSeriesSplit out-of-fold predictions tune the delay-risk threshold.
  4. A regressor is still trained for predicted delay minutes and MAE/RMSE.

The model uses only pre-flight columns plus leak-safe derived history/cascade
features. Do not add answer-key columns such as delay_cause or delay components
as predictors.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------

DATA_PATH = "flights_weather_sample.csv"

BUCKET_LABELS = {
    0: "On time (<15 min)",
    1: "Short delay (15-30 min)",
    2: "Medium delay (30-60 min)",
    3: "Long delay (60-90 min)",
    4: "Severe delay (>=90 min)",
}

BUCKET_COLS = [
    "prob_on_time",
    "prob_15_30_min",
    "prob_30_60_min",
    "prob_60_90_min",
    "prob_90plus_min",
]

REG_BUCKET_COLS = [
    "reg_prob_on_time",
    "reg_prob_15_30_min",
    "reg_prob_30_60_min",
    "reg_prob_60_90_min",
    "reg_prob_90plus_min",
]

RANDOM_STATE = 42
N_BUCKETS = 5


# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------

def resolve_data_path(path: str) -> Path:
    """Resolve a dataset path, with a few useful local fallbacks."""
    requested = Path(path).expanduser()
    candidates = [
        requested,
        Path.cwd() / requested,
        Path.home() / "Desktop" / requested.name,
        Path.home() / "oneplussixteen" / requested.name,
        Path.home() / "Desktop" / "starter_kit_datability" / requested.name,
    ]
    script_file = globals().get("__file__")
    if script_file:
        candidates.insert(2, Path(script_file).resolve().parent / requested)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset not found: {path}\n"
        "Pass it explicitly with --data /path/to/flights_weather_sample.csv"
    )


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame."""
    resolved = resolve_data_path(path)
    return pd.read_csv(resolved, low_memory=False)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise if any required columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# 3. TIME FEATURES
# ---------------------------------------------------------------------------

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse scheduled times and derive useful time features."""
    df = df.copy()

    df["sched_dep_local"] = df["sched_dep_local"].astype(str).str.zfill(5)
    df["sched_dep_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["sched_dep_local"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    df["scheduled_arr_local"] = df["scheduled_arr_local"].astype(str).str.zfill(5)
    df["sched_arr_dt"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["scheduled_arr_local"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    overnight_mask = df["sched_arr_dt"] < df["sched_dep_dt"]
    df.loc[overnight_mask, "sched_arr_dt"] += pd.Timedelta(days=1)

    if "dep_hour" not in df.columns or df["dep_hour"].isna().all():
        df["dep_hour"] = df["sched_dep_dt"].dt.hour

    if "day_of_week" not in df.columns or df["day_of_week"].isna().all():
        df["day_of_week"] = df["sched_dep_dt"].dt.dayofweek + 1

    df["route"] = df["origin"].astype(str) + "_" + df["dest"].astype(str)

    return df


# ---------------------------------------------------------------------------
# 4. BUCKET FUNCTIONS
# ---------------------------------------------------------------------------

def delay_minutes_to_bucket_array(delay_minutes: np.ndarray) -> np.ndarray:
    """Vectorized bucket mapping for delay minutes."""
    arr = np.asarray(delay_minutes, dtype=float)
    buckets = np.zeros(len(arr), dtype=int)
    buckets[arr >= 15] = 1
    buckets[arr >= 30] = 2
    buckets[arr >= 60] = 3
    buckets[arr >= 90] = 4
    return buckets


def full_bucket_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return a 5-column probability matrix, including absent classes as zero."""
    raw = model.predict_proba(X)
    classes = model.named_steps["model"].classes_
    full = np.zeros((len(X), N_BUCKETS), dtype=float)
    for j, cls in enumerate(classes):
        full[:, int(cls)] = raw[:, j]
    return full


def regression_bucket_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Convert per-tree regressor predictions into an empirical bucket distribution."""
    transformed = model.named_steps["preprocess"].transform(X)
    forest = model.named_steps["model"]
    tree_log_preds = np.column_stack([
        tree.predict(transformed)
        for tree in forest.estimators_
    ])
    tree_min_preds = np.expm1(tree_log_preds).clip(min=0)
    tree_buckets = np.zeros_like(tree_min_preds, dtype=int)
    tree_buckets[tree_min_preds >= 15] = 1
    tree_buckets[tree_min_preds >= 30] = 2
    tree_buckets[tree_min_preds >= 60] = 3
    tree_buckets[tree_min_preds >= 90] = 4
    return np.stack(
        [(tree_buckets == bucket).mean(axis=1) for bucket in range(N_BUCKETS)],
        axis=1,
    )


def combine_risk_and_bucket_probs(
    delay_risk: np.ndarray,
    raw_bucket_probs: np.ndarray,
) -> np.ndarray:
    """
    Make a coherent bucket distribution:
      - binary risk controls total P(delay >= 15)
      - bucket classifier controls the split among delayed buckets
    """
    risk = np.asarray(delay_risk, dtype=float).clip(0.0, 1.0)
    probs = np.zeros((len(risk), N_BUCKETS), dtype=float)
    probs[:, 0] = 1.0 - risk

    delayed = raw_bucket_probs[:, 1:].clip(0.0, 1.0)
    delayed_sum = delayed.sum(axis=1, keepdims=True)
    zero_rows = delayed_sum[:, 0] <= 0
    delayed_sum[zero_rows, :] = 1.0
    delayed_norm = delayed / delayed_sum
    delayed_norm[zero_rows, :] = 0.25
    probs[:, 1:] = delayed_norm * risk[:, None]
    return probs


def predict_hybrid_buckets(
    delay_risk: np.ndarray,
    raw_bucket_probs: np.ndarray,
    risk_threshold: float,
) -> np.ndarray:
    """Predict bucket 0 unless delay risk clears the tuned risk threshold."""
    delayed_choice = raw_bucket_probs.argmax(axis=1)
    delayed_choice = np.where(delayed_choice == 0, 1, delayed_choice)
    return np.where(delay_risk >= risk_threshold, delayed_choice, 0).astype(int)


# ---------------------------------------------------------------------------
# 5. CASCADE / INBOUND AIRCRAFT FEATURES
# ---------------------------------------------------------------------------

def add_cascade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Look up the previous leg flown by the same tail number.
    The shift uses prior scheduled order only, so the current row's outcome is
    never used as its own predictor.
    """
    df = df.copy()
    df = df.sort_values(["tail_number", "sched_dep_dt"]).reset_index(drop=True)

    grp = df.groupby("tail_number", sort=False)
    df["prev_flight_id"] = grp["flight_id"].shift(1)
    df["prev_arr_delay_min_raw"] = grp["arr_delay_min"].shift(1)
    df["prev_dep_delay_min_raw"] = grp["dep_delay_min"].shift(1)
    df["prev_sched_arr_dt"] = grp["sched_arr_dt"].shift(1)

    df["has_prev_leg"] = df["prev_flight_id"].notna().astype(int)
    df["turnaround_buffer_min"] = (
        df["sched_dep_dt"] - df["prev_sched_arr_dt"]
    ).dt.total_seconds() / 60.0

    invalid_turnaround = (
        (df["turnaround_buffer_min"] < 0)
        | (df["turnaround_buffer_min"] > 480)
    )
    df.loc[invalid_turnaround, "turnaround_buffer_min"] = np.nan

    df["missing_turnaround"] = df["turnaround_buffer_min"].isna().astype(int)
    df["prev_arr_delay_min"] = df["prev_arr_delay_min_raw"].fillna(0)
    df["prev_dep_delay_min"] = df["prev_dep_delay_min_raw"].fillna(0)
    df["inbound_delay_after_buffer_min"] = (
        df["prev_arr_delay_min"] - df["turnaround_buffer_min"]
    ).clip(lower=0)
    df["had_late_inbound"] = (df["prev_arr_delay_min"] >= 15).astype(int)
    df["tight_turnaround"] = (df["turnaround_buffer_min"] <= 45).astype(int)

    return df


# ---------------------------------------------------------------------------
# 6. CONGESTION FEATURES
# ---------------------------------------------------------------------------

def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count how busy each origin-hour slot is."""
    df = df.copy()

    dep_hour_int = df["dep_hour"].astype(float).astype("Int64")
    df["_dep_hour_key"] = dep_hour_int

    df["departures_origin_hour"] = (
        df.groupby(["origin", "date", "_dep_hour_key"])["flight_id"]
        .transform("count")
    )
    df["origin_hour_congestion_pctile"] = df.groupby("origin")[
        "departures_origin_hour"
    ].rank(pct=True)

    df["is_early_bank"] = ((df["dep_hour"] >= 5) & (df["dep_hour"] <= 8)).astype(int)
    df["is_evening_bank"] = (
        (df["dep_hour"] >= 16) & (df["dep_hour"] <= 20)
    ).astype(int)

    df.drop(columns=["_dep_hour_key"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 7. WEATHER FEATURES
# ---------------------------------------------------------------------------

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive weather severity signals from WMO codes and raw measurements."""
    df = df.copy()

    weather_cols_defaults = {
        "temp_c": 0.0,
        "wind_speed_kmh": 0.0,
        "wind_gust_kmh": 0.0,
        "precip_mm": 0.0,
        "snowfall_cm": 0.0,
        "cloud_cover_pct": 0.0,
        "weather_code": 0,
    }
    for col, default in weather_cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    wc = pd.to_numeric(df["weather_code"], errors="coerce").fillna(0)

    df["is_snow_code"] = wc.between(71, 77).astype(int)
    df["is_rain_code"] = wc.between(51, 67).astype(int)
    df["is_shower_code"] = wc.between(80, 82).astype(int)
    df["is_fog_code"] = (wc == 45).astype(int)
    df["is_thunderstorm_code"] = (wc == 95).astype(int)

    df["weather_severity_score"] = (
        2.5 * pd.to_numeric(df["snowfall_cm"], errors="coerce").fillna(0)
        + 1.5 * pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0)
        + 0.03 * pd.to_numeric(df["wind_gust_kmh"], errors="coerce").fillna(0)
        + 2.0 * df["is_fog_code"]
        + 3.0 * df["is_thunderstorm_code"]
    )

    return df


# ---------------------------------------------------------------------------
# 8. ROLLING DELAY FEATURES
# ---------------------------------------------------------------------------

def add_rolling_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical delay rates for origin, carrier, and route, using only flights
    before the current row.
    """
    df = df.copy().sort_values("sched_dep_dt").reset_index(drop=True)
    df["was_delayed_15"] = (
        pd.to_numeric(df["dep_delay_min"], errors="coerce") >= 15
    ).astype(float)

    def leak_safe_rolling(series: pd.Series, window: int, min_periods: int) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=min_periods).mean()

    df["origin_recent_delay_rate"] = (
        df.groupby("origin")["was_delayed_15"]
        .transform(lambda s: leak_safe_rolling(s, window=50, min_periods=10))
    )
    df["carrier_recent_delay_rate"] = (
        df.groupby("carrier")["was_delayed_15"]
        .transform(lambda s: leak_safe_rolling(s, window=100, min_periods=20))
    )
    df["route_recent_delay_rate"] = (
        df.groupby("route")["was_delayed_15"]
        .transform(lambda s: leak_safe_rolling(s, window=50, min_periods=10))
    )

    return df


# ---------------------------------------------------------------------------
# 9. EXTRA SCHEDULE FEATURES
# ---------------------------------------------------------------------------

def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap pre-flight calendar and schedule-pressure features."""
    df = df.copy()
    df["dep_hour"] = pd.to_numeric(df["dep_hour"], errors="coerce")
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")

    df["dep_hour_sin"] = np.sin(2 * np.pi * df["dep_hour"] / 24)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["dep_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

    date_dt = pd.to_datetime(df["date"], errors="coerce")
    df["date_ordinal"] = (date_dt - date_dt.min()).dt.days

    df["dest_origin_hour_count"] = df.groupby(
        ["dest", "date", "dep_hour"], dropna=False
    )["flight_id"].transform("count")
    df["carrier_origin_hour_count"] = df.groupby(
        ["carrier", "origin", "date", "dep_hour"], dropna=False
    )["flight_id"].transform("count")
    df["route_day_count"] = df.groupby(
        ["route", "date"], dropna=False
    )["flight_id"].transform("count")

    return df


# ---------------------------------------------------------------------------
# 10. BUILD FEATURE TABLE
# ---------------------------------------------------------------------------

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the enriched DataFrame."""
    df = add_time_columns(df)
    df = add_cascade_features(df)
    df = add_congestion_features(df)
    df = add_weather_features(df)
    df = add_rolling_delay_features(df)
    df = add_schedule_features(df)
    df = df.sort_values("sched_dep_dt").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 11. FEATURE COLUMNS
# ---------------------------------------------------------------------------

def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) lists."""
    numeric_features = [
        "day_of_week",
        "dep_hour",
        "distance_km",
        "temp_c",
        "wind_speed_kmh",
        "wind_gust_kmh",
        "precip_mm",
        "snowfall_cm",
        "cloud_cover_pct",
        "weather_code",
        "prev_arr_delay_min",
        "prev_dep_delay_min",
        "turnaround_buffer_min",
        "inbound_delay_after_buffer_min",
        "had_late_inbound",
        "tight_turnaround",
        "has_prev_leg",
        "missing_turnaround",
        "departures_origin_hour",
        "origin_hour_congestion_pctile",
        "is_early_bank",
        "is_evening_bank",
        "is_snow_code",
        "is_rain_code",
        "is_shower_code",
        "is_fog_code",
        "is_thunderstorm_code",
        "weather_severity_score",
        "origin_recent_delay_rate",
        "carrier_recent_delay_rate",
        "route_recent_delay_rate",
        "dep_hour_sin",
        "dep_hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "date_ordinal",
        "dest_origin_hour_count",
        "carrier_origin_hour_count",
        "route_day_count",
    ]
    categorical_features = ["carrier", "origin", "dest", "route"]
    return numeric_features, categorical_features


# ---------------------------------------------------------------------------
# 12. MODELS
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Build the shared preprocessing stage."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=sparse_output,
            ),
        ),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_binary_delay_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Predict whether the flight will be delayed by at least 15 minutes."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.10,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_bucket_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Predict the delay-size bucket."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.10,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_regression_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Predict delay minutes for MAE/RMSE and operator context."""
    return Pipeline([
        (
            "preprocess",
            build_preprocessor(
                numeric_features,
                categorical_features,
                sparse_output=False,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=14,
                min_samples_leaf=8,
                min_samples_split=20,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# ---------------------------------------------------------------------------
# 13. CROSS-VALIDATION
# ---------------------------------------------------------------------------

def tune_risk_threshold(
    y_true_bucket: np.ndarray,
    oof_delay_risk: np.ndarray,
    oof_delayed_bucket: np.ndarray,
) -> tuple[float, float]:
    """Choose the risk threshold that maximizes out-of-fold bucket accuracy."""
    valid = ~np.isnan(oof_delay_risk)
    if valid.sum() == 0:
        return 0.50, float("nan")

    best_threshold = 0.50
    best_accuracy = -np.inf
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = np.where(
            oof_delay_risk[valid] >= threshold,
            oof_delayed_bucket[valid],
            0,
        ).astype(int)
        acc = accuracy_score(y_true_bucket[valid], pred)
        if acc > best_accuracy or (
            np.isclose(acc, best_accuracy) and threshold > best_threshold
        ):
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold, float(best_accuracy)


def run_time_series_cv(
    X_train: pd.DataFrame,
    y_train_bucket: np.ndarray,
    y_train_binary: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    n_splits: int = 4,
) -> tuple[float, pd.DataFrame]:
    """Run expanding-window CV and tune the hybrid risk threshold."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_delay_risk = np.full(len(X_train), np.nan, dtype=float)
    oof_delayed_bucket = np.full(len(X_train), np.nan, dtype=float)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_bucket_fold_train = y_train_bucket[train_idx]
        y_bucket_fold_val = y_train_bucket[val_idx]
        y_binary_fold_train = y_train_binary[train_idx]
        y_binary_fold_val = y_train_binary[val_idx]

        binary_model = build_binary_delay_model(numeric_features, categorical_features)
        bucket_model = build_bucket_model(numeric_features, categorical_features)

        binary_model.fit(X_fold_train, y_binary_fold_train)
        bucket_model.fit(X_fold_train, y_bucket_fold_train)

        delay_risk = binary_model.predict_proba(X_fold_val)[:, 1]
        raw_bucket_probs = full_bucket_proba(bucket_model, X_fold_val)
        bucket_probs = combine_risk_and_bucket_probs(delay_risk, raw_bucket_probs)
        bucket_argmax = bucket_probs.argmax(axis=1)
        delayed_bucket = raw_bucket_probs.argmax(axis=1)
        delayed_bucket = np.where(delayed_bucket == 0, 1, delayed_bucket)

        oof_delay_risk[val_idx] = delay_risk
        oof_delayed_bucket[val_idx] = delayed_bucket

        baseline = np.zeros(len(val_idx), dtype=int)
        try:
            roc_auc = roc_auc_score(y_binary_fold_val, delay_risk)
        except ValueError:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_binary_fold_val, delay_risk)
        except ValueError:
            pr_auc = np.nan

        rows.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "val_start": int(val_idx[0]),
            "val_end": int(val_idx[-1]),
            "baseline_accuracy": accuracy_score(y_bucket_fold_val, baseline),
            "bucket_argmax_accuracy": accuracy_score(y_bucket_fold_val, bucket_argmax),
            "bucket_argmax_macro_f1": f1_score(
                y_bucket_fold_val,
                bucket_argmax,
                average="macro",
                zero_division=0,
            ),
            "binary_roc_auc": roc_auc,
            "binary_pr_auc": pr_auc,
        })

    threshold, oof_acc = tune_risk_threshold(
        y_train_bucket,
        oof_delay_risk,
        oof_delayed_bucket,
    )
    cv_df = pd.DataFrame(rows)
    cv_df.attrs["risk_threshold"] = threshold
    cv_df.attrs["oof_hybrid_accuracy"] = oof_acc
    return threshold, cv_df


# ---------------------------------------------------------------------------
# 14. EXPLANATION HELPERS
# ---------------------------------------------------------------------------

def _bar(prob: float, width: int = 20) -> str:
    """ASCII progress bar for a probability value."""
    filled = int(round(float(prob) * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {prob:.0%}"


def format_distribution(probs: np.ndarray) -> str:
    """Multi-line bucket distribution string."""
    lines = []
    for i, label in BUCKET_LABELS.items():
        lines.append(f"    {label:28s} {_bar(probs[i])}")
    return "\n".join(lines)


def explain_flight(
    row: pd.Series,
    probs: np.ndarray,
    predicted_delay_min: float,
    delay_risk: float,
) -> str:
    """Generate a concise operations briefing for a high-risk flight."""
    if delay_risk >= 0.60:
        priority = "HIGH"
    elif delay_risk >= 0.35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    lines = [
        f"  {'-' * 60}",
        f"  Priority: {priority}",
        f"  Flight:   {row.get('flight_id', 'N/A')} | Route: {row.get('origin', '?')} -> {row.get('dest', '?')}",
        f"  Departs:  {row.get('sched_dep_local', '?')} | Predicted delay: {predicted_delay_min:.0f} min",
        f"  Delay risk (P(>=15 min late)): {delay_risk:.0%}",
        "",
        "  Estimated delay distribution:",
        format_distribution(probs),
        "",
        "  Reasons flagged:",
    ]

    reasons = []
    actions = []

    prev_arr = float(row.get("prev_arr_delay_min", 0) or 0)
    inbound_after = float(row.get("inbound_delay_after_buffer_min", 0) or 0)
    turnaround = row.get("turnaround_buffer_min", None)
    tight = int(row.get("tight_turnaround", 0) or 0)
    snowfall = float(row.get("snowfall_cm", 0) or 0)
    precip = float(row.get("precip_mm", 0) or 0)
    gusts = float(row.get("wind_gust_kmh", 0) or 0)
    fog = int(row.get("is_fog_code", 0) or 0)
    thunder = int(row.get("is_thunderstorm_code", 0) or 0)
    congestion_pctile = float(row.get("origin_hour_congestion_pctile", 0) or 0)
    origin_delay_rate = float(row.get("origin_recent_delay_rate", 0) or 0)
    carrier_delay_rate = float(row.get("carrier_recent_delay_rate", 0) or 0)
    evening = int(row.get("is_evening_bank", 0) or 0)

    if prev_arr >= 30:
        reasons.append(f"    Late inbound aircraft: +{prev_arr:.0f} min on previous leg")
        actions.append("    Consider aircraft swap or standby crew to contain cascade.")
    elif prev_arr >= 15:
        reasons.append(f"    Moderately late inbound: +{prev_arr:.0f} min")
        actions.append("    Monitor cascade risk and flag connecting passengers.")

    if tight and turnaround is not None:
        reasons.append(
            f"    Tight turnaround: only {float(turnaround):.0f} min between inbound arrival and pushback"
        )
        actions.append("    Prioritize ground crew for fast turn; alert gate team.")

    if inbound_after >= 15:
        reasons.append(
            f"    Unrecovered inbound delay after buffer: +{inbound_after:.0f} min"
        )
        actions.append("    Proactively adjust downstream schedule.")

    if snowfall >= 1.0:
        reasons.append(f"    Snowfall: {snowfall:.1f} cm; de-icing and ground ops slowed")
        actions.append("    Check de-icing queue capacity and add ground-time buffer.")

    if precip >= 2.0:
        reasons.append(f"    Precipitation: {precip:.1f} mm; ramp operations degraded")

    if gusts >= 50:
        reasons.append(f"    High gusts: {gusts:.0f} km/h; ground handling speed limited")
        actions.append("    Check pushback restrictions and alert tow crews.")

    if fog:
        reasons.append("    Fog detected; low-visibility procedures may apply")
        actions.append("    Verify ILS categories and LVP status with ATC.")

    if thunder:
        reasons.append("    Thunderstorm signal; ground stop or flow restrictions possible")
        actions.append("    Alert ramp team and prepare for potential ground stop.")

    if congestion_pctile >= 0.85:
        reasons.append("    High origin congestion slot; NAS delays more likely")

    if origin_delay_rate >= 0.30:
        reasons.append(
            f"    Recent origin delay rate: {origin_delay_rate:.0%} of recent departures late"
        )

    if carrier_delay_rate >= 0.30:
        reasons.append(
            f"    Recent carrier delay rate: {carrier_delay_rate:.0%} across fleet"
        )

    if evening:
        reasons.append("    Evening bank: congestion and cascade risk elevated")

    if not reasons:
        reasons.append("    No single dominant signal; risk comes from combined weak factors")

    lines += reasons
    lines.append("")
    if actions:
        lines.append("  Recommended actions:")
        lines += list(dict.fromkeys(actions))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 15. MAIN PIPELINE
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cross-validated flight-delay prediction pipeline."
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to flights_weather_sample.csv",
    )
    parser.add_argument(
        "--output",
        default="delay_cv_scored_flights.csv",
        help="Output scored CSV path",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="Number of expanding-window TimeSeriesSplit folds",
    )
    args, unknown = parser.parse_known_args(argv)

    # Jupyter/ipykernel injects "-f <kernel.json>" into sys.argv. Ignore that
    # when the script is called from a notebook, but keep normal CLI validation.
    remaining = []
    i = 0
    while i < len(unknown):
        if unknown[i] == "-f" and i + 1 < len(unknown) and unknown[i + 1].endswith(".json"):
            i += 2
        else:
            remaining.append(unknown[i])
            i += 1

    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    print("=" * 72)
    print("  DATAbility 'Ready for Takeoff' - Cross-Validated Delay Prediction")
    print("=" * 72)

    print("\n[1/8] Loading dataset ...")
    raw = load_data(args.data)
    require_columns(
        raw,
        [
            "flight_id",
            "date",
            "dep_delay_min",
            "cancelled",
            "tail_number",
            "origin",
            "dest",
            "carrier",
        ],
    )
    print(f"      Loaded {len(raw):,} rows, {len(raw.columns)} columns.")

    print("[2/8] Engineering features ...")
    df = build_feature_table(raw)
    numeric_features, categorical_features = get_feature_columns()

    print("[3/8] Filtering rows ...")
    if "cancelled" in df.columns:
        df_model = df[df["cancelled"] != 1].copy()
        print(f"      Dropped {(df['cancelled'] == 1).sum():,} cancelled flights.")
    else:
        df_model = df.copy()

    df_model["dep_delay_min"] = pd.to_numeric(
        df_model["dep_delay_min"],
        errors="coerce",
    )
    df_model = df_model.dropna(subset=["dep_delay_min"])
    df_model = df_model.sort_values("sched_dep_dt").reset_index(drop=True)
    print(f"      Rows for modelling: {len(df_model):,}")

    y_minutes = df_model["dep_delay_min"].values.clip(min=0)
    y_log = np.log1p(y_minutes)
    y_buckets = delay_minutes_to_bucket_array(y_minutes)
    y_binary = (y_minutes >= 15).astype(int)
    X = df_model[numeric_features + categorical_features].copy()

    print("\n[4/8] Dataset summary")
    print(f"      Total modelling rows : {len(df_model):,}")
    print("\n      Bucket distribution (full dataset):")
    unique, counts = np.unique(y_buckets, return_counts=True)
    for bucket, count in zip(unique, counts):
        pct = count / len(y_buckets) * 100
        print(f"        Bucket {bucket} - {BUCKET_LABELS[bucket]:28s}: {count:4d} ({pct:.1f}%)")

    n = len(df_model)
    cutoff = int(n * 0.80)
    train_df = df_model.iloc[:cutoff]
    test_df = df_model.iloc[cutoff:]

    X_train = X.iloc[:cutoff]
    X_test = X.iloc[cutoff:]
    y_train_minutes = y_minutes[:cutoff]
    y_test_minutes = y_minutes[cutoff:]
    y_train_log = y_log[:cutoff]
    y_train_buckets = y_buckets[:cutoff]
    y_test_buckets = y_buckets[cutoff:]
    y_train_binary = y_binary[:cutoff]
    y_test_binary = y_binary[cutoff:]

    print(f"\n      Train rows : {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"      Split date : {train_df['date'].iloc[-1]} -> {test_df['date'].iloc[0]}")

    baseline_preds = np.zeros(len(y_test_buckets), dtype=int)
    baseline_acc = accuracy_score(y_test_buckets, baseline_preds)
    print(f"\n      Baseline (always bucket 0) accuracy: {baseline_acc:.3f}")

    print("\n[5/8] Time-series cross-validation ...")
    print("      Folds use expanding windows; no random future-to-past leakage.")
    risk_threshold, cv_df = run_time_series_cv(
        X_train,
        y_train_buckets,
        y_train_binary,
        numeric_features,
        categorical_features,
        n_splits=args.cv_splits,
    )
    for _, row in cv_df.iterrows():
        print(
            f"      Fold {int(row['fold'])}: "
            f"base_acc={row['baseline_accuracy']:.3f}, "
            f"bucket_acc={row['bucket_argmax_accuracy']:.3f}, "
            f"macro_f1={row['bucket_argmax_macro_f1']:.3f}, "
            f"risk_ROC={row['binary_roc_auc']:.3f}, "
            f"risk_PR={row['binary_pr_auc']:.3f}"
        )
    print(
        f"      Tuned risk threshold: {risk_threshold:.3f} "
        f"(OOF hybrid accuracy: {cv_df.attrs['oof_hybrid_accuracy']:.3f})"
    )

    print("\n[6/8] Training final models ...")
    print("      Binary risk model + bucket model + minute regressor")
    binary_model = build_binary_delay_model(numeric_features, categorical_features)
    bucket_model = build_bucket_model(numeric_features, categorical_features)
    regression_model = build_regression_model(numeric_features, categorical_features)

    binary_model.fit(X_train, y_train_binary)
    bucket_model.fit(X_train, y_train_buckets)
    regression_model.fit(X_train, y_train_log)
    print("      Training complete.")

    delay_risk = binary_model.predict_proba(X_test)[:, 1]
    raw_bucket_probs = full_bucket_proba(bucket_model, X_test)
    bucket_probs = combine_risk_and_bucket_probs(delay_risk, raw_bucket_probs)
    pred_buckets = predict_hybrid_buckets(delay_risk, raw_bucket_probs, risk_threshold)
    pred_minutes = np.expm1(regression_model.predict(X_test)).clip(min=0)
    reg_bucket_probs = regression_bucket_proba(regression_model, X_test)

    print("\n[7/8] Evaluation")
    print("\n  - Regression Metrics -")
    mae = mean_absolute_error(y_test_minutes, pred_minutes)
    rmse = np.sqrt(mean_squared_error(y_test_minutes, pred_minutes))
    print(f"      MAE  : {mae:.2f} min")
    print(f"      RMSE : {rmse:.2f} min")

    print("\n  - Bucket Classification Metrics -")
    bucket_acc = accuracy_score(y_test_buckets, pred_buckets)
    bucket_bal_acc = balanced_accuracy_score(y_test_buckets, pred_buckets)
    bucket_macro_f1 = f1_score(
        y_test_buckets,
        pred_buckets,
        average="macro",
        zero_division=0,
    )
    print(f"      Accuracy          : {bucket_acc:.3f} (baseline: {baseline_acc:.3f})")
    print(f"      Balanced accuracy : {bucket_bal_acc:.3f}")
    print(f"      Macro F1          : {bucket_macro_f1:.3f}")
    print("\n  Classification Report:")
    print(classification_report(
        y_test_buckets,
        pred_buckets,
        labels=[0, 1, 2, 3, 4],
        target_names=[f"B{i}" for i in range(N_BUCKETS)],
        zero_division=0,
    ))

    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test_buckets, pred_buckets, labels=[0, 1, 2, 3, 4])
    print("      " + "  ".join(f"B{i}" for i in range(N_BUCKETS)))
    for i, row_vals in enumerate(cm):
        print(f"  B{i}  " + "  ".join(f"{v:4d}" for v in row_vals))

    print("\n  - Binary Delay-Risk Metrics (delayed >=15 min) -")
    try:
        roc_auc = roc_auc_score(y_test_binary, delay_risk)
        print(f"      ROC-AUC : {roc_auc:.4f}")
    except ValueError as exc:
        print(f"      ROC-AUC could not be computed: {exc}")

    try:
        pr_auc = average_precision_score(y_test_binary, delay_risk)
        print(f"      PR-AUC  : {pr_auc:.4f}")
    except ValueError as exc:
        print(f"      PR-AUC could not be computed: {exc}")

    print("\n[8/8] Saving scored CSV ...")
    scored = test_df.copy().reset_index(drop=True)
    scored["predicted_delay_min"] = pred_minutes.round(1)
    scored["predicted_delay_bucket"] = pred_buckets
    scored["delay_risk"] = delay_risk.round(4)
    scored[BUCKET_COLS] = pd.DataFrame(
        bucket_probs,
        columns=BUCKET_COLS,
    ).round(4).values
    scored[REG_BUCKET_COLS] = pd.DataFrame(
        reg_bucket_probs,
        columns=REG_BUCKET_COLS,
    ).round(4).values
    scored["actual_delay_bucket"] = y_test_buckets
    scored["risk_threshold_used"] = round(risk_threshold, 4)

    output_cols = [
        "flight_id",
        "date",
        "carrier",
        "origin",
        "dest",
        "sched_dep_local",
        "dep_hour",
        "predicted_delay_min",
        "predicted_delay_bucket",
        "actual_delay_bucket",
        "dep_delay_min",
        "delay_risk",
        "risk_threshold_used",
    ] + BUCKET_COLS + [
        "reg_prob_on_time",
        "reg_prob_15_30_min",
        "reg_prob_30_60_min",
        "reg_prob_60_90_min",
        "reg_prob_90plus_min",
        "prev_arr_delay_min",
        "inbound_delay_after_buffer_min",
        "had_late_inbound",
        "tight_turnaround",
        "turnaround_buffer_min",
        "weather_severity_score",
        "snowfall_cm",
        "precip_mm",
        "wind_gust_kmh",
        "is_fog_code",
        "is_thunderstorm_code",
        "origin_recent_delay_rate",
        "carrier_recent_delay_rate",
        "route_recent_delay_rate",
        "is_evening_bank",
        "origin_hour_congestion_pctile",
        "date_ordinal",
    ]
    output_cols = [c for c in output_cols if c in scored.columns]

    out_df = scored[output_cols].sort_values("delay_risk", ascending=False)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"      Saved {len(out_df):,} rows to '{out_path}'")

    print("\n" + "=" * 72)
    print("  TOP 10 HIGHEST-RISK FLIGHTS - Operations Controller Briefing")
    print("=" * 72)

    top10_idx = delay_risk.argsort()[-10:][::-1]
    for rank, idx in enumerate(top10_idx, start=1):
        row = scored.iloc[idx]
        print(f"\n  #{rank}")
        print(explain_flight(row, bucket_probs[idx], pred_minutes[idx], delay_risk[idx]))

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
