#!/usr/bin/env python3
"""
Local flight-delay operations dashboard.

Run:
    python3 app.py --port 8765

Then open:
    http://localhost:8765
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
LOGIC_PATH = BASE_DIR / "logic" / "flight_delay_prediction_cv.py"

DEFAULT_DATA_PATH = DATA_DIR / "flights_weather_sample.csv"
UPLOADED_DATA_PATH = DATA_DIR / "uploaded_flights_weather_sample.csv"
SCORED_PATH = DATA_DIR / "delay_cv_scored_flights.csv"

BUCKET_LABELS = {
    0: "On time",
    1: "15-30 min",
    2: "30-60 min",
    3: "60-90 min",
    4: "90+ min",
}

BUCKET_REPRESENTATIVE_MINUTES = {
    0: 7,
    1: 22,
    2: 45,
    3: 75,
    4: 105,
}


def risk_level(score: float) -> str:
    if score >= 0.60:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def pct(value) -> int:
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return 0


def minutes(value) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def delay_range_from_minutes(value) -> str:
    predicted = minutes(value)
    if predicted < 15:
        return "On time"
    if predicted < 30:
        return "15-30 min"
    if predicted < 60:
        return "30-60 min"
    if predicted < 90:
        return "60-90 min"
    return "90+ min"


def probability_value(row: dict, primary: str, fallback: str) -> float:
    value = row.get(primary)
    if value is None:
        value = row.get(fallback)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def distribution_delay_estimate(row: dict, fallback_minutes: int) -> tuple[int, str]:
    """Choose a single user-facing delay estimate from the bucket distribution."""
    probs = [
        probability_value(row, "reg_prob_on_time", "prob_on_time"),
        probability_value(row, "reg_prob_15_30_min", "prob_15_30_min"),
        probability_value(row, "reg_prob_30_60_min", "prob_30_60_min"),
        probability_value(row, "reg_prob_60_90_min", "prob_60_90_min"),
        probability_value(row, "reg_prob_90plus_min", "prob_90plus_min"),
    ]
    total = sum(probs)
    if total <= 0:
        return fallback_minutes, delay_range_from_minutes(fallback_minutes)

    cumulative = 0.0
    for bucket, prob in enumerate(probs):
        cumulative += prob / total
        if cumulative >= 0.5:
            return BUCKET_REPRESENTATIVE_MINUTES[bucket], BUCKET_LABELS[bucket]

    return BUCKET_REPRESENTATIVE_MINUTES[4], BUCKET_LABELS[4]


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def recommend_action(row: dict) -> dict:
    """Controller-facing recommendation rules, ordered by operational priority."""
    inbound_after = as_float(row.get("inbound_delay_after_buffer_min"))
    cascade_count = int(as_float(row.get("cascade_count")))
    turnaround = as_float(row.get("turnaround_buffer_min"), default=9999.0)
    weather_score = as_float(row.get("weather_severity_score"))
    snow = as_float(row.get("snowfall_cm")) > 0
    fog = int(as_float(row.get("is_fog_code"))) == 1
    thunderstorm = int(as_float(row.get("is_thunderstorm_code"))) == 1
    wind_gust_high = as_float(row.get("wind_gust_kmh")) >= 50
    origin_delay_rate_recent = as_float(row.get("origin_recent_delay_rate"))
    carrier_delay_rate_recent = as_float(row.get("carrier_recent_delay_rate"))
    route_delay_rate_recent = as_float(row.get("route_recent_delay_rate"))
    origin_hour_departure_count = as_float(row.get("departures_origin_hour"))
    high_congestion_threshold = as_float(row.get("high_congestion_threshold"), default=0.0)
    congestion_pctile = as_float(row.get("origin_hour_congestion_pctile"))
    departure_hour = int(as_float(row.get("dep_hour")))
    delay_risk_score = as_float(row.get("delay_risk")) * 100

    moderate_driver_count = sum([
        inbound_after > 0,
        origin_delay_rate_recent >= 0.25,
        carrier_delay_rate_recent >= 0.25,
        route_delay_rate_recent >= 0.25,
        congestion_pctile >= 0.75,
        weather_score >= 2 or snow or fog or thunderstorm or wind_gust_high,
        departure_hour >= 17,
    ])

    if inbound_after > 20 and cascade_count >= 2:
        return {
            "title": "Protect aircraft rotation",
            "action": "Alert ops desk and downstream station, track inbound aircraft arrival, and prepare recovery options.",
            "reason": "Inbound delay exceeds recovery buffer and aircraft has downstream flights.",
            "escalation": "Escalate if aircraft has not landed 45 minutes before scheduled departure.",
        }

    if inbound_after > 0 and turnaround < 30:
        return {
            "title": "Trigger turnaround priority",
            "action": "Pre-alert ramp, cleaning, fueling, catering and baggage teams before aircraft arrival.",
            "reason": "Recovery buffer is too small to absorb inbound delay.",
            "escalation": "Escalate if available turnaround time falls below minimum turnaround threshold.",
        }

    # Catches any remaining positive carry-over (e.g. large buffer that still
    # cannot fully absorb a very late inbound aircraft).  Without this branch
    # such flights would fall through to the weather or pressure checks even
    # when the inbound delay is the dominant operational driver.
    if inbound_after > 0:
        return {
            "title": "Track inbound carry-over",
            "action": "Monitor inbound aircraft arrival closely, coordinate early ramp handoff, and alert the gate team to the expected late arrival.",
            "reason": "Inbound delay is not fully absorbed by the scheduled ground time and will carry over to this departure.",
            "escalation": "Escalate if the carry-over grows or downstream connections become at risk.",
        }

    if weather_score >= 70 or snow or fog or thunderstorm or wind_gust_high:
        return {
            "title": "Activate weather disruption protocol",
            "action": "Confirm de-icing or ramp limitations, monitor taxi/runway flow, and coordinate with weather or ATC desk.",
            "reason": "Severe weather risk detected.",
            "escalation": "Escalate if severe weather persists within the departure window.",
        }

    if origin_delay_rate_recent >= 0.35:
        return {
            "title": "Monitor origin flow restrictions",
            "action": "Watch ATC restrictions, airport departure flow and close-in flights from this origin.",
            "reason": "Recent departures from this airport show elevated delay pressure.",
            "escalation": "Escalate if origin delay pressure remains high 60 minutes before departure.",
        }

    if carrier_delay_rate_recent >= 0.35:
        return {
            "title": "Check carrier recovery capacity",
            "action": "Check aircraft and crew availability and protect priority flights.",
            "reason": "Carrier's recent flights show elevated delay pressure.",
            "escalation": "Escalate if carrier pressure combines with inbound or weather risk.",
        }

    if route_delay_rate_recent >= 0.35:
        return {
            "title": "Monitor route-specific disruption",
            "action": "Check destination status, route flow and recent similar flights.",
            "reason": "This origin-destination route has elevated recent delay risk.",
            "escalation": "Escalate if route risk combines with weather or inbound aircraft delay.",
        }

    if (
        high_congestion_threshold > 0
        and origin_hour_departure_count >= high_congestion_threshold
    ) or congestion_pctile >= 0.85:
        return {
            "title": "Secure pushback readiness",
            "action": "Coordinate gate and ramp timing to keep the flight ready before slot pressure increases.",
            "reason": "Flight departs during a congested origin-hour window.",
            "escalation": "Escalate if readiness drops below threshold during peak departure bank.",
        }

    if departure_hour >= 17 and delay_risk_score >= 60:
        return {
            "title": "Prevent end-of-day stack-up",
            "action": "Check rotation risk, crew legality exposure and last-leg connections earlier.",
            "reason": "Evening flights have less recovery slack.",
            "escalation": "Escalate severe-risk evening departures earlier.",
        }

    if moderate_driver_count >= 3:
        return {
            "title": "Place on active watchlist",
            "action": "Refresh risk closer to departure and monitor inbound aircraft and gate readiness.",
            "reason": "Multiple weak signals are accumulating without one dominant cause.",
            "escalation": "Escalate only if another signal worsens.",
        }

    return {
        "title": "Continue normal monitoring",
        "action": "No immediate dispatcher action required.",
        "reason": "No strong operational risk pattern detected.",
        "escalation": "Reassess at next scheduled update.",
    }


def build_user_brief(row: dict) -> dict:
    impacts: list[str] = []
    actions: list[str] = []
    risk_factors: list[dict] = []
    absorbed_inbound_factor = None
    unknown_inbound_factor = None

    prev_arr = float(row.get("prev_arr_delay_min") or 0)
    turnaround = row.get("turnaround_buffer_min")
    inbound_after_raw = row.get("inbound_delay_after_buffer_min")
    inbound_after = float(inbound_after_raw or 0)
    buffer_text = f"{minutes(turnaround)} min" if turnaround is not None else "unavailable"
    carryover_text = f"{minutes(inbound_after)} min" if inbound_after_raw is not None else "unknown"
    tight_turnaround = int(row.get("tight_turnaround") or 0)
    snow = float(row.get("snowfall_cm") or 0)
    rain = float(row.get("precip_mm") or 0)
    gust = float(row.get("wind_gust_kmh") or 0)
    fog = int(row.get("is_fog_code") or 0)
    thunder = int(row.get("is_thunderstorm_code") or 0)
    origin_rate = float(row.get("origin_recent_delay_rate") or 0)
    carrier_rate = float(row.get("carrier_recent_delay_rate") or 0)
    route_rate = float(row.get("route_recent_delay_rate") or 0)
    congestion = float(row.get("origin_hour_congestion_pctile") or 0)
    evening = int(row.get("is_evening_bank") or 0)
    missing_buffer = turnaround is None or inbound_after_raw is None

    if prev_arr >= 15:
        inbound_factor = {
            "label": "Inbound aircraft",
            "value": f"Prev leg {minutes(prev_arr)} min late",
            "level": "High" if prev_arr >= 30 or inbound_after >= 15 else "Medium",
            "detail": "Buffer is the scheduled ground time available before this flight departs.",
            "metrics": [
                {"label": "Previous leg", "value": f"{minutes(prev_arr)} min late"},
                {"label": "Buffer", "value": buffer_text},
                {"label": "Carry-over", "value": carryover_text},
            ],
        }

        if turnaround is not None and inbound_after_raw is not None and inbound_after <= 0:
            absorbed_inbound_factor = {
                **inbound_factor,
                "value": "Absorbed by buffer",
                "level": "Low",
                "detail": "The previous leg was late, but scheduled ground time is larger than that delay.",
            }
        elif missing_buffer:
            unknown_inbound_factor = {
                **inbound_factor,
                "value": "Buffer unavailable",
                "level": "Low",
                "detail": "The previous leg was late, but the dataset does not provide a reliable buffer, so carry-over cannot be confirmed.",
            }
        elif inbound_after >= 15:
            impacts.append(
                "The departure is exposed to a cascade delay because the aircraft may reach the gate late."
            )
            actions.append("Check whether an aircraft swap or reserve crew can protect the departure.")
            risk_factors.append(inbound_factor)
        else:
            impacts.append(
                "The flight may lose boarding and pushback buffer if the inbound aircraft slips further."
            )
            actions.append("Monitor inbound arrival and alert the gate team.")
            risk_factors.append(inbound_factor)

    if inbound_after >= 15:
        impacts.append(
            "The planned ground time may not be enough to recover before scheduled departure."
        )
        actions.append("Prioritize ground handling and update the downstream station.")
        risk_factors.append({
            "label": "Turnaround buffer",
            "value": f"{minutes(inbound_after)} min carry-over",
            "level": "High",
            "detail": "The planned ground time may not absorb the inbound delay.",
        })

    if tight_turnaround and turnaround is not None:
        impacts.append(
            "A small ground-time buffer leaves little room for cleaning, fueling, boarding, or crew handoff issues."
        )
        actions.append("Assign ground crew early and keep boarding plan ready.")
        risk_factors.append({
            "label": "Turnaround",
            "value": f"{minutes(turnaround)} min",
            "level": "Medium",
            "detail": "There is limited room to recover before pushback.",
        })

    weather_bits = []
    if snow >= 1:
        weather_bits.append(f"snow {snow:.1f} cm")
    if rain >= 2:
        weather_bits.append(f"precipitation {rain:.1f} mm")
    if gust >= 50:
        weather_bits.append(f"gusts {minutes(gust)} km/h")
    if fog:
        weather_bits.append("fog")
    if thunder:
        weather_bits.append("thunderstorm signal")
    if weather_bits:
        impacts.append(
            "Weather can slow ramp work, de-icing, taxi flow, or airport acceptance rates."
        )
        actions.append("Confirm weather procedures, de-icing needs, and ramp capacity.")
        risk_factors.append({
            "label": "Weather",
            "value": ", ".join(weather_bits),
            "level": "Medium",
            "detail": "Weather can slow ramp, de-icing, and airport flow.",
        })

    if congestion >= 0.85:
        impacts.append(
            "A late pushback is harder to recover during a crowded departure window."
        )
        actions.append("Watch ATC flow restrictions and keep passengers informed early.")
        risk_factors.append({
            "label": "Airport congestion",
            "value": "Busy slot",
            "level": "Medium",
            "detail": "The flight departs during a crowded origin-airport window.",
        })
    elif evening:
        impacts.append(
            "End-of-day delays can stack up, reducing spare aircraft, crew, and rebooking flexibility."
        )
        risk_factors.append({
            "label": "Schedule timing",
            "value": "Evening bank",
            "level": "Medium",
            "detail": "End-of-day schedules are more exposed to accumulated delay.",
        })

    if origin_rate >= 0.35:
        impacts.append(
            "The airport is currently running hot, so even flights without a single obvious issue deserve attention."
        )
        risk_factors.append({
            "label": "Origin pressure",
            "value": f"{pct(origin_rate)}% recently late",
            "level": "High" if origin_rate >= 0.60 else "Medium",
            "detail": "Recent flights from this airport have been running late.",
        })
    if carrier_rate >= 0.35:
        impacts.append(
            "Carrier-side pressure can mean less slack in aircraft rotation, crews, or gate recovery."
        )
        risk_factors.append({
            "label": "Carrier pressure",
            "value": f"{pct(carrier_rate)}% recently late",
            "level": "High" if carrier_rate >= 0.60 else "Medium",
            "detail": "This carrier's recent flights show delay pressure.",
        })
    if route_rate >= 0.35:
        impacts.append(
            "This route has recent delay history, so the controller should watch it earlier than a normal departure."
        )
        risk_factors.append({
            "label": "Route pattern",
            "value": f"{pct(route_rate)}% recently late",
            "level": "High" if route_rate >= 0.60 else "Medium",
            "detail": "This origin-destination pair has recent delay history.",
        })

    if absorbed_inbound_factor:
        impacts.append(
            "The previous-leg delay appears covered by scheduled ground time; remaining risk is driven more by airport, route, carrier, or timing pressure."
        )
        risk_factors.append(absorbed_inbound_factor)

    if unknown_inbound_factor:
        impacts.append(
            "The previous leg was late, but buffer is unavailable, so it is a watch item rather than a confirmed carry-over delay."
        )
        risk_factors.append(unknown_inbound_factor)

    if not impacts:
        impacts.append(
            "No single driver dominates; the risk comes from several smaller schedule and operating signals."
        )
        actions.append("Keep on the watchlist and refresh status before pushback.")
        risk_factors.append({
            "label": "Combined signals",
            "value": "No dominant driver",
            "level": "Low",
            "detail": "The score comes from several smaller schedule and operating signals.",
        })

    first_factor = risk_factors[0] if risk_factors else None
    main_reason = (
        f"{first_factor['label']}: {first_factor['value']}"
        if first_factor
        else impacts[0].rstrip(".")
    )
    return {
        "main_reason": main_reason,
        "reasons": list(dict.fromkeys(impacts))[:5],
        "recommended_actions": list(dict.fromkeys(actions))[:4],
        "controller_suggestion": recommend_action(row),
        "risk_factors": risk_factors[:6],
    }


def load_scored_rows(limit: int | None = None) -> dict:
    if not SCORED_PATH.exists():
        return {
            "summary": {
                "total": 0,
                "highRisk": 0,
                "mediumRisk": 0,
                "averageDelay": 0,
                "lastUpdated": None,
            },
            "filters": {"origins": [], "destinations": [], "carriers": []},
            "flights": [],
            "error": "No scored CSV found. Run prediction first.",
        }

    df = pd.read_csv(SCORED_PATH, low_memory=False)
    df = df.sort_values("delay_risk", ascending=False).reset_index(drop=True)
    if limit:
        df = df.head(limit)

    flights = []
    display_delay_values = []
    for _, series in df.iterrows():
        row = series.where(pd.notna(series), None).to_dict()
        brief = build_user_brief(row)
        bucket = int(row.get("predicted_delay_bucket") or 0)
        score = float(row.get("delay_risk") or 0)
        predicted_minutes = minutes(row.get("predicted_delay_min"))
        display_delay_minutes, display_delay_range = distribution_delay_estimate(
            row,
            predicted_minutes,
        )
        display_delay_values.append(display_delay_minutes)
        flight = {
            "flight_id": row.get("flight_id"),
            "date": row.get("date"),
            "carrier": row.get("carrier"),
            "origin": row.get("origin"),
            "dest": row.get("dest"),
            "route": f"{row.get('origin')} -> {row.get('dest')}",
            "sched_dep_local": row.get("sched_dep_local"),
            "dep_hour": row.get("dep_hour"),
            "predicted_delay_min": display_delay_minutes,
            "predicted_delay_range_label": display_delay_range,
            "regression_mean_delay_min": predicted_minutes,
            "predicted_delay_bucket": bucket,
            "predicted_delay_bucket_label": BUCKET_LABELS.get(bucket, "Unknown"),
            "actual_delay_bucket": row.get("actual_delay_bucket"),
            "actual_delay_min": row.get("dep_delay_min"),
            "delay_risk": round(score, 4),
            "delay_risk_pct": pct(score),
            "risk_level": risk_level(score),
            "probabilities": {
                "on_time": probability_value(row, "reg_prob_on_time", "prob_on_time"),
                "delay_15_30": probability_value(row, "reg_prob_15_30_min", "prob_15_30_min"),
                "delay_30_60": probability_value(row, "reg_prob_30_60_min", "prob_30_60_min"),
                "delay_60_90": probability_value(row, "reg_prob_60_90_min", "prob_60_90_min"),
                "delay_90_plus": probability_value(row, "reg_prob_90plus_min", "prob_90plus_min"),
            },
            **brief,
        }
        flights.append(flight)

    full = pd.read_csv(SCORED_PATH, low_memory=False)
    high = int((full["delay_risk"] >= 0.60).sum())
    medium = int(((full["delay_risk"] >= 0.30) & (full["delay_risk"] < 0.60)).sum())
    avg_delay = float(sum(display_delay_values) / len(display_delay_values)) if display_delay_values else 0
    busiest_origin = (
        full["origin"].value_counts().idxmax()
        if "origin" in full and len(full)
        else None
    )
    top_route = None
    if len(full):
        top = full.sort_values("delay_risk", ascending=False).iloc[0]
        top_route = f"{top.get('origin')} -> {top.get('dest')}"

    return {
        "summary": {
            "total": int(len(full)),
            "highRisk": high,
            "mediumRisk": medium,
            "averageDelay": round(avg_delay, 1),
            "busiestOrigin": busiest_origin,
            "topRiskRoute": top_route,
            "lastUpdated": SCORED_PATH.stat().st_mtime,
        },
        "filters": {
            "origins": sorted(full["origin"].dropna().unique().tolist()),
            "destinations": sorted(full["dest"].dropna().unique().tolist()),
            "carriers": sorted(full["carrier"].dropna().unique().tolist()),
        },
        "flights": flights,
    }


def run_prediction(csv_text: str | None = None) -> dict:
    data_path = DEFAULT_DATA_PATH
    if csv_text:
        UPLOADED_DATA_PATH.write_text(csv_text)
        data_path = UPLOADED_DATA_PATH

    spec = importlib.util.spec_from_file_location("flight_delay_prediction_cv", LOGIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load prediction pipeline.")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = pipeline.main([
            "--data",
            str(data_path),
            "--output",
            str(SCORED_PATH),
            "--cv-splits",
            "4",
        ])

    return {
        "exitCode": exit_code,
        "logTail": buffer.getvalue().splitlines()[-40:],
        **load_scored_rows(),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "FlightDelayOps/1.0"

    def log_message(self, format: str, *args) -> None:
        print("[server]", format % args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._send_file(FRONTEND_DIR / "index.html")
            return
        if path == "/api/results":
            params = parse_qs(parsed.query)
            limit = None
            if "limit" in params:
                try:
                    limit = int(params["limit"][0])
                except ValueError:
                    limit = None
            self._send_json(load_scored_rows(limit=limit))
            return
        if path == "/api/download":
            self._send_file(SCORED_PATH, "delay_cv_scored_flights.csv")
            return
        if path == "/api/health":
            self._send_json({
                "ok": True,
                "hasData": DEFAULT_DATA_PATH.exists(),
                "hasScoredOutput": SCORED_PATH.exists(),
            })
            return
        if path.startswith("/assets/"):
            candidate = (FRONTEND_DIR / path.lstrip("/")).resolve()
            if FRONTEND_DIR.resolve() not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            self._send_file(candidate)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run-prediction":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = {}
            if length:
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
            csv_text = payload.get("csvText")
            result = run_prediction(csv_text=csv_text)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the flight delay dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Flight delay dashboard running at http://{args.host}:{args.port}")
    print(f"Using data folder: {DATA_DIR}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
