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


def build_user_brief(row: dict) -> dict:
    impacts: list[str] = []
    actions: list[str] = []
    risk_factors: list[dict] = []

    prev_arr = float(row.get("prev_arr_delay_min") or 0)
    inbound_after = float(row.get("inbound_delay_after_buffer_min") or 0)
    turnaround = row.get("turnaround_buffer_min")
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

    if prev_arr >= 30:
        impacts.append(
            "The departure is exposed to a cascade delay because the aircraft may reach the gate late."
        )
        actions.append("Check whether an aircraft swap or reserve crew can protect the departure.")
        risk_factors.append({
            "label": "Inbound aircraft",
            "value": f"{minutes(prev_arr)} min late",
            "level": "High",
            "detail": "A late aircraft can carry delay into this departure.",
        })
    elif prev_arr >= 15:
        impacts.append(
            "The flight may lose boarding and pushback buffer if the inbound aircraft slips further."
        )
        actions.append("Monitor inbound arrival and alert the gate team.")
        risk_factors.append({
            "label": "Inbound aircraft",
            "value": f"{minutes(prev_arr)} min late",
            "level": "Medium",
            "detail": "The inbound leg is already behind schedule.",
        })

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
    for _, series in df.iterrows():
        row = series.where(pd.notna(series), None).to_dict()
        brief = build_user_brief(row)
        bucket = int(row.get("predicted_delay_bucket") or 0)
        score = float(row.get("delay_risk") or 0)
        flight = {
            "flight_id": row.get("flight_id"),
            "date": row.get("date"),
            "carrier": row.get("carrier"),
            "origin": row.get("origin"),
            "dest": row.get("dest"),
            "route": f"{row.get('origin')} -> {row.get('dest')}",
            "sched_dep_local": row.get("sched_dep_local"),
            "dep_hour": row.get("dep_hour"),
            "predicted_delay_min": minutes(row.get("predicted_delay_min")),
            "predicted_delay_bucket": bucket,
            "predicted_delay_bucket_label": BUCKET_LABELS.get(bucket, "Unknown"),
            "actual_delay_bucket": row.get("actual_delay_bucket"),
            "actual_delay_min": row.get("dep_delay_min"),
            "delay_risk": round(score, 4),
            "delay_risk_pct": pct(score),
            "risk_level": risk_level(score),
            "probabilities": {
                "on_time": float(row.get("prob_on_time") or 0),
                "delay_15_30": float(row.get("prob_15_30_min") or 0),
                "delay_30_60": float(row.get("prob_30_60_min") or 0),
                "delay_60_90": float(row.get("prob_60_90_min") or 0),
                "delay_90_plus": float(row.get("prob_90plus_min") or 0),
            },
            **brief,
        }
        flights.append(flight)

    full = pd.read_csv(SCORED_PATH, low_memory=False)
    high = int((full["delay_risk"] >= 0.60).sum())
    medium = int(((full["delay_risk"] >= 0.30) & (full["delay_risk"] < 0.60)).sum())
    avg_delay = float(full["predicted_delay_min"].mean()) if len(full) else 0
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
