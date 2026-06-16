# Flight Delay Ops Dashboard

Local operations dashboard for the DATAbility flight-delay project.

## What Is Included

- `app.py` - local Python server and API.
- `frontend/` - dashboard UI.
- `logic/flight_delay_prediction_cv.py` - Python prediction pipeline.
- `data/flights_weather_sample.csv` - source flight/weather dataset.
- `data/delay_cv_scored_flights.csv` - latest scored output.
- `notebooks/RFSixtyFiveAccuracy.ipynb` - source notebook from the project files.

## Run

```bash
cd /Users/alizhanaskarov/Documents/Codex/2026-06-16/files-mentioned-by-the-user-build/outputs/flight-delay-ops-dashboard
/Users/alizhanaskarov/miniconda3/bin/python app.py --port 8765
```

Open:

```text
http://localhost:8765
```

## Use

- The dashboard loads the latest scored flights immediately.
- Use filters to narrow by risk, origin, carrier, destination, or departure hour.
- Click a flight row to see the controller brief.
- `Run` re-runs the Python scoring pipeline on the bundled CSV.
- `CSV` lets you select a different CSV, then `Run` scores that file.
- The download button exports the current scored CSV.

## API

```text
GET  /api/results
POST /api/run-prediction
GET  /api/download
GET  /api/health
```

The UI keeps model details out of the controller view and focuses on risk, likely delay range, reasons, and recommended action.
