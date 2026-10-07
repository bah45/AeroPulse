# AEROPULSE — Know Your Air. Predict Tomorrow.

## Status (read this first)
This is the **ML + data + schema foundation**, not the full platform.

Implemented:
- Open-Meteo provider (hourly pollutants + weather). Open-Meteo air quality is CAMS **model** output, so rows are labelled `modeled`, never `observed`.
- Cleaning pipeline: dedupe, sort, hourly resample, impossible-value nulling, interpolation of gaps <=3h only, future-row removal, quality report.
- Leakage-safe features: lags, rolling stats, calendar + cyclical encodings, weather, other pollutants, missingness count.
- Per-horizon training (6/24/72/168h, direct method): persistence baseline, Ridge, Random Forest, HistGradientBoosting. Chronological 70/15/15 split, selection on validation MAE, MAE/RMSE/R2/MAPE on test. Persistence is selected if no ML model beats it.
- Prediction range = empirical 10-90% quantiles of validation residuals (not a confidence interval).
- Readiness checks return explicit `unavailable` responses. Append-only model registry (joblib + registry.jsonl).
- FastAPI: GET /health, GET|POST /forecast (lat/lon), GET /model-info, GET /metrics, POST /train (X-Admin-Token).
- database/schema.sql: all 16 tables, RLS on user-owned tables.

Not built yet: Next.js frontend (liquid-glass UI, map, charts, AEROBOT), Next.js API routes, OpenAQ/CPCB providers, anomaly detection, environmental/compliance sources, auth, alerts, exports, E2E tests, Docker, ingestion/evaluation scripts.

## Run
```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python ../scripts/train_models.py --lat 13.08 --lon 80.27
ADMIN_TOKEN=change-me uvicorn app.main:app --reload
curl "http://localhost:8000/forecast?lat=13.08&lon=80.27&horizon_hours=24"
```
Limitations: inputs are modeled (CAMS), so forecasts inherit its error; provider history is capped at 92 days; 168h forecasts often won't beat persistence.
