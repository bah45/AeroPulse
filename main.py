import hmac
import os
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.ml import forecasting as fc
from app.ml.pipeline import InsufficientData
from app.ml.provider import ProviderError

app = FastAPI(title="AEROPULSE ML API", version="0.1.0")


class ForecastRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    horizon_hours: int = Field(default=24, ge=1, le=168)


@app.get("/health")
def health():
    return {"status": "ok", "horizons": fc.HORIZONS}


@app.get("/forecast")
def get_forecast(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180),
                 horizon_hours: int = Query(24, ge=1, le=168)):
    return fc.forecast(lat, lon, horizon_hours)


@app.post("/forecast")
def post_forecast(req: ForecastRequest):
    return fc.forecast(req.lat, req.lon, req.horizon_hours)


@app.get("/model-info")
def model_info(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    models = fc.latest_models(lat, lon)
    if not models:
        return {"status": "unavailable", "reason": "No trained model for this location"}
    return {"status": "success", "models": {h: {k: v for k, v in m.items() if k != "metrics"} for h, m in models.items()}}


@app.get("/metrics")
def get_metrics(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    models = fc.latest_models(lat, lon)
    return {"status": "success" if models else "unavailable",
            "metrics": {h: m["metrics"] for h, m in models.items()}}


@app.post("/train")
def train(req: ForecastRequest, x_admin_token: str = Header(default="")):
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(503, "ADMIN_TOKEN not configured")
    if not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Unauthorized")
    try:
        return fc.train_location(req.lat, req.lon)
    except InsufficientData as e:
        return {"status": "unavailable", "reason": str(e)}
    except ProviderError as e:
        raise HTTPException(502, str(e))
