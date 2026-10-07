"""Training, evaluation, registry and inference (direct multi-horizon: one model per horizon)."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import FEATURE_VERSION, InsufficientData, build_features, clean, supervised
from .provider import DATA_STATUS, PROVIDER, fetch_history

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
HORIZONS = [6, 24, 72, 168]
MIN_ROWS = 24 * 21         # >= 3 weeks of usable hourly samples
MAX_MISSING_FRAC = 0.2
MAX_LATEST_AGE_H = 3
PI_LOW, PI_HIGH = 0.10, 0.90


def location_key(lat: float, lon: float) -> str:
    return f"{lat:.2f}_{lon:.2f}"


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y >= 1.0  # MAPE only where the denominator is not ~0
    mape = float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.sum() >= 10 else None
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)), "mape": mape, "n": int(len(y))}


def _candidates():
    imp = lambda: SimpleImputer(strategy="median")  # fitted on train only, inside the pipeline
    return {
        "ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(imp(), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
    }


def chrono_split(n: int):
    a, b = int(n * 0.70), int(n * 0.85)  # never shuffled
    return slice(0, a), slice(a, b), slice(b, n)


def train_horizon(df: pd.DataFrame, target: str, h: int) -> dict:
    X, y = supervised(df, target, h)
    if len(X) < MIN_ROWS:
        raise InsufficientData(f"{len(X)} usable rows < {MIN_ROWS}")
    tr, va, te = chrono_split(len(X))
    base = lambda sl: X["now"].iloc[sl].to_numpy()  # persistence: prediction(t+h) = value(t)
    results = {"persistence": {"val": metrics(y.iloc[va], base(va)), "test": metrics(y.iloc[te], base(te))}}
    fitted = {}
    for name, est in _candidates().items():
        est.fit(X.iloc[tr], y.iloc[tr])
        fitted[name] = est
        results[name] = {"val": metrics(y.iloc[va], est.predict(X.iloc[va])),
                         "test": metrics(y.iloc[te], est.predict(X.iloc[te]))}
    best = min(results, key=lambda k: results[k]["val"]["mae"])  # selection on validation only
    pred_va = base(va) if best == "persistence" else fitted[best].predict(X.iloc[va])
    q_lo, q_hi = np.quantile(y.iloc[va].to_numpy() - pred_va, [PI_LOW, PI_HIGH])
    importance = None
    if best == "random_forest":
        rf = fitted[best][-1]
        importance = dict(sorted(zip(X.columns, map(float, rf.feature_importances_)),
                                 key=lambda kv: -kv[1])[:10])
    return {"estimator": None if best == "persistence" else fitted[best],
            "meta": {"target": target, "horizon_hours": h, "model_type": best,
                     "feature_version": FEATURE_VERSION, "features": list(X.columns),
                     "training_range": [X.index[tr][0].isoformat(), X.index[tr][-1].isoformat()],
                     "validation_range": [X.index[va][0].isoformat(), X.index[va][-1].isoformat()],
                     "test_range": [X.index[te][0].isoformat(), X.index[te][-1].isoformat()],
                     "metrics": results,
                     "beats_persistence_on_test": best != "persistence" and
                         results[best]["test"]["mae"] < results["persistence"]["test"]["mae"],
                     "residual_q": [float(q_lo), float(q_hi)],
                     "prediction_range": "10th-90th percentile of validation residuals",
                     "feature_importance_not_causal": importance,
                     "dataset_size": int(len(X)),
                     "trained_at": datetime.now(timezone.utc).isoformat()}}


def train_location(lat: float, lon: float, target: str = "pm25", past_days: int = 92) -> dict:
    df, report = clean(fetch_history(lat, lon, past_days))
    if report["pm25_missing_frac"] > MAX_MISSING_FRAC or report["pm25_valid_rows"] < MIN_ROWS:
        raise InsufficientData("Insufficient historical data")
    key = location_key(lat, lon)
    out = {"location": key, "quality": report, "horizons": {}, "skipped": {}}
    (MODEL_DIR / key).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for h in HORIZONS:
        try:
            art = train_horizon(df, target, h)
        except InsufficientData as e:
            out["skipped"][h] = str(e)
            continue
        art["meta"]["model_id"] = f"{key}-{target}-h{h}-{stamp}"
        art["meta"]["data_quality"] = report
        path = MODEL_DIR / key / f"{art['meta']['model_id']}.joblib"
        joblib.dump(art, path)
        with open(MODEL_DIR / key / "registry.jsonl", "a") as fh:  # append-only versioning
            fh.write(json.dumps({**art["meta"], "artifact": path.name}) + "\n")
        out["horizons"][h] = art["meta"]
    return out


def registry(lat: float, lon: float) -> list[dict]:
    p = MODEL_DIR / location_key(lat, lon) / "registry.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def latest_models(lat: float, lon: float) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for rec in registry(lat, lon):  # later lines win
        latest[rec["horizon_hours"]] = rec
    return latest


def forecast(lat: float, lon: float, horizon_hours: int, target: str = "pm25") -> dict:
    gen = datetime.now(timezone.utc).isoformat()
    unavailable = lambda reason: {"status": "unavailable", "reason": reason, "target": target, "generated_at": gen}
    models = {h: m for h, m in latest_models(lat, lon).items() if h <= horizon_hours}
    if not models:
        return unavailable("Forecast unavailable: no trained model for this location")
    try:
        df, _ = clean(fetch_history(lat, lon, 10))
    except Exception as exc:
        return unavailable(f"Provider temporarily unavailable: {exc}")
    last = df["pm25"].last_valid_index() if "pm25" in df else None
    if last is None:
        return unavailable("No recent observation")
    age = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
    if age > MAX_LATEST_AGE_H:
        return unavailable(f"No recent observation (latest is {age:.1f}h old)")
    points = []
    for h, rec in sorted(models.items()):
        art = joblib.load(MODEL_DIR / location_key(lat, lon) / rec["artifact"])
        row = build_features(df, target, h).reindex(columns=rec["features"]).loc[[last]]
        if row.isna().mean(axis=1).iloc[0] > 0.3:
            return unavailable("Feature availability too low for a reliable forecast")
        pred = float(row["now"].iloc[0]) if art["estimator"] is None else float(art["estimator"].predict(row)[0])
        lo, hi = art["meta"]["residual_q"]
        points.append({"timestamp": (last + pd.Timedelta(hours=h)).isoformat(), "horizon_hours": h,
                       "model": rec["model_type"], "prediction": max(pred, 0.0),
                       "lower": max(pred + lo, 0.0), "upper": max(pred + hi, 0.0)})
    return {"status": "success", "target": target, "unit": "µg/m³", "issued_from": last.isoformat(),
            "forecast": points,
            "prediction_range_label": "Prediction range (empirical 10-90% of validation residuals; not a confidence interval)",
            "provider": PROVIDER, "data_status": DATA_STATUS,
            "note": "Inputs are CAMS model output via Open-Meteo, not station observations.",
            "generated_at": gen, "performance": {h: models[h]["metrics"] for h in models}}
