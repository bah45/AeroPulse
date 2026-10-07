"""Cleaning + leakage-safe feature engineering. Every feature at time t uses data <= t."""
import numpy as np
import pandas as pd

FEATURE_VERSION = "v1"
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "o3"]
WEATHER = ["temperature", "relative_humidity", "wind_speed", "wind_direction", "pressure", "precipitation"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
ROLL_MEAN = [3, 6, 12, 24]
ROLL_STD = [6, 24]
MAX_GAP_HOURS = 3   # only short gaps are interpolated; longer gaps stay missing
MAX_VALID = 2000    # µg/m³ ceiling: above this is treated as a provider error


class InsufficientData(Exception):
    pass


def clean(raw: pd.DataFrame, now: pd.Timestamp | None = None):
    """Validate, dedupe, sort, resample hourly, drop future rows, short-gap interpolate."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    dup = int(df.duplicated("timestamp").sum())
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df = df.resample("1h").mean(numeric_only=True)
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    df = df[df.index <= now]  # providers return forecast hours; never use them
    impossible = 0
    for c in POLLUTANTS:
        if c in df:
            bad = (df[c] < 0) | (df[c] > MAX_VALID)
            impossible += int(bad.sum())
            df.loc[bad, c] = np.nan
    for c in [c for c in POLLUTANTS + WEATHER if c in df]:
        df[c] = df[c].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    pm = df["pm25"] if "pm25" in df else pd.Series(dtype=float)
    last_valid = pm.last_valid_index()
    report = {
        "rows": int(len(df)), "duplicates_removed": dup, "impossible_values_nulled": impossible,
        "pm25_valid_rows": int(pm.notna().sum()),
        "pm25_missing_frac": float(pm.isna().mean()) if len(pm) else 1.0,
        "start": df.index.min().isoformat() if len(df) else None,
        "end": df.index.max().isoformat() if len(df) else None,
        "latest_pm25_at": last_valid.isoformat() if last_valid is not None else None,
        "weather_available": [c for c in WEATHER if c in df and df[c].notna().any()],
        "pollutants_available": [c for c in POLLUTANTS if c in df and df[c].notna().any()],
    }
    return df, report


def build_features(df: pd.DataFrame, target: str = "pm25", horizon: int = 6) -> pd.DataFrame:
    """Feature matrix indexed by issue time t. All inputs are <= t; calendar fields are of t+h."""
    s = df[target]
    f = pd.DataFrame(index=df.index)
    f["now"] = s
    for k in LAGS:
        f[f"lag_{k}"] = s.shift(k)
    for w in ROLL_MEAN:
        f[f"rolling_mean_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).mean()
    for w in ROLL_STD:
        f[f"rolling_std_{w}"] = s.rolling(w, min_periods=max(2, w // 2)).std()
    for c in [c for c in POLLUTANTS if c != target and c in df]:
        f[f"{c}_now"] = df[c]
    for c in [c for c in WEATHER if c in df]:
        f[c] = df[c]
    f["missing_recent_6h"] = s.isna().rolling(6, min_periods=1).sum()
    tt = df.index + pd.Timedelta(hours=horizon)  # target time is known in advance: legitimate
    f["hour"], f["day_of_week"], f["day_of_month"] = tt.hour, tt.dayofweek, tt.day
    f["month"], f["day_of_year"] = tt.month, tt.dayofyear
    f["week_of_year"] = tt.isocalendar().week.to_numpy().astype(int)
    for name, val, period in [("hour", tt.hour, 24), ("day_of_week", tt.dayofweek, 7), ("month", tt.month, 12)]:
        f[f"sin_{name}"] = np.sin(2 * np.pi * np.asarray(val) / period)
        f[f"cos_{name}"] = np.cos(2 * np.pi * np.asarray(val) / period)
    return f.dropna(axis=1, how="all")


def supervised(df: pd.DataFrame, target: str, horizon: int):
    X = build_features(df, target, horizon)
    y = df[target].shift(-horizon)
    keep = y.notna() & X["now"].notna()
    return X[keep], y[keep]
