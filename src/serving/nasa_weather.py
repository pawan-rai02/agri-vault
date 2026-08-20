"""
AgriVault – NASA POWER API Weather Client
==========================================
Fetches daily weather data from NASA POWER (Prediction Of Worldwide Energy
Resources) for arbitrary lat/lon coordinates.  Used by the live prediction
endpoint to enrich feature vectors with real-time weather when the serving
snapshot is unavailable or stale.

NASA POWER API is free, no-auth, and covers global daily weather from 1981
to near-present (~2-3 month lag).

Endpoint
--------
    https://power.larc.nasa.gov/api/temporal/daily/point

Parameters fetched
------------------
    T2M          : Temperature at 2 m (°C)
    PRECTOTCORR  : Precipitation, corrected (mm/day)
    RH2M         : Relative Humidity at 2 m (%)
    WS2M         : Wind Speed at 2 m (m/s)

Usage
-----
    from src.serving.nasa_weather import fetch_nasa_weather

    df = fetch_nasa_weather(26.8467, 80.9462, days_back=30)
    # df has columns: date, temperature_mean, precipitation_mm, humidity, wind_speed
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

# NASA POWER API base URL
_API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameter mapping: NASA name → our column name
_PARAM_MAP = {
    "T2M":         "temperature_mean",
    "PRECTOTCORR": "precipitation_mm",
    "RH2M":        "humidity",
    "WS2M":        "wind_speed",
}

# Maximum date range NASA POWER supports per request (they cap at ~3 years,
# but we only ever need the last ~60 days for 7d/30d rolling features).
_MAX_DAYS = 90


def fetch_nasa_weather(
    latitude: float,
    longitude: float,
    days_back: int = 60,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily weather from NASA POWER for a given location.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees (positive = North).
    longitude : float
        Longitude in decimal degrees (positive = East).
    days_back : int
        How many days of history to fetch (default 60, max 90).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Columns: date, latitude, longitude, temperature_mean, precipitation_mm,
        humidity, wind_speed.
        Empty DataFrame if the API call fails or returns no data.
    """
    days_back = min(days_back, _MAX_DAYS)
    end_date = date.today() - timedelta(days=2)  # NASA lags ~1-2 days
    start_date = end_date - timedelta(days=days_back)

    params = {
        "parameters": ",".join(_PARAM_MAP.keys()),
        "community": "AG",
        "longitude": round(longitude, 4),
        "latitude": round(latitude, 4),
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }

    log.info(
        "Fetching NASA POWER weather for (%.4f, %.4f) from %s to %s",
        latitude, longitude, start_date, end_date,
    )

    try:
        resp = requests.get(_API_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.warning("NASA POWER API request failed: %s", exc)
        return pd.DataFrame()
    except ValueError as exc:
        log.warning("NASA POWER API returned invalid JSON: %s", exc)
        return pd.DataFrame()

    # ── Parse response ────────────────────────────────────────────────────
    # Response structure: {"properties": {"parameter": {"T2M": {"20250101": val, ...}, ...}}}
    try:
        parameters = data["properties"]["parameter"]
    except (KeyError, TypeError):
        log.warning("Unexpected NASA POWER response structure: %s", str(data)[:300])
        return pd.DataFrame()

    # Find the date keys (they appear in any parameter dict)
    first_param = next(iter(_PARAM_MAP.keys()))
    date_keys = parameters.get(first_param, {})
    if not date_keys:
        log.warning("NASA POWER returned no date keys")
        return pd.DataFrame()

    rows = []
    for date_key, value in date_keys.items():
        if value is None or value == -999.0 or value < -900:  # NASA uses -999.0 / -999 for missing
            continue
        try:
            dt = pd.to_datetime(date_key, format="%Y%m%d")
        except ValueError:
            continue

        row = {"date": dt, "latitude": latitude, "longitude": longitude}
        for nasa_param, col_name in _PARAM_MAP.items():
            raw_val = parameters.get(nasa_param, {}).get(date_key)
            if raw_val is not None and raw_val > -900 and raw_val != -999.0:
                row[col_name] = float(raw_val)
            else:
                row[col_name] = None
        rows.append(row)

    if not rows:
        log.warning("NASA POWER returned no valid weather records")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "NASA POWER: fetched %d days of weather for (%.4f, %.4f)",
        len(df), latitude, longitude,
    )
    return df


def build_weather_features(weather_df: pd.DataFrame) -> dict:
    """Build 7-day rolling weather features from a daily weather DataFrame.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Daily weather with columns: date, temperature_mean, precipitation_mm, humidity.

    Returns
    -------
    dict
        Keys: temp_mean_7d, precip_sum_7d, humidity_mean_7d, wind_speed_mean_7d.
        Values are floats or None if insufficient data.
    """
    if weather_df is None or weather_df.empty:
        return {
            "temp_mean_7d": None,
            "precip_sum_7d": None,
            "humidity_mean_7d": None,
            "wind_speed_mean_7d": None,
        }

    # Use the most recent 7 days of data
    recent = weather_df.tail(7)

    def _safe_mean(series):
        vals = series.dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def _safe_sum(series):
        vals = series.dropna()
        return float(vals.sum()) if len(vals) > 0 else None

    return {
        "temp_mean_7d": _safe_mean(recent.get("temperature_mean", pd.Series())),
        "precip_sum_7d": _safe_sum(recent.get("precipitation_mm", pd.Series())),
        "humidity_mean_7d": _safe_mean(recent.get("humidity", pd.Series())),
        "wind_speed_mean_7d": _safe_mean(recent.get("wind_speed", pd.Series())),
    }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fetch NASA POWER weather data")
    parser.add_argument("latitude", type=float, help="Latitude (e.g. 26.8467 for Lucknow)")
    parser.add_argument("longitude", type=float, help="Longitude (e.g. 80.9462 for Lucknow)")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default 30)")
    args = parser.parse_args()

    df = fetch_nasa_weather(args.latitude, args.longitude, days_back=args.days)
    if df.empty:
        print("No data returned.")
    else:
        print(df.to_string(index=False))
        features = build_weather_features(df)
        print("\n7-day rolling features:")
        for k, v in features.items():
            print(f"  {k}: {v}")
