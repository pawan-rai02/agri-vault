"""
AgriVault – Open-Meteo Forecast Weather Client
================================================
Fetches daily weather **forecasts** (up to 16 days ahead) from the Open-Meteo
free API for arbitrary lat/lon coordinates.  Used by the live prediction
endpoint to provide *forward-looking* weather features — what the weather is
*expected to do* over the forecast window, not what it did historically.

This is complementary to the NASA POWER client (historical weather) in
``nasa_weather.py``.

API:  https://open-meteo.com/en/docs  (no API key required)

Parameters fetched (daily):
    temperature_2m_max     : Maximum temperature at 2 m (°C)
    temperature_2m_min     : Minimum temperature at 2 m (°C)
    precipitation_sum      : Precipitation sum (mm)
    relative_humidity_2m_max : Maximum relative humidity at 2 m (%)

Usage
-----
    from src.serving.weather_forecast import fetch_forecast, build_forecast_features

    raw = fetch_forecast(latitude=26.85, longitude=80.95)
    features = build_forecast_features(raw)
    # → {
    #     "forecast_temp_7d_avg":  32.1,
    #     "forecast_precip_7d_sum": 12.4,
    #     "forecast_precip_14d_sum": 28.7,
    #     "forecast_humidity_7d_avg": 75.3,
    # }
"""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables to request
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_max",
]

# Request timeout in seconds
_TIMEOUT = 10


def fetch_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 16,
    timeout: int = _TIMEOUT,
) -> dict[str, Any] | None:
    """Fetch daily weather forecast from Open-Meteo.

    Parameters
    ----------
    latitude, longitude : float
        WGS84 coordinates.
    forecast_days : int
        Number of forecast days to request (max 16).
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    dict or None
        Raw JSON response from Open-Meteo, or None if the request fails.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(_DAILY_VARS),
        "timezone": "auto",
        "forecast_days": min(forecast_days, 16),
    }

    try:
        resp = requests.get(_FORECAST_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "daily" not in data:
            log.warning("Open-Meteo response missing 'daily' key: %s", list(data.keys()))
            return None

        n_days = len(data["daily"].get("time", []))
        log.info(
            "Open-Meteo forecast: %d days for (%.4f, %.4f)",
            n_days, latitude, longitude,
        )
        return data

    except requests.RequestException as exc:
        log.warning("Open-Meteo forecast request failed: %s", exc)
        return None
    except Exception as exc:
        log.warning("Open-Meteo forecast unexpected error: %s", exc)
        return None


def build_forecast_features(raw: dict[str, Any]) -> dict[str, float | None]:
    """Extract aggregate forecast features from the raw Open-Meteo response.

    Computes:
        forecast_temp_7d_avg      — mean of daily max temps over next 7 days (°C)
        forecast_precip_7d_sum    — total precipitation over next 7 days (mm)
        forecast_precip_14d_sum   — total precipitation over next 14 days (mm)
        forecast_humidity_7d_avg  — mean of daily max humidity over next 7 days (%)

    Parameters
    ----------
    raw : dict
        Raw JSON response from ``fetch_forecast()``.

    Returns
    -------
    dict
        Feature name → value (may contain None if insufficient data).
    """
    daily = raw.get("daily", {})
    temps_max = daily.get("temperature_2m_max", [])
    temps_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    humidity = daily.get("relative_humidity_2m_max", [])

    features: dict[str, float | None] = {
        "forecast_temp_7d_avg": None,
        "forecast_precip_7d_sum": None,
        "forecast_precip_14d_sum": None,
        "forecast_humidity_7d_avg": None,
    }

    if not temps_max:
        return features

    # ── Temperature: average of (max+min)/2 over 7 days ────────────────
    temps_avg = []
    for i in range(min(7, len(temps_max))):
        t_max = temps_max[i] if i < len(temps_max) else None
        t_min = temps_min[i] if i < len(temps_min) else None
        if t_max is not None and t_min is not None:
            temps_avg.append((t_max + t_min) / 2)
    if temps_avg:
        features["forecast_temp_7d_avg"] = round(sum(temps_avg) / len(temps_avg), 2)

    # ── Precipitation: sum over 7 days and 14 days ─────────────────────
    precip_7d = [p for p in precip[:7] if p is not None]
    features["forecast_precip_7d_sum"] = round(sum(precip_7d), 2) if precip_7d else None

    precip_14d = [p for p in precip[:14] if p is not None]
    features["forecast_precip_14d_sum"] = round(sum(precip_14d), 2) if precip_14d else None

    # ── Humidity: average of daily max over 7 days ─────────────────────
    humid_7d = [h for h in humidity[:7] if h is not None]
    features["forecast_humidity_7d_avg"] = round(sum(humid_7d) / len(humid_7d), 2) if humid_7d else None

    return features


def fetch_and_build_features(
    latitude: float,
    longitude: float,
) -> dict[str, float | None]:
    """Convenience: fetch forecast and build features in one call.

    Returns empty feature dict (all None) if the API is unreachable.
    """
    raw = fetch_forecast(latitude, longitude)
    if raw is None:
        log.warning("Using empty forecast features (Open-Meteo unreachable)")
        return {
            "forecast_temp_7d_avg": None,
            "forecast_precip_7d_sum": None,
            "forecast_precip_14d_sum": None,
            "forecast_humidity_7d_avg": None,
        }
    return build_forecast_features(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json
    import sys as _sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    lat = float(_sys.argv[1]) if len(_sys.argv) > 1 else 26.8467
    lon = float(_sys.argv[2]) if len(_sys.argv) > 2 else 80.9462

    print(f"Fetching Open-Meteo forecast for ({lat}, {lon})...")
    raw = fetch_forecast(lat, lon)
    if raw:
        features = build_forecast_features(raw)
        print("\nForecast features:")
        print(_json.dumps(features, indent=2))
    else:
        print("Failed to fetch forecast.")
