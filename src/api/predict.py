"""
AgriVault – Live Prediction Endpoint
======================================
POST /api/predict — takes commodity + location, returns forecast band,
risk score, lending decision, and recommended LTV.

Request body (JSON)
-------------------
    commodity           : str   (required) e.g. "WHEAT"
    state               : str   (required) e.g. "MAHARASHTRA"
    district            : str   (optional)
    market              : str   (optional)
    requested_loan_amount: float (optional)
    quantity_kg         : float (optional)
    warehouse_grade     : str   (optional, default "B")

Response (JSON)
---------------
    mandi_id            : str
    resolution_type     : str   ("exact_match", "district_fallback", etc.)
    commodity           : str
    forecast            : dict  (7d, 15d, 30d × low/median/high)
    forecast_method     : str   ("quantile_gbm" or "historical_percentile_fallback")
    risk_score          : float [0, 1]
    decision            : str   ("APPROVE" / "CONDITIONAL" / "REJECT")
    recommended_ltv_pct : float (0–100)
    recommended_loan_amount : float or null
    explanation         : dict
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from src.models.risk_ltv_model import COMMODITY_RISK_TIER, RiskLTVModel
from src.serving.location_resolver import resolve_mandi, resolve_mandi_by_coords
from src.serving.model_registry import get_model
from src.serving.nasa_weather import fetch_nasa_weather, build_weather_features
from src.serving.weather_forecast import fetch_and_build_features as fetch_forecast_features

log = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)

# Feature columns expected by the QuantileGBM (same as train.py)
_FEATURE_COLS = [
    "price_lag_1d", "price_lag_7d", "price_lag_14d", "price_lag_30d",
    "price_mean_7d", "price_std_7d",
    "price_mean_14d", "price_std_14d",
    "price_mean_30d", "price_std_30d",
    "price_momentum_7d",
    "arrivals_tonnes", "arrivals_mean_7d",
    "temp_mean_7d", "precip_sum_7d", "humidity_mean_7d",
    "ndvi", "ndvi_delta_30d",
    "food_cpi_index", "food_wpi_index",
    "day_of_week", "day_of_month", "month", "is_weekend",
]

_HORIZONS = [7, 15, 30]
_QUANTILES = [0.10, 0.50, 0.90]

# Month → season mapping
_MONTH_TO_SEASON = {
    6: "Kharif", 7: "Kharif", 8: "Kharif", 9: "Kharif",   # Jun–Sep
    10: "Rabi", 11: "Rabi", 12: "Rabi",                      # Oct–Dec
    1: "Rabi", 2: "Rabi", 3: "Rabi",                         # Jan–Mar
    4: "Zaid", 5: "Zaid",                                     # Apr–May
}


def _get_snapshot():
    """Return the serving snapshot from the app module.

    When the app runs as ``python -m src.api.app``, Python loads it as
    ``__main__``, not ``src.api.app``.  ``from X import Y`` inside a
    function looks up the *real* module name which may still have the
    initial ``None``.  We therefore check both ``__main__`` and the
    named module to find whichever holds the loaded snapshot.
    """
    import sys
    import src.api.app as _app_mod
    snap = _app_mod._serving_snapshot
    if snap is None:
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and hasattr(main_mod, "_serving_snapshot"):
            snap = main_mod._serving_snapshot
    return snap


def _run_forecast(features: pd.Series) -> tuple[dict, str]:
    """Run QuantileGBM forecast for all horizons.

    Returns (forecast_dict, forecast_method).
    """
    # Build feature vector
    feat_values = []
    missing_cols = []
    for col in _FEATURE_COLS:
        val = features.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing_cols.append(col)
            feat_values.append(0.0)
        else:
            feat_values.append(float(val))

    X = np.array(feat_values).reshape(1, -1)

    # Check if trained models are available for this commodity
    commodity = features.get("commodity", "")
    all_models_available = all(
        get_model(commodity, h) is not None for h in _HORIZONS
    )

    if all_models_available:
        # ── Tier 1: Quantile GBM forecast ─────────────────────────────
        # Use the trained model even if some features were NaN (replaced
        # with 0.0 above).  The model was trained on data with NaN → 0
        # substitutions, so this is consistent.
        if missing_cols:
            log.info(
                "Using Quantile GBM with %d NaN features zero-filled: %s",
                len(missing_cols), missing_cols,
            )
        forecast = {}
        for horizon in _HORIZONS:
            model = get_model(commodity, horizon)
            preds = model.predict(X)  # {0.10: array, 0.50: array, 0.90: array}
            forecast[f"{horizon}d"] = {
                "low": round(float(preds[0.10][0]), 2),
                "median": round(float(preds[0.50][0]), 2),
                "high": round(float(preds[0.90][0]), 2),
            }
        return forecast, "quantile_gbm"

    # ── Tier 2: Historical percentile fallback ──────────────────────────
    # Use price stats from the feature vector as a simple fallback
    price_mean = features.get("price_mean_30d")
    price_std = features.get("price_std_30d")

    if price_mean is not None and not np.isnan(price_mean):
        price_mean = float(price_mean)
        price_std = float(price_std) if price_std is not None and not np.isnan(price_std) else price_mean * 0.1
        # Simple percentile band: median ± 1σ for each horizon
        # Wider band for longer horizons
        forecast = {}
        for horizon in _HORIZONS:
            scale = 1.0 + (horizon / 30.0) * 0.5  # scale uncertainty with horizon
            forecast[f"{horizon}d"] = {
                "low": round(price_mean - 1.28 * price_std * scale, 2),
                "median": round(price_mean, 2),
                "high": round(price_mean + 1.28 * price_std * scale, 2),
            }
        return forecast, "historical_percentile_fallback"

    # ── Tier 3: No data at all ──────────────────────────────────────────
    return {}, "insufficient_data"


def _compute_risk(
    features: pd.Series,
    forecast: dict,
    commodity: str,
    warehouse_grade: str = "B",
) -> dict:
    """Compute risk score, decision, and recommended LTV."""
    # Build a single-row DataFrame for RiskLTVModel.score()
    price_cv = features.get("price_cv")
    if price_cv is None or np.isnan(price_cv):
        # Compute from mean/std if available
        price_mean = features.get("price_mean_7d")
        price_std = features.get("price_std_7d")
        if price_mean and price_std and price_mean > 0:
            price_cv = price_std / price_mean
        else:
            price_cv = 0.1  # default

    # Forecast uncertainty from quantile band width
    forecast_uncertainty = 0.3  # default
    q7d = forecast.get("7d", {})
    if q7d.get("low") is not None and q7d.get("high") is not None:
        median = q7d.get("median", 1)
        if median > 0:
            forecast_uncertainty = (q7d["high"] - q7d["low"]) / (2 * median)
            forecast_uncertainty = min(max(forecast_uncertainty, 0.0), 1.0)

    # Season from month
    month = features.get("month")
    season = _MONTH_TO_SEASON.get(int(month) if month and not np.isnan(month) else 6, "Kharif")

    # Number of warehouses (use 2 as default if not in snapshot)
    n_warehouses = features.get("n_warehouses")
    if n_warehouses is None or (isinstance(n_warehouses, float) and np.isnan(n_warehouses)):
        n_warehouses = 2  # conservative default

    # Warehouse grade (default "B" — standard)
    warehouse_grade = (warehouse_grade or "B").upper().strip()
    if warehouse_grade not in ("A", "B", "C"):
        warehouse_grade = "B"

    risk_df = pd.DataFrame([{
        "commodity": commodity,
        "price_cv": price_cv,
        "forecast_uncertainty": forecast_uncertainty,
        "n_warehouses": n_warehouses,
        "warehouse_grade": warehouse_grade,
        "season": season,
    }])

    model = RiskLTVModel()
    scored = model.score(risk_df)
    row = scored.iloc[0]

    # ── Feature importance (% contribution of each component) ─────────
    feat_importance = {}
    for col in scored.columns:
        if col.startswith("_feat_"):
            name = col.replace("_feat_", "")
            feat_importance[name] = round(float(row[col]) * 100, 1)

    # Build explanation
    explanation = {
        "components": {
            "price_cv": round(float(price_cv), 4),
            "forecast_uncertainty": round(float(forecast_uncertainty), 4),
            "n_warehouses": int(n_warehouses),
            "warehouse_grade": warehouse_grade,
            "commodity_tier": COMMODITY_RISK_TIER.get(commodity.upper(), 0.5),
            "season": season,
        },
        "feature_importance": feat_importance,
        "weights": model.WEIGHTS,
        "thresholds": model.THRESHOLDS,
    }

    return {
        "risk_score": round(float(row["risk_score"]), 4),
        "decision": row["decision"],
        "recommended_ltv_pct": round(float(row["recommended_ltv"]) * 100, 1),
        "explanation": explanation,
    }


def _log_prediction(
    request_id: str,
    body: dict,
    mandi_id: str,
    resolution_type: str,
    result: dict,
    duration_ms: float,
) -> None:
    """Log prediction to S3 for monitoring (best-effort, never crashes the request)."""
    try:
        from src.storage.s3_client import S3Client

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 2),
            "request": body,
            "mandi_id": mandi_id,
            "resolution_type": resolution_type,
            "commodity": body.get("commodity"),
            "forecast_method": result.get("forecast_method"),
            "risk_score": result.get("risk_score"),
            "decision": result.get("decision"),
            "recommended_ltv_pct": result.get("recommended_ltv_pct"),
        }

        s3 = S3Client()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"logs/predictions/{today}/{request_id}.json"
        s3.write_bytes(json.dumps(log_entry, default=str).encode(), key)
    except Exception as exc:
        # Never let logging failure break the prediction
        log.debug("Failed to log prediction: %s", exc)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

import os

# Simple API key check — if AGRIVAULT_API_KEY is set, require it in header
_API_KEY = os.environ.get("AGRIVAULT_API_KEY", "")

# Simple in-memory rate limiter (per-IP, sliding window)
_rate_limits: dict[str, list[float]] = {}
_MAX_REQUESTS = int(os.environ.get("AGRIVAULT_RATE_LIMIT", "30"))  # per minute
_WINDOW = 60.0  # seconds


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    import time as _time
    now = _time.time()
    # Clean old entries
    _rate_limits[ip] = [t for t in _rate_limits.get(ip, []) if now - t < _WINDOW]
    if len(_rate_limits.get(ip, [])) >= _MAX_REQUESTS:
        return False
    _rate_limits.setdefault(ip, []).append(now)
    return True


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    """Live prediction endpoint.

    Accepts commodity + location, resolves to a mandi, runs the forecast
    model, and returns risk score / decision / recommended LTV.
    """
    # ── API key check ────────────────────────────────────────────────────
    if _API_KEY:
        provided = request.headers.get("X-API-Key", "")
        if provided != _API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401

    # ── Rate limiting ────────────────────────────────────────────────────
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({
            "error": "Rate limit exceeded",
            "retry_after_seconds": int(_WINDOW),
        }), 429

    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    # ── Parse request ────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    commodity = body.get("commodity")
    state = body.get("state")
    latitude = body.get("latitude")
    longitude = body.get("longitude")

    # Lat/lon is required when state is not provided
    if not commodity:
        return jsonify({
            "error": "Missing required field: commodity",
        }), 400

    if not state and (latitude is None or longitude is None):
        return jsonify({
            "error": "Missing required fields",
            "required": ["commodity", "state or (latitude + longitude)"],
            "optional": ["district", "market", "requested_loan_amount",
                         "quantity_kg", "warehouse_grade",
                         "latitude", "longitude"],
        }), 400

    commodity = commodity.strip().upper()
    state = state.strip() if state else None
    district = body.get("district", "").strip() or None
    market = body.get("market", "").strip() or None
    latitude = float(latitude) if latitude is not None else None
    longitude = float(longitude) if longitude is not None else None

    # ── Resolve mandi ────────────────────────────────────────────────────
    mandi_id = None
    resolution_type = "not_found"
    mandi_distance_km = None
    mandi_lat = None
    mandi_lon = None

    # Try text-based resolution first
    if state:
        mandi_id, resolution_type = resolve_mandi(state, district, market)

    # If no match from text, or if we have lat/lon, try geo resolution
    if mandi_id is None and latitude is not None and longitude is not None:
        nearest = resolve_mandi_by_coords(latitude, longitude, top_n=1)
        if not nearest.empty:
            mandi_id = nearest.iloc[0]["mandi_id"]
            mandi_distance_km = round(float(nearest.iloc[0]["distance_km"]), 2)
            resolution_type = "geo_nearest"
            mandi_lat = float(nearest.iloc[0]["latitude"])
            mandi_lon = float(nearest.iloc[0]["longitude"])

    if mandi_id is None:
        return jsonify({
            "error": "No mandi data found for this location",
            "state": state,
            "district": district,
            "market": market,
            "latitude": latitude,
            "longitude": longitude,
        }), 404

    # ── Look up feature vector ───────────────────────────────────────────
    snapshot = _get_snapshot()
    features = None
    used_snapshot = False
    nasa_weather_features = {}

    # Try to get features from snapshot
    if snapshot is not None and not snapshot.empty:
        # Normalize mandi_id variants: snapshot may use %20 for spaces
        # (e.g. "UTTAR%20PRADESH_LUCKNOW_334_LUCKNOW")
        mandi_id_variants = [mandi_id]
        if mandi_id and "%20" in mandi_id:
            mandi_id_variants.append(mandi_id.replace("%20", "_"))
        elif mandi_id:
            mandi_id_variants.append(mandi_id.replace("_", "%20", 1))

        row = pd.DataFrame()
        for mid in mandi_id_variants:
            row = snapshot[
                (snapshot["mandi_id"] == mid)
                & (snapshot["commodity"].str.upper() == commodity)
            ]
            log.debug("Snapshot lookup [%s] commodity=%s => %d rows", mid, commodity, len(row))
            if not row.empty:
                break

        # If exact mandi match failed, try fuzzy: find this commodity
        # at any mandi whose state matches the resolved mandi's state prefix
        if row.empty and mandi_id:
            state_prefix = mandi_id.split("_")[0] if "_" in mandi_id else mandi_id
            state_prefix = state_prefix.replace("%20", " ")
            candidates = snapshot[
                (snapshot["commodity"].str.upper() == commodity)
                & (snapshot["mandi_id"].str.upper().str.startswith(state_prefix.upper().replace(" ", "%20")))
            ]
            if candidates.empty:
                # Also try without %20
                candidates = snapshot[
                    (snapshot["commodity"].str.upper() == commodity)
                    & (snapshot["mandi_id"].str.upper().str.startswith(state_prefix.upper().replace(" ", "_")))
                ]
            if not candidates.empty:
                # Pick the one with the shortest mandi_id (closest name match)
                row = candidates.head(1)
                log.info("Fuzzy commodity match: found %s at %s", commodity, row.iloc[0]["mandi_id"])

        if not row.empty:
            features = row.iloc[0]
            used_snapshot = True

    # ── Fetch live NASA weather if lat/lon provided ───────────────────────
    if latitude is not None and longitude is not None:
        try:
            weather_df = fetch_nasa_weather(latitude, longitude, days_back=60)
            if not weather_df.empty:
                nasa_weather_features = build_weather_features(weather_df)
                log.info(
                    "NASA weather features for (%.4f, %.4f): %s",
                    latitude, longitude, nasa_weather_features,
                )
        except Exception as exc:
            log.warning("Failed to fetch NASA weather: %s", exc)

    # If we still have no features at all, return an error
    if features is None:
        # If we have NASA weather, build a minimal feature set for fallback
        if nasa_weather_features and any(v is not None for v in nasa_weather_features.values()):
            # Build minimal features from NASA weather + defaults
            features = pd.Series({
                "commodity": commodity,
                "mandi_id": mandi_id,
                "price_lag_1d": None,
                "price_lag_7d": None,
                "price_lag_14d": None,
                "price_lag_30d": None,
                "price_mean_7d": None,
                "price_std_7d": None,
                "price_mean_14d": None,
                "price_std_14d": None,
                "price_mean_30d": None,
                "price_std_30d": None,
                "price_momentum_7d": None,
                "arrivals_tonnes": None,
                "arrivals_mean_7d": None,
                "ndvi": None,
                "ndvi_delta_30d": None,
                "food_cpi_index": None,
                "food_wpi_index": None,
                "modal_price": 0,
                "day_of_week": pd.Timestamp.now().dayofweek,
                "day_of_month": pd.Timestamp.now().day,
                "month": pd.Timestamp.now().month,
                "is_weekend": 1 if pd.Timestamp.now().dayofweek >= 5 else 0,
                **nasa_weather_features,
            })
            log.info(
                "Building minimal features from NASA weather for %s @%s",
                commodity, mandi_id,
            )
        else:
            return jsonify({
                "error": f"No feature data for {commodity} at mandi {mandi_id}.",
                "mandi_id": mandi_id,
                "resolution_type": resolution_type,
                "hint": "Provide latitude and longitude to fetch live NASA weather data.",
            }), 404

    # ── Override weather features with live NASA data ─────────────────────
    # If we got NASA weather and snapshot features exist, prefer the live data
    if nasa_weather_features and any(v is not None for v in nasa_weather_features.values()):
        for key, val in nasa_weather_features.items():
            if val is not None:
                features[key] = val
        log.info("Overrode weather features with live NASA data")

    # ── Fetch forward weather forecast (Open-Meteo) ──────────────────────
    forecast_weather_features: dict[str, float | None] = {}
    if latitude is not None and longitude is not None:
        try:
            forecast_weather_features = fetch_forecast_features(latitude, longitude)
            log.info(
                "Open-Meteo forecast features for (%.4f, %.4f): %s",
                latitude, longitude, forecast_weather_features,
            )
        except Exception as exc:
            log.warning("Failed to fetch Open-Meteo forecast: %s", exc)

    # ── Run forecast ─────────────────────────────────────────────────────
    forecast, forecast_method = _run_forecast(features)

    # If forecast failed but we used NASA weather, still return risk assessment
    nasa_used = bool(nasa_weather_features and any(v is not None for v in nasa_weather_features.values()))
    if forecast_method == "insufficient_data" and not nasa_used:
        return jsonify({
            "error": "Insufficient data for this commodity-location combination",
            "mandi_id": mandi_id,
            "resolution_type": resolution_type,
            "commodity": commodity,
            "hint": "Provide latitude and longitude to fetch live NASA weather data.",
        }), 404

    # ── Compute risk & decision ──────────────────────────────────────────
    warehouse_grade = body.get("warehouse_grade", "B") or "B"
    risk_result = _compute_risk(features, forecast, commodity, warehouse_grade)

    # ── Build response ───────────────────────────────────────────────────
    modal_price = float(features.get("modal_price", 0) or 0)
    quantity_kg = body.get("quantity_kg")
    requested_loan = body.get("requested_loan_amount")

    recommended_loan_amount = None
    if requested_loan and quantity_kg:
        max_ltv_decimal = risk_result["recommended_ltv_pct"] / 100
        collateral_value = modal_price * float(quantity_kg)
        max_loan = max_ltv_decimal * collateral_value
        recommended_loan_amount = round(min(float(requested_loan), max_loan), 2)

    duration_ms = (time.time() - t0) * 1000

    # Determine which weather sources were used
    forecast_weather_used = bool(forecast_weather_features and any(v is not None for v in forecast_weather_features.values()))

    # -- MODIS NDVI anomaly info (if available) ------------------------
    ndvi_anomaly_info = {}
    for ndvi_col in ["modis_ndvi", "ndvi_anomaly", "ndvi_anomaly_7d_avg",
                     "ndvi_stress_flag", "ndvi_surplus_flag"]:
        val = features.get(ndvi_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            ndvi_anomaly_info[ndvi_col] = round(float(val), 4) if isinstance(val, (int, float)) else val
    if ndvi_anomaly_info:
        ndvi_anomaly_info["ndvi_signal"] = (
            "STRESS" if ndvi_anomaly_info.get("ndvi_stress_flag", 0) == 1
            else "SURPLUS" if ndvi_anomaly_info.get("ndvi_surplus_flag", 0) == 1
            else "NORMAL"
        )

    result = {
        "request_id": request_id,
        "mandi_id": mandi_id,
        "resolution_type": resolution_type,
        "commodity": commodity,
        "used_snapshot": used_snapshot,
        "nasa_weather_used": nasa_used,
        "forecast_weather_used": forecast_weather_used,
        "mandi_distance_km": mandi_distance_km,
        "forecast": forecast,
        "forecast_method": forecast_method,
        "risk_score": risk_result["risk_score"],
        "decision": risk_result["decision"],
        "recommended_ltv_pct": risk_result["recommended_ltv_pct"],
        "recommended_loan_amount": recommended_loan_amount,
        "modal_price": round(modal_price, 2),
        "ndvi_anomaly": ndvi_anomaly_info or None,
        "explanation": risk_result["explanation"],
        "duration_ms": round(duration_ms, 2),
    }

    # Include forward forecast weather features if available
    if forecast_weather_used:
        result["forecast_weather"] = {k: round(v, 2) if v is not None else None for k, v in forecast_weather_features.items()}

    # ── Log prediction (best-effort) ─────────────────────────────────────
    _log_prediction(request_id, body, mandi_id, resolution_type, result, duration_ms)

    log.info(
        "[%s] %s @%s → %s (risk=%.3f, LTV=%.1f%%) in %.0fms",
        request_id, commodity, mandi_id[:30],
        risk_result["decision"], risk_result["risk_score"],
        risk_result["recommended_ltv_pct"], duration_ms,
    )

    return jsonify(result)
