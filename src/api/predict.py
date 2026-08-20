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
from src.serving.location_resolver import resolve_mandi
from src.serving.model_registry import get_model

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
    """Import and return the serving snapshot from the app module."""
    from src.api.app import _serving_snapshot
    return _serving_snapshot


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

    # Check if all required models are available
    commodity = features.get("commodity", "")
    all_models_available = all(
        get_model(commodity, h) is not None for h in _HORIZONS
    )

    if all_models_available and not missing_cols:
        # ── Tier 1: Full Quantile GBM forecast ──────────────────────────
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

    risk_df = pd.DataFrame([{
        "commodity": commodity,
        "price_cv": price_cv,
        "forecast_uncertainty": forecast_uncertainty,
        "n_warehouses": n_warehouses,
        "season": season,
    }])

    model = RiskLTVModel()
    scored = model.score(risk_df)
    row = scored.iloc[0]

    # Build explanation
    explanation = {
        "components": {
            "price_cv": round(float(price_cv), 4),
            "forecast_uncertainty": round(float(forecast_uncertainty), 4),
            "n_warehouses": int(n_warehouses),
            "commodity_tier": COMMODITY_RISK_TIER.get(commodity.upper(), 0.5),
            "season": season,
        },
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

@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    """Live prediction endpoint.

    Accepts commodity + location, resolves to a mandi, runs the forecast
    model, and returns risk score / decision / recommended LTV.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    # ── Parse request ────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    commodity = body.get("commodity")
    state = body.get("state")

    if not commodity or not state:
        return jsonify({
            "error": "Missing required fields",
            "required": ["commodity", "state"],
            "optional": ["district", "market", "requested_loan_amount",
                         "quantity_kg", "warehouse_grade"],
        }), 400

    commodity = commodity.strip().upper()
    state = state.strip()
    district = body.get("district", "").strip() or None
    market = body.get("market", "").strip() or None

    # ── Resolve mandi ────────────────────────────────────────────────────
    mandi_id, resolution_type = resolve_mandi(state, district, market)

    if mandi_id is None:
        return jsonify({
            "error": "No mandi data found for this location",
            "state": state,
            "district": district,
            "market": market,
        }), 404

    # ── Look up feature vector ───────────────────────────────────────────
    snapshot = _get_snapshot()
    if snapshot is None or snapshot.empty:
        return jsonify({
            "error": "Serving snapshot not available — data not yet loaded",
        }), 503

    row = snapshot[
        (snapshot["mandi_id"] == mandi_id)
        & (snapshot["commodity"] == commodity)
    ]

    if row.empty:
        # Try case-insensitive commodity match
        row = snapshot[
            (snapshot["mandi_id"] == mandi_id)
            & (snapshot["commodity"].str.upper() == commodity)
        ]

    if row.empty:
        # Check if this mandi has ANY data (any commodity)
        mandi_data = snapshot[snapshot["mandi_id"] == mandi_id]
        available_commodities = mandi_data["commodity"].unique().tolist() if not mandi_data.empty else []
        return jsonify({
            "error": f"No recent feature data for {commodity} at mandi {mandi_id}",
            "mandi_id": mandi_id,
            "available_commodities": available_commodities[:10],
            "suggestion": "Try a different mandi or check available commodities",
        }), 404

    features = row.iloc[0]

    # ── Run forecast ─────────────────────────────────────────────────────
    forecast, forecast_method = _run_forecast(features)

    if forecast_method == "insufficient_data":
        return jsonify({
            "error": "Insufficient data for this commodity-location combination",
            "mandi_id": mandi_id,
            "resolution_type": resolution_type,
            "commodity": commodity,
        }), 404

    # ── Compute risk & decision ──────────────────────────────────────────
    risk_result = _compute_risk(features, forecast, commodity)

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

    result = {
        "request_id": request_id,
        "mandi_id": mandi_id,
        "resolution_type": resolution_type,
        "commodity": commodity,
        "forecast": forecast,
        "forecast_method": forecast_method,
        "risk_score": risk_result["risk_score"],
        "decision": risk_result["decision"],
        "recommended_ltv_pct": risk_result["recommended_ltv_pct"],
        "recommended_loan_amount": recommended_loan_amount,
        "modal_price": round(modal_price, 2),
        "explanation": risk_result["explanation"],
        "duration_ms": round(duration_ms, 2),
    }

    # ── Log prediction (best-effort) ─────────────────────────────────────
    _log_prediction(request_id, body, mandi_id, resolution_type, result, duration_ms)

    log.info(
        "[%s] %s @%s → %s (risk=%.3f, LTV=%.1f%%) in %.0fms",
        request_id, commodity, mandi_id[:30],
        risk_result["decision"], risk_result["risk_score"],
        risk_result["recommended_ltv_pct"], duration_ms,
    )

    return jsonify(result)
