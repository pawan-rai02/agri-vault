"""
AgriVault – Decision-Support Dashboard & API
=============================================
Flask application serving risk scores, lending decisions, and forecasts.

Endpoints
---------
    GET /                          → Dashboard overview
    GET /predict                   → Prediction form
    GET /commodity/<name>          → Commodity detail page
    POST /api/predict              → Live prediction (JSON)
    GET /api/scores                → All risk scores (JSON)
    GET /api/scores/<commodity>    → Scores for one commodity (JSON)
    GET /api/summary               → Summary statistics (JSON)
    GET /api/decisions             → Decision distribution (JSON)
    GET /api/commodities           → List of all commodities (JSON)

Run
---
    python -m src.api.app
    python -m src.api.app --port 5000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.api.predict import predict_bp
from src.features.build_serving_snapshot import load_snapshot
from src.serving.model_registry import load_all_models, loaded_model_count
from src.storage.s3_client import S3Client

# Load .env file if present (for local development)
load_dotenv()

log = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# Secret key for session signing — read from env, fall back to dev default
app.secret_key = os.environ.get(
    "AGRIVAULT_SECRET_KEY",
    os.environ.get("SECRET_KEY", "dev-only-change-in-production"),
)

# Register blueprints
app.register_blueprint(predict_bp)

# Application startup timestamp for /health endpoint
_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------

_scored_df: pd.DataFrame | None = None
_serving_snapshot: pd.DataFrame | None = None


def _load_scored_data() -> pd.DataFrame:
    """Load scored risk features from S3 into memory."""
    global _scored_df
    if _scored_df is not None:
        return _scored_df

    s3 = S3Client()

    log.info("Loading scored risk features from S3...")
    _scored_df = s3.read_parquet_s3("features/risk_features/")
    log.info("Loaded %d rows", len(_scored_df))
    return _scored_df


def _load_serving_snapshot() -> pd.DataFrame:
    """Load the latest feature-serving snapshot from S3 into memory.

    The snapshot contains one row per (mandi_id, commodity) with the most
    recent feature vector — used by the live prediction endpoint.
    """
    global _serving_snapshot
    if _serving_snapshot is not None:
        return _serving_snapshot

    try:
        log.info("Loading serving snapshot from S3...")
        _serving_snapshot = load_snapshot()
        log.info(
            "Loaded serving snapshot: %d rows, %d cols",
            len(_serving_snapshot), len(_serving_snapshot.columns),
        )
    except FileNotFoundError:
        log.warning(
            "Serving snapshot not found on S3 — live predictions will be "
            "unavailable until build_serving_snapshot.py is run."
        )
        _serving_snapshot = pd.DataFrame()

    return _serving_snapshot



# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

@app.route("/predict")
def predict_page():
    """Prediction form page."""
    return render_template("predict.html")


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------

@app.route("/health")
def health_check():
    """Liveness / readiness probe.

    Returns 200 if the API is running and scored data is loaded.
    Used by load balancers, container orchestrators, and monitoring.
    """
    uptime_s = round(time.time() - _START_TIME, 1)
    data_loaded = _scored_df is not None and len(_scored_df) > 0
    snapshot_loaded = _serving_snapshot is not None and len(_serving_snapshot) > 0
    n_models = loaded_model_count()
    status = "healthy" if data_loaded else "degraded"
    http_code = 200 if data_loaded else 503

    return jsonify({
        "status": status,
        "uptime_seconds": uptime_s,
        "data_loaded": data_loaded,
        "rows_loaded": len(_scored_df) if _scored_df is not None else 0,
        "serving_snapshot_loaded": snapshot_loaded,
        "serving_snapshot_rows": len(_serving_snapshot) if _serving_snapshot is not None else 0,
        "models_loaded": n_models,
        "version": os.environ.get("AGRIVAULT_VERSION", "unknown"),
    }), http_code


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Main dashboard overview."""
    df = _load_scored_data()

    # Summary stats
    total = len(df)
    decisions = df["decision"].value_counts().to_dict()
    avg_risk = float(df["risk_score"].mean())
    avg_ltv = float(df["recommended_ltv"].mean()) * 100

    # Top commodities by row count
    top_commodities = (
        df.groupby("commodity")
        .agg(
            count=("risk_score", "count"),
            avg_risk=("risk_score", "mean"),
            avg_ltv=("recommended_ltv", "mean"),
        )
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index()
    )
    top_commodities["avg_ltv"] = (top_commodities["avg_ltv"] * 100).round(1)
    top_commodities["avg_risk"] = top_commodities["avg_risk"].round(3)

    # Risk distribution by commodity category
    if "commodity_category" in df.columns:
        cat_risk = (
            df.groupby("commodity_category")
            .agg(
                avg_risk=("risk_score", "mean"),
                count=("risk_score", "count"),
            )
            .sort_values("avg_risk", ascending=False)
            .reset_index()
        )
        cat_risk["avg_risk"] = cat_risk["avg_risk"].round(3)
    else:
        cat_risk = pd.DataFrame()

    # State-level summary
    state_summary = (
        df.groupby("state")
        .agg(
            count=("risk_score", "count"),
            avg_risk=("risk_score", "mean"),
            approve_pct=("decision", lambda x: (x == "APPROVE").mean() * 100),
        )
        .sort_values("count", ascending=False)
        .head(15)
        .reset_index()
    )
    state_summary["avg_risk"] = state_summary["avg_risk"].round(3)
    state_summary["approve_pct"] = state_summary["approve_pct"].round(1)

    return render_template(
        "dashboard.html",
        total=total,
        decisions=decisions,
        avg_risk=round(avg_risk, 3),
        avg_ltv=round(avg_ltv, 1),
        top_commodities=top_commodities.to_dict("records"),
        cat_risk=cat_risk.to_dict("records"),
        state_summary=state_summary.to_dict("records"),
    )


@app.route("/commodity/<name>")
def commodity_detail(name: str):
    """Detail page for a single commodity."""
    df = _load_scored_data()
    name_upper = name.upper().replace("-", " ")

    sub = df[df["commodity"] == name_upper]
    if sub.empty:
        return render_template("404.html", name=name), 404

    # Stats
    stats = {
        "commodity": name_upper,
        "total": len(sub),
        "avg_risk": round(float(sub["risk_score"].mean()), 3),
        "avg_ltv": round(float(sub["recommended_ltv"].mean()) * 100, 1),
        "decisions": sub["decision"].value_counts().to_dict(),
        "categories": sub["commodity_category"].unique().tolist(),
    }

    # Per-state breakdown
    state_breakdown = (
        sub.groupby("state")
        .agg(
            count=("risk_score", "count"),
            avg_risk=("risk_score", "mean"),
            avg_ltv=("recommended_ltv", "mean"),
            approve_pct=("decision", lambda x: (x == "APPROVE").mean() * 100),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )
    state_breakdown["avg_risk"] = state_breakdown["avg_risk"].round(3)
    state_breakdown["avg_ltv"] = (state_breakdown["avg_ltv"] * 100).round(1)
    state_breakdown["approve_pct"] = state_breakdown["approve_pct"].round(1)

    # Top districts
    district_breakdown = (
        sub.groupby(["state", "district"])
        .agg(
            count=("risk_score", "count"),
            avg_risk=("risk_score", "mean"),
            avg_ltv=("recommended_ltv", "mean"),
        )
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index()
    )
    district_breakdown["avg_risk"] = district_breakdown["avg_risk"].round(3)
    district_breakdown["avg_ltv"] = (district_breakdown["avg_ltv"] * 100).round(1)

    return render_template(
        "commodity.html",
        stats=stats,
        state_breakdown=state_breakdown.to_dict("records"),
        district_breakdown=district_breakdown.to_dict("records"),
    )


# ---------------------------------------------------------------------------
# API routes (JSON)
# ---------------------------------------------------------------------------

@app.route("/api/scores")
def api_scores():
    """All risk scores, optionally filtered by commodity or decision."""
    df = _load_scored_data()

    commodity = request.args.get("commodity")
    decision = request.args.get("decision")
    state = request.args.get("state")

    if commodity:
        df = df[df["commodity"] == commodity.upper()]
    if decision:
        df = df[df["decision"] == decision.upper()]
    if state:
        df = df[df["state"].str.contains(state.upper(), na=False)]

    # Limit response size (validated to avoid ValueError on bad input)
    try:
        limit = int(request.args.get("limit", 1000))
    except (TypeError, ValueError):
        limit = 1000
    limit = max(1, min(limit, 10000))
    df = df.head(limit)

    cols = [
        "state", "district", "commodity", "mandi_mean_price",
        "price_cv", "forecast_uncertainty", "n_warehouses",
        "risk_score", "decision", "recommended_ltv",
    ]
    cols = [c for c in cols if c in df.columns]

    return jsonify({
        "count": len(df),
        "data": df[cols].to_dict("records"),
    })


@app.route("/api/scores/<commodity>")
def api_scores_commodity(commodity: str):
    """Risk scores for a specific commodity."""
    df = _load_scored_data()
    sub = df[df["commodity"] == commodity.upper()]

    if sub.empty:
        return jsonify({"error": f"Commodity '{commodity}' not found"}), 404

    cols = [
        "state", "district", "mandi_mean_price",
        "price_cv", "forecast_uncertainty", "n_warehouses",
        "risk_score", "decision", "recommended_ltv",
    ]
    cols = [c for c in cols if c in sub.columns]

    return jsonify({
        "commodity": commodity.upper(),
        "count": len(sub),
        "avg_risk": round(float(sub["risk_score"].mean()), 3),
        "avg_ltv": round(float(sub["recommended_ltv"].mean()) * 100, 1),
        "decisions": sub["decision"].value_counts().to_dict(),
        "data": sub[cols].to_dict("records"),
    })


@app.route("/api/summary")
def api_summary():
    """Summary statistics across all commodities."""
    df = _load_scored_data()

    summary = {
        "total_rows": len(df),
        "total_commodities": df["commodity"].nunique(),
        "total_states": df["state"].nunique(),
        "avg_risk_score": round(float(df["risk_score"].mean()), 3),
        "avg_recommended_ltv": round(float(df["recommended_ltv"].mean()) * 100, 1),
        "decision_distribution": df["decision"].value_counts().to_dict(),
        "risk_percentiles": {
            "p10": round(float(df["risk_score"].quantile(0.10)), 3),
            "p25": round(float(df["risk_score"].quantile(0.25)), 3),
            "p50": round(float(df["risk_score"].quantile(0.50)), 3),
            "p75": round(float(df["risk_score"].quantile(0.75)), 3),
            "p90": round(float(df["risk_score"].quantile(0.90)), 3),
        },
    }
    return jsonify(summary)


@app.route("/api/decisions")
def api_decisions():
    """Decision distribution by commodity."""
    df = _load_scored_data()

    result = (
        df.groupby(["commodity", "decision"])
        .size()
        .reset_index(name="count")
        .sort_values(["commodity", "decision"])
    )

    return jsonify({
        "count": len(result),
        "data": result.to_dict("records"),
    })


@app.route("/api/commodities")
def api_commodities():
    """List of all commodities with stats."""
    df = _load_scored_data()

    result = (
        df.groupby("commodity")
        .agg(
            count=("risk_score", "count"),
            avg_risk=("risk_score", "mean"),
            avg_ltv=("recommended_ltv", "mean"),
            approve_pct=("decision", lambda x: (x == "APPROVE").mean() * 100),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )
    result["avg_risk"] = result["avg_risk"].round(3)
    result["avg_ltv"] = (result["avg_ltv"] * 100).round(1)
    result["approve_pct"] = result["approve_pct"].round(1)

    return jsonify({
        "count": len(result),
        "data": result.to_dict("records"),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AgriVault Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Pre-load data
    _load_scored_data()
    _load_serving_snapshot()

    # Load trained models into memory
    try:
        n = load_all_models()
        log.info("Loaded %d models into registry", n)
    except Exception as exc:
        log.warning("Could not load models: %s — predictions will be unavailable", exc)

    log.info("Starting AgriVault Dashboard on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
