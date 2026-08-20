"""
AgriVault – Build Gold Risk Features (Gold Layer)
==================================================
Joins price features + WDRA warehouse data + loan risk proxy
to produce a risk feature table for the Loan Risk / LTV model.

Risk feature groups
-------------------
1. Price forecast features  : price_mean_30d, price_std_30d, price_lag_7d
2. Forecast uncertainty     : interval width (pred_q90 - pred_q10) from Phase 3
3. NDVI / vegetation health : ndvi, ndvi_delta_30d
4. Warehouse (WDRA)         : capacity_mt, n_warehouses_in_district
5. Loan proxy features      : grade_default_rate, debt_to_income, ltv_ratio
6. Commodity category       : mapped from commodity name
7. Season                   : Kharif / Rabi / Zaid from month

Input
-----
    standardized/apmc/         → price + mandi identity
    standardized/wdra/         → warehouse capacity per district
    standardized/loans/        → loan proxy risk scores
    models/qgbm_*_predictions  → forecast + uncertainty (Phase 3 output)

Output
------
    s3://agrivault-lake-pawan/features/risk_features/

Run
---
    python -m src.features.build_risk_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage.s3_client import S3Client, load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Season mapping
# ---------------------------------------------------------------------------

SEASON_MAP = {
    1: "Rabi", 2: "Rabi", 3: "Rabi",
    4: "Zaid", 5: "Zaid", 6: "Zaid",
    7: "Kharif", 8: "Kharif", 9: "Kharif",
    10: "Kharif", 11: "Rabi", 12: "Rabi",
}

COMMODITY_CATEGORY = {
    "WHEAT": "Cereal", "RICE": "Cereal", "MAIZE": "Cereal", "JOWAR": "Cereal",
    "BAJRA": "Cereal", "RAGI": "Cereal",
    "SOYABEAN": "Oilseed", "GROUNDNUT": "Oilseed", "SUNFLOWER": "Oilseed",
    "MUSTARD": "Oilseed", "RAPE SEED": "Oilseed",
    "COTTON": "Fiber",
    "SUGARCANE": "Cash Crop",
    "ONION": "Vegetable", "POTATO": "Vegetable", "TOMATO": "Vegetable",
    "BANANA": "Fruit", "MANGO": "Fruit",
    "CHILLI": "Spice", "TURMERIC": "Spice", "GINGER": "Spice",
    "ARHAR": "Pulse", "MOONG": "Pulse", "URAD": "Pulse",
}


def map_commodity_category(commodity: str) -> str:
    return COMMODITY_CATEGORY.get(str(commodity).upper().strip(), "Other")


# ---------------------------------------------------------------------------
# WDRA aggregation
# ---------------------------------------------------------------------------

def build_wdra_district_features(wdra: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate WDRA warehouse data to district level.
    Returns: state, district, total_capacity_mt, n_warehouses
    """
    wdra["district"] = wdra["district"].str.strip().str.upper()
    if "state" in wdra.columns:
        wdra["state"] = wdra["state"].str.strip().str.upper()

    # Normalize column names to what we expect
    col_map = {}
    if "capacityin_mt" in wdra.columns and "capacity_mt" not in wdra.columns:
        col_map["capacityin_mt"] = "capacity_mt"
    if "wh_name" in wdra.columns and "warehouse_name" not in wdra.columns:
        col_map["wh_name"] = "warehouse_name"
    if col_map:
        wdra = wdra.rename(columns=col_map)
    # Cast capacity to numeric (some values are stored as strings)
    if "capacity_mt" in wdra.columns:
        wdra["capacity_mt"] = pd.to_numeric(wdra["capacity_mt"], errors="coerce").fillna(0)

    grp_cols = ["state", "district"] if "state" in wdra.columns else ["district"]
    grp = wdra.groupby(grp_cols).agg(
        total_capacity_mt=("capacity_mt", "sum"),
        n_warehouses=("warehouse_name", "count"),
    ).reset_index()
    return grp


# ---------------------------------------------------------------------------
# Loan proxy aggregation
# ---------------------------------------------------------------------------

def build_loan_features(loans: pd.DataFrame) -> pd.DataFrame:
    """
    Extract portfolio-level risk features from the loan proxy.
    Returns a single-row DataFrame of national averages (broadcast join).
    """
    result = {
        "portfolio_default_rate": loans["is_default"].mean() if "is_default" in loans.columns else np.nan,
        "portfolio_mean_ltv":     loans["ltv_ratio"].mean()  if "ltv_ratio"  in loans.columns else np.nan,
    }

    if "credit_grade" in loans.columns and "is_default" in loans.columns:
        grade_rates = (
            loans.groupby("credit_grade")["is_default"]
            .mean()
            .rename("grade_default_rate")
            .reset_index()
        )
        log.info("Grade default rates:\n%s", grade_rates.to_string(index=False))

    return pd.DataFrame([result])


# ---------------------------------------------------------------------------
# Main join and feature assembly
# ---------------------------------------------------------------------------

def build_risk_features(
    apmc: pd.DataFrame,
    wdra: pd.DataFrame,
    loans: pd.DataFrame,
    forecast_preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble risk feature table.

    Parameters
    ----------
    apmc           : standardized APMC (mandi × commodity × date)
    wdra           : standardized WDRA (district-level warehouses)
    loans          : standardized loan proxy
    forecast_preds : Phase-3 quantile predictions (optional — adds uncertainty)
    """
    # ── Price summary per mandi × commodity ──────────────────────────────
    price_grp = (
        apmc
        .groupby(["state", "district", "commodity"])
        .agg(
            mandi_mean_price=("modal_price", "mean"),
            mandi_std_price= ("modal_price", "std"),
            mandi_n_days=    ("modal_price", "count"),
        )
        .reset_index()
    )

    # ── WDRA warehouse features ───────────────────────────────────────────
    wdra_feat = build_wdra_district_features(wdra)
    price_grp["district"] = price_grp["district"].str.strip().str.upper()
    price_grp["state"]    = price_grp["state"].str.strip().str.upper()

    df = price_grp.merge(wdra_feat, on=["state", "district"], how="left")
    df["total_capacity_mt"] = df["total_capacity_mt"].fillna(0)
    df["n_warehouses"]      = df["n_warehouses"].fillna(0)

    # ── Commodity category ────────────────────────────────────────────────
    df["commodity_category"] = df["commodity"].apply(map_commodity_category)

    # ── Loan portfolio features (broadcast) ───────────────────────────────
    loan_feat = build_loan_features(loans)
    for col, val in loan_feat.iloc[0].items():
        df[col] = val

    # ── Price volatility as collateral risk proxy ─────────────────────────
    # Higher std/mean = higher price risk → lower safe LTV
    df["price_cv"] = df["mandi_std_price"] / df["mandi_mean_price"].clip(lower=1)

    # ── Forecast uncertainty (Phase 3) ───────────────────────────────────
    if forecast_preds is not None:
        # Example: average interval width across horizons
        width_cols = [c for c in forecast_preds.columns if "q90" in c and "pred" in c]
        low_cols   = [c.replace("q90", "q10") for c in width_cols]
        if width_cols and low_cols:
            for wc, lc in zip(width_cols, low_cols):
                if wc in forecast_preds.columns and lc in forecast_preds.columns:
                    forecast_preds[f"width_{wc}"] = (
                        forecast_preds[wc] - forecast_preds[lc]
                    )
            width_mean = forecast_preds[[c for c in forecast_preds.columns
                                         if c.startswith("width_")]].mean(axis=1)
            fc_grp = (
                forecast_preds
                .assign(forecast_uncertainty=width_mean)
                .groupby("commodity")["forecast_uncertainty"]
                .mean()
                .reset_index()
            )
            df = df.merge(fc_grp, on="commodity", how="left")
    else:
        df["forecast_uncertainty"] = np.nan

    # ── Risk score (simple weighted composite — placeholder) ──────────────
    # In production this will be the output of the trained risk model.
    # Here we build a rules-based proxy for initial evaluation:
    df["risk_score_proxy"] = (
        df["price_cv"].fillna(0.1) * 0.4
        + (1 - df["n_warehouses"].clip(upper=50) / 50) * 0.3
        + df["portfolio_default_rate"].fillna(0.06) * 0.3
    )

    # ── LTV recommendation (simplified) ──────────────────────────────────
    # Max LTV decreases as risk_score rises
    df["recommended_max_ltv"] = (0.75 - df["risk_score_proxy"].clip(0, 0.5)).clip(
        lower=0.40, upper=0.75
    )

    log.info("Risk feature table: %d rows, %d cols", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    s3 = S3Client()
    bucket = s3.bucket

    # ── Load Silver tables ─────────────────────────────────────────────────
    log.info("Loading APMC Silver from S3...")
    apmc = s3.read_parquet_s3("standardized/apmc/")

    log.info("Loading WDRA Silver from S3...")
    wdra = s3.read_parquet_s3("standardized/wdra/")

    log.info("Loading Loans Silver from S3...")
    loans = s3.read_parquet_s3("standardized/loans/")

    # ── Optionally load Phase-3 predictions ───────────────────────────────
    forecast_preds = None  # Set to loaded DataFrame once Phase 3 is done

    # ── Build risk features ────────────────────────────────────────────────
    risk_df = build_risk_features(apmc, wdra, loans, forecast_preds)

    # ── Upload to S3 ─────────────────────────────────────────────────────
    out_key = s3.key("features", "risk_features/risk_features.parquet")
    s3.write_parquet_s3(risk_df, out_key)
    log.info("✓ Risk feature table written to S3")


if __name__ == "__main__":
    main()
