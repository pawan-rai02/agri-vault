"""
AgriVault - Join MODIS NDVI Anomalies with APMC Price Data
============================================================
Matches NASA MODIS NDVI anomaly features with Agmarknet APMC daily prices
at the (mandi_id, date) grain.

This produces a combined dataset where each row has:
  - APMC: modal_price, arrivals, market metadata
  - MODIS: ndvi_raw, ndvi_anomaly, ndvi_baseline_mean, ndvi_baseline_std

NDVI anomaly = (current NDVI − multi-year baseline mean) / baseline std
  -> Positive anomaly = greener-than-usual vegetation (potential oversupply)
  -> Negative anomaly = stressed vegetation (potential drought / undersupply)

Input
-----
  data/raw/ndvi_modis/ndvi_modis_current.csv     -- MODIS NDVI + anomalies
  data/raw/apmc/apmc_market_prices_2021_2025.csv  -- APMC historical prices
  data/reference/mandi_locations.csv               -- Mandi lat/lon reference

Output
------
  data/standardized/joined/ndvi_anomaly_apmc_joined.csv
  s3://agrivault-lake-pawan/standardized/joined/ndvi_anomaly_apmc/

Run
---
    python -m src.standardization.join_ndvi_anomaly
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NDVI_MODIS_DIR = PROJECT_ROOT / "data" / "raw" / "ndvi_modis"
APMC_DIR = PROJECT_ROOT / "data" / "raw" / "apmc"
MANDI_FILE = PROJECT_ROOT / "data" / "reference" / "mandi_locations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized" / "joined"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_modis_current() -> pd.DataFrame:
    """Load MODIS NDVI current-year data with anomalies."""
    # Find the most recent current file
    candidates = sorted(NDVI_MODIS_DIR.glob("ndvi_modis_current*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No MODIS current NDVI files found in {NDVI_MODIS_DIR}\n"
            "Run: python -m src.ingestion.fetch_modis_ndvi"
        )
    path = candidates[-1]
    log.info("Loading MODIS current NDVI from %s", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info("  -> %d rows, %d mandis", len(df), df["mandi_id"].nunique())
    return df


def load_modis_baseline() -> pd.DataFrame:
    """Load MODIS NDVI baseline statistics."""
    candidates = sorted(NDVI_MODIS_DIR.glob("ndvi_modis_baseline*.csv"))
    candidates = [c for c in candidates if "raw" not in c.name]
    if not candidates:
        raise FileNotFoundError(
            f"No MODIS baseline files found in {NDVI_MODIS_DIR}\n"
            "Run: python -m src.ingestion.fetch_modis_ndvi"
        )
    path = candidates[-1]
    log.info("Loading MODIS baseline from %s", path)
    df = pd.read_csv(path)
    log.info("  -> %d rows, %d mandis", len(df), df["mandi_id"].nunique())
    return df


def load_apmc() -> pd.DataFrame:
    """Load APMC historical prices (2021-2025)."""
    # Collect all APMC CSV files
    all_candidates = []
    for pattern in ["apmc_market_prices_2021_*.csv", "apmc_market_prices*.csv"]:
        for f in APMC_DIR.glob(pattern):
            if "source" not in f.name and f not in all_candidates:
                all_candidates.append(f)
    if not all_candidates:
        raise FileNotFoundError(
            f"No APMC historical files found in {APMC_DIR}\n"
            "Run: python -m src.ingestion.fetch_apmc_history"
        )
    # Prefer the file with the most rows (the real data, not sample)
    path = max(all_candidates, key=lambda f: f.stat().st_size)
    log.info("Loading APMC from %s", path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df.get("report_date", df.get("date", "")), errors="coerce"
    )
    # Standardise column names for joining
    if "modal_price" in df.columns:
        df["modal_price"] = pd.to_numeric(
            df["modal_price"].astype(str).str.replace(",", ""), errors="coerce"
        )
    log.info("  -> %d rows, %d unique dates", len(df),
             df["date"].nunique() if "date" in df.columns else 0)
    return df


def load_mandi_meta() -> pd.DataFrame:
    """Load mandi reference with lat/lon."""
    log.info("Loading mandi reference from %s", MANDI_FILE)
    df = pd.read_csv(MANDI_FILE)
    log.info("  -> %d mandis", len(df))
    return df


# ---------------------------------------------------------------------------
# Build mandi_id for APMC data (to match with MODIS mandi_ids)
# ---------------------------------------------------------------------------
def build_apmc_mandi_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create mandi_id column in APMC data to match MODIS mandi_id format.

    MODIS mandi_id: STATE_DISTRICT_MARKETCode_MARKETName
    e.g. ANDHRA_PRADESH_ANANTAPUR_1048_RAYADURG
    """
    df = df.copy()

    # Clean text fields
    for col in ("state_name", "district_name", "market_center"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r"[^A-Z0-9]", "_", regex=True)
                .str.replace(r"_+", "_", regex=True)
                .str.strip("_")
            )

    # Build mandi_id from available columns
    if "mandi_id" in df.columns and df["mandi_id"].notna().sum() > 0:
        # Already has mandi_id from enrichment
        return df

    # Build from components
    parts = []
    if "state_name" in df.columns:
        parts.append(df["state_name"])
    if "district_name" in df.columns:
        parts.append(df["district_name"])
    if "market_code" in df.columns:
        parts.append(df["market_code"].astype(str))
    if "market_center" in df.columns:
        parts.append(df["market_center"])

    if parts:
        df["mandi_id"] = parts[0]
        for p in parts[1:]:
            df["mandi_id"] = df["mandi_id"] + "_" + p

    return df


# ---------------------------------------------------------------------------
# Join logic
# ---------------------------------------------------------------------------
def join_ndvi_with_apmc(
    ndvi: pd.DataFrame,
    apmc: pd.DataFrame,
    baseline: pd.DataFrame,
    mandi_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join MODIS NDVI anomalies onto APMC price data.

    Join keys: (mandi_id, date)
    If mandi_id doesn't match, falls back to (state, district, date).
    """
    ndvi = ndvi.copy()
    apmc = apmc.copy()

    # -- Align date columns ---------------------------------------------
    ndvi["date"] = pd.to_datetime(ndvi["date"]).dt.normalize()
    apmc["date"] = pd.to_datetime(apmc["date"]).dt.normalize()

    # -- Add day-of-year for seasonal matching --------------------------
    ndvi["doy"] = ndvi["date"].dt.dayofyear

    # -- Prepare NDVI features for join ---------------------------------
    ndvi_features = ndvi[[
        "mandi_id", "date", "ndvi", "ndvi_anomaly", "doy",
    ]].copy()
    ndvi_features = ndvi_features.rename(columns={
        "ndvi": "modis_ndvi",
    })

    # -- Ensure APMC has mandi_id ---------------------------------------
    apmc = build_apmc_mandi_id(apmc)

    # -- Primary join: mandi_id + date ----------------------------------
    log.info("Joining on (mandi_id, date) ...")
    merged = apmc.merge(
        ndvi_features,
        on=["mandi_id", "date"],
        how="left",
        suffixes=("", "_ndvi"),
    )

    n_matched = merged["modis_ndvi"].notna().sum()
    match_pct = n_matched / len(merged) * 100 if len(merged) > 0 else 0
    log.info("Direct date match: %d / %d rows (%.1f%%)",
             n_matched, len(merged), match_pct)

    # -- Fallback 1: month-level match (MODIS monthly, APMC daily) ------
    if match_pct < 50:
        log.info("Low match rate. Trying month-level NDVI fallback...")
        ndvi_for_monthly = ndvi_features.copy()
        ndvi_for_monthly["year"] = ndvi_for_monthly["date"].dt.year
        ndvi_for_monthly["month"] = ndvi_for_monthly["date"].dt.month
        ndvi_monthly_avg = (
            ndvi_for_monthly.groupby(["mandi_id", "year", "month"])
            .agg({"modis_ndvi": "mean", "ndvi_anomaly": "mean"})
            .reset_index()
        )

        apmc_for_monthly = apmc.copy()
        apmc_for_monthly["year"] = apmc_for_monthly["date"].dt.year
        apmc_for_monthly["month"] = apmc_for_monthly["date"].dt.month

        merged["year"] = merged["date"].dt.year
        merged["month"] = merged["date"].dt.month

        # Vectorized month-level fill using merge
        merged = merged.merge(
            ndvi_monthly_avg.rename(columns={
                "modis_ndvi": "modis_ndvi_monthly",
                "ndvi_anomaly": "ndvi_anomaly_monthly",
            }),
            on=["mandi_id", "year", "month"],
            how="left",
        )
        # Fill NaN with monthly averages
        mask = merged["modis_ndvi"].isna()
        merged.loc[mask, "modis_ndvi"] = merged.loc[mask, "modis_ndvi_monthly"]
        merged.loc[mask, "ndvi_anomaly"] = merged.loc[mask, "ndvi_anomaly_monthly"]
        merged = merged.drop(columns=["modis_ndvi_monthly", "ndvi_anomaly_monthly"],
                            errors="ignore")

        n_after = merged["modis_ndvi"].notna().sum()
        log.info("After month fallback: %d / %d rows (%.1f%%)",
                 n_after, len(merged), n_after / len(merged) * 100)
        merged = merged.drop(columns=["year", "month"], errors="ignore")

    # -- Fallback 2: state-level seasonal NDVI --------------------------
    if merged["modis_ndvi"].notna().sum() / len(merged) < 0.5:
        log.info("Still low. Adding state-level seasonal NDVI fallback...")
        state_doy_avg = (
            ndvi_features
            .merge(mandi_meta[["mandi_id", "state"]], on="mandi_id", how="left")
            .groupby(["state", "doy"])["modis_ndvi"]
            .mean()
            .reset_index()
            .rename(columns={"modis_ndvi": "state_avg_ndvi"})
        )
        state_doy_anomaly = (
            ndvi_features
            .merge(mandi_meta[["mandi_id", "state"]], on="mandi_id", how="left")
            .groupby(["state", "doy"])["ndvi_anomaly"]
            .mean()
            .reset_index()
            .rename(columns={"ndvi_anomaly": "state_avg_ndvi_anomaly"})
        )
        state_fallback = state_doy_avg.merge(state_doy_anomaly, on=["state", "doy"])

        # Add state + DOY to unmatched rows
        unmatched_mask = merged["modis_ndvi"].isna()
        if "state_name" in merged.columns:
            merged.loc[unmatched_mask, "_doy"] = (
                pd.to_datetime(merged.loc[unmatched_mask, "date"]).dt.dayofyear
            )
            merged = merged.merge(
                state_fallback,
                left_on=["state_name", "_doy"],
                right_on=["state", "doy"],
                how="left",
                suffixes=("", "_state"),
            )
            # Fill missing NDVI with state averages
            merged["modis_ndvi"] = merged["modis_ndvi"].fillna(merged["state_avg_ndvi"])
            merged["ndvi_anomaly"] = merged["ndvi_anomaly"].fillna(
                merged["state_avg_ndvi_anomaly"]
            )
            merged = merged.drop(
                columns=["_doy", "state", "doy_state",
                         "state_avg_ndvi", "state_avg_ndvi_anomaly"],
                errors="ignore",
            )

            n_after = merged["modis_ndvi"].notna().sum()
            log.info("After state fallback: %d / %d rows (%.1f%%)",
                     n_after, len(merged), n_after / len(merged) * 100)

    # -- Add NDVI trend features ----------------------------------------
    # 7-day rolling NDVI anomaly trend
    merged = merged.sort_values(["mandi_id", "date"])
    merged["ndvi_anomaly_7d_avg"] = (
        merged.groupby("mandi_id")["ndvi_anomaly"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # NDVI anomaly direction (positive = improving, negative = declining)
    merged["ndvi_anomaly_direction"] = np.sign(merged["ndvi_anomaly"])

    # NDVI stress indicator (anomaly < -1.0 = significant stress)
    merged["ndvi_stress_flag"] = (merged["ndvi_anomaly"] < -1.0).astype(int)

    # NDVI surplus indicator (anomaly > 1.0 = excess vegetation)
    merged["ndvi_surplus_flag"] = (merged["ndvi_anomaly"] > 1.0).astype(int)

    return merged


# ---------------------------------------------------------------------------
# Feature engineering for combined dataset
# ---------------------------------------------------------------------------
def add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between NDVI anomaly and price data."""
    df = df.copy()

    # Price-NDVI interaction: high anomaly + high price = oversupply risk
    if "modal_price" in df.columns and "ndvi_anomaly" in df.columns:
        df["price_ndvi_interaction"] = df["modal_price"] * df["ndvi_anomaly"]

    # Normalized anomaly per commodity
    if "commodity" in df.columns and "ndvi_anomaly" in df.columns:
        def _safe_zscore(x):
            std_val = x.std()
            if std_val is None or std_val == 0 or pd.isna(std_val):
                return pd.Series(0.0, index=x.index)
            return (x - x.mean()) / max(std_val, 1e-6)
        df["ndvi_anomaly_commodity_z"] = (
            df.groupby("commodity")["ndvi_anomaly"]
            .transform(_safe_zscore)
        )

    # Days since last NDVI observation
    if "modis_ndvi" in df.columns:
        df["has_modis_obs"] = df["modis_ndvi"].notna().astype(int)
        df["days_since_ndvi_obs"] = (
            df.groupby("mandi_id")["has_modis_obs"]
            .cumsum()
        )

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  AgriVault NDVI Anomaly × APMC Joiner")
    print("=" * 70)

    # -- Load data ------------------------------------------------------
    ndvi = load_modis_current()
    baseline = load_modis_baseline()
    apmc = load_apmc()
    mandi_meta = load_mandi_meta()

    print(f"  MODIS NDVI rows : {len(ndvi):,}")
    print(f"  APMC rows       : {len(apmc):,}")
    print(f"  Mandis          : {mandi_meta['mandi_id'].nunique():,}")
    print("=" * 70)

    # -- Join -----------------------------------------------------------
    log.info("Joining NDVI anomalies with APMC prices ...")
    joined = join_ndvi_with_apmc(ndvi, apmc, baseline, mandi_meta)
    log.info("Joined dataset: %d rows, %d cols", len(joined), len(joined.columns))

    # -- Combined features ----------------------------------------------
    log.info("Adding combined features ...")
    joined = add_combined_features(joined)
    log.info("Final dataset: %d rows, %d cols", len(joined), len(joined.columns))

    # -- Write output ---------------------------------------------------
    output_file = OUTPUT_DIR / "ndvi_anomaly_apmc_joined.csv"
    joined.to_csv(output_file, index=False)
    log.info("Saved: %s", output_file)

    # -- Summary --------------------------------------------------------
    print()
    print("=" * 70)
    print("  JOIN COMPLETE")
    print("=" * 70)
    print(f"  Output file : {output_file}")
    print(f"  Total rows  : {len(joined):,}")
    print(f"  Columns     : {len(joined.columns)}")
    if "modis_ndvi" in joined.columns:
        n_with_ndvi = joined["modis_ndvi"].notna().sum()
        print(f"  Rows with NDVI: {n_with_ndvi:,} ({n_with_ndvi/len(joined)*100:.1f}%)")
    if "ndvi_anomaly" in joined.columns:
        valid = joined["ndvi_anomaly"].dropna()
        if len(valid) > 0:
            print(f"  Anomaly range : {valid.min():.3f} -> {valid.max():.3f}")
            print(f"  Anomaly mean  : {valid.mean():.3f}")
            print(f"  Stress flags  : {joined['ndvi_stress_flag'].sum():,} rows")
            print(f"  Surplus flags : {joined['ndvi_surplus_flag'].sum():,} rows")
    if "date" in joined.columns:
        print(f"  Date range    : {joined['date'].min().date()} -> {joined['date'].max().date()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
