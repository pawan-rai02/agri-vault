"""
AgriVault – Build Gold Price Features (Gold Layer) — pandas/PyArrow version
============================================================================
Reads all Silver-layer Parquet tables from S3 with boto3 + PyArrow
(no Spark dependency), joins them at the mandi × commodity × date grain,
and produces a feature table ready for quantile price forecasting.

Why pandas here instead of Spark?
    The APMC Silver Parquet (partitioned by state) is ~200-400 MB in memory
    after cleaning, well within 8-16 GB RAM. Using pandas avoids the
    BlockManager race condition seen on Windows when chaining Spark sessions.

Feature groups
--------------
1. Lagged modal price       : 1d, 7d, 14d, 30d
2. Rolling statistics       : 7d/14d/30d mean and std-dev
3. Price momentum           : (price - lag_7d) / lag_7d
4. Arrivals                 : raw arrivals_tonnes + 7d rolling mean
5. Weather aggregates       : 7d rolling temp_mean, precip_sum, humidity_mean
6. NDVI                     : daily forward-filled NDVI + 30d delta
7. Macro / CPI / WPI        : food CPI index + food WPI index (monthly join)
8. Temporal                 : day-of-week, day-of-month, month, is_weekend
9. Targets (supervised)     : target_price_7d, target_price_15d, target_price_30d

Input  : s3://agrivault-lake-pawan/standardized/
Output : s3://agrivault-lake-pawan/features/price_features/

Run
---
    python -m src.features.build_price_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage.s3_client import S3Client, load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_apmc(s3: S3Client) -> pd.DataFrame:
    # Don't request 'state' — it's a Hive partition key, stored in the directory
    # path (state=MAHARASHTRA/), not inside the Parquet file itself.
    # read_parquet_s3() extracts partition values from the S3 key path automatically.
    cols = [
        "market_code", "district", "market",
        "commodity", "date", "modal_price", "arrivals_tonnes",
        "latitude", "longitude",
    ]
    df = s3.read_parquet_s3("standardized/apmc/", columns=cols)
    df["date"] = pd.to_datetime(df["date"])

    # Build mandi_id surrogate
    df["market_code"] = df["market_code"].astype(str)
    df["market_clean"] = df["market"].str.upper().str.replace(r"[^A-Z0-9]", "_", regex=True)
    df["mandi_id"] = (
        df["state"].str.upper() + "_"
        + df["district"].str.upper() + "_"
        + df["market_code"] + "_"
        + df["market_clean"]
    )
    df = df.drop(columns=["market_clean"], errors="ignore")
    return df


def load_ndvi(s3: S3Client) -> pd.DataFrame:
    cols = ["mandi_id", "date", "ndvi"]
    df = s3.read_parquet_s3("standardized/ndvi/", columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    # Keep only what we need — drop any Hive partition keys injected by the reader
    # (e.g. 'state' from state=MAHARASHTRA/ paths). Keeping them would cause a
    # state_x / state_y column collision when merged with APMC.
    return df[["mandi_id", "date", "ndvi"]]


def load_weather(s3: S3Client) -> pd.DataFrame:
    cols = [
        "date", "latitude", "longitude",
        "temperature_mean", "precipitation_mm", "humidity", "wind_speed",
    ]
    df = s3.read_parquet_s3("standardized/weather/",
                           columns=[c for c in cols])
    df["date"] = pd.to_datetime(df["date"])
    # Round for grid join; drop any injected partition cols
    df["lat2"] = df["latitude"].round(2)
    df["lon2"] = df["longitude"].round(2)
    # Keep only necessary cols to avoid collision on merge
    keep = ["date", "lat2", "lon2",
            "temperature_mean", "precipitation_mm", "humidity", "wind_speed"]
    return df[[c for c in keep if c in df.columns]]


def load_cpi(s3: S3Client) -> pd.DataFrame:
    df = s3.read_parquet_s3("standardized/cpi/", columns=None)
    if df.empty:
        raise FileNotFoundError("No CPI data found at standardized/cpi/")
    df["date"] = pd.to_datetime(df["date"])
    # Filter to national food CPI: state='ALL INDIA', division='FOOD AND BEVERAGES'
    nat = df[
        (df.get("state", pd.Series(dtype=str)).str.strip().str.upper() == "ALL INDIA")
        & (df.get("division", pd.Series(dtype=str)).str.upper().str.contains("FOOD", na=False))
    ]
    if len(nat) == 0:
        nat = df  # fallback: use whatever is there
    monthly = (
        nat.groupby(nat["date"].dt.to_period("M"))["cpi_index"]
        .mean()
        .reset_index()
        .rename(columns={"date": "period", "cpi_index": "food_cpi_index"})
    )
    monthly["year"]  = monthly["period"].dt.year
    monthly["month"] = monthly["period"].dt.month
    return monthly[["year", "month", "food_cpi_index"]]


def load_wpi(s3: S3Client) -> pd.DataFrame:
    df = s3.read_parquet_s3("standardized/wpi/", columns=None)
    if df.empty:
        raise FileNotFoundError("No WPI data found at standardized/wpi/")
    df["date"] = pd.to_datetime(df["date"])
    # WPI level values: 'Group', 'Sub Group', 'Item', 'ALL', 'Major Group', etc.
    # Use level='ALL' for the composite index; fall back to full table average.
    if "level" in df.columns:
        level_all = df[df["level"].astype(str).str.strip().str.upper() == "ALL"]
        df = level_all if len(level_all) > 0 else df
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["wpi_index"]
        .mean()
        .reset_index()
        .rename(columns={"date": "period", "wpi_index": "food_wpi_index"})
    )
    monthly["year"]  = monthly["period"].dt.year
    monthly["month"] = monthly["period"].dt.month
    return monthly[["year", "month", "food_wpi_index"]]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _sort_group(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date within each mandi+commodity group (required for rolling)."""
    return df.sort_values(["mandi_id", "commodity", "date"])


def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["mandi_id", "commodity"])["modal_price"]
    for lag in (1, 7, 14, 30):
        df[f"price_lag_{lag}d"] = g.shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["mandi_id", "commodity"])["modal_price"]
    for w in (7, 14, 30):
        rolled = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"price_mean_{w}d"] = rolled
        rolled_std = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        df[f"price_std_{w}d"] = rolled_std
    return df


def add_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["price_momentum_7d"] = (
        (df["modal_price"] - df["price_lag_7d"]) / df["price_lag_7d"].clip(lower=1e-6)
    )
    return df


def add_arrivals_features(df: pd.DataFrame) -> pd.DataFrame:
    if "arrivals_tonnes" not in df.columns:
        return df
    g = df.groupby(["mandi_id", "commodity"])["arrivals_tonnes"]
    df["arrivals_mean_7d"] = g.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    return df


def join_weather(apmc: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid join on lat2/lon2 + date, then rolling weather aggregates.
    weather is expected to already have lat2/lon2 columns (built in load_weather).
    """
    apmc = apmc.copy()
    apmc["lat2"] = apmc["latitude"].round(2)
    apmc["lon2"] = apmc["longitude"].round(2)

    # weather already has lat2/lon2 from load_weather()
    wx_cols = ["date", "lat2", "lon2",
               "temperature_mean", "precipitation_mm", "humidity", "wind_speed"]
    wx = weather[[c for c in wx_cols if c in weather.columns]]
    merged = apmc.merge(wx, on=["date", "lat2", "lon2"], how="left")
    merged = merged.drop(columns=["lat2", "lon2"])

    # 7-day rolling weather per mandi (ordered by date)
    for col, agg in [("temperature_mean", "mean"),
                     ("precipitation_mm", "sum"),
                     ("humidity", "mean")]:
        if col in merged.columns:
            rolled = (
                merged.sort_values("date")
                .groupby("mandi_id")[col]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=1).agg(agg))
            )
            out_col = col.replace("temperature_mean", "temp_mean")\
                         .replace("precipitation_mm", "precip_sum")\
                         .replace("humidity", "humidity_mean")
            merged[f"{out_col}_7d"] = rolled

    return merged


def join_ndvi(apmc: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily NDVI and add 30-day NDVI delta."""
    merged = apmc.merge(ndvi, on=["mandi_id", "date"], how="left")

    # 30-day lagged NDVI per mandi (LOCF)
    merged = merged.sort_values(["mandi_id", "date"])
    merged["ndvi_lag_30d"] = (
        merged.groupby("mandi_id")["ndvi"]
        .transform(lambda x: x.shift(30))
    )
    merged["ndvi_delta_30d"] = merged["ndvi"] - merged["ndvi_lag_30d"]
    return merged


def join_macro(df: pd.DataFrame,
               cpi: pd.DataFrame, wpi: pd.DataFrame) -> pd.DataFrame:
    """Monthly join on year + month."""
    df["_year"]  = df["date"].dt.year
    df["_month"] = df["date"].dt.month
    df = df.merge(cpi, left_on=["_year", "_month"],
                  right_on=["year", "month"], how="left")
    df = df.drop(columns=["year", "month"], errors="ignore")
    df = df.merge(wpi, left_on=["_year", "_month"],
                  right_on=["year", "month"], how="left")
    df = df.drop(columns=["year", "month", "_year", "_month"], errors="ignore")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking price targets at 7d, 15d, 30d."""
    g = df.groupby(["mandi_id", "commodity"])["modal_price"]
    for h in (7, 15, 30):
        df[f"target_price_{h}d"] = g.shift(-h)
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

    # ── Load Silver ───────────────────────────────────────────────────────
    log.info("Loading APMC Silver...")
    apmc = load_apmc(s3)

    log.info("Loading NDVI Silver...")
    ndvi = load_ndvi(s3)

    log.info("Loading Weather Silver...")
    weather = load_weather(s3)

    log.info("Loading CPI Silver...")
    cpi = load_cpi(s3)

    log.info("Loading WPI Silver...")
    wpi = load_wpi(s3)

    # ── Join + feature engineering ────────────────────────────────────────
    # ── Update join_weather to expect pre-built lat2/lon2 in weather ─────
    log.info("Joining weather...")
    df = join_weather(apmc, weather)

    log.info("Joining NDVI...")
    df = join_ndvi(df, ndvi)

    log.info("Joining macro (CPI + WPI)...")
    df = join_macro(df, cpi, wpi)

    # Recover 'state' from 'state_x' if merge collision occurred
    if "state" not in df.columns and "state_x" in df.columns:
        log.warning("'state' column missing after joins — recovering from 'state_x'")
        df = df.rename(columns={"state_x": "state"})
        df = df.drop(columns=["state_y"], errors="ignore")

    log.info("Adding temporal features...")
    df = add_temporal_features(df)

    log.info("Sorting by mandi/commodity/date for windowed features...")
    df = _sort_group(df)

    log.info("Adding price lags...")
    df = add_price_lags(df)

    log.info("Adding rolling stats...")
    df = add_rolling_stats(df)

    log.info("Adding price momentum...")
    df = add_price_momentum(df)

    log.info("Adding arrivals features...")
    df = add_arrivals_features(df)

    log.info("Adding targets (7d, 15d, 30d)...")
    df = add_targets(df)

    log.info("Final feature table: %d rows, %d cols", len(df), len(df.columns))

    # ── Write to S3 in state chunks ───────────────────────────────────────
    out_prefix = s3.key("features", "price_features")

    if "state" not in df.columns:
        log.warning("'state' column not found — writing as single partition")
        key = f"{out_prefix}/state=ALL/price_features.parquet"
        s3.write_parquet_s3(df, key)
    else:
        states = df["state"].dropna().unique()
        log.info("Writing %d state partitions to s3://%s/%s/",
                 len(states), s3.bucket, out_prefix)
        total_written = 0
        for state in states:
            sdf = df[df["state"] == state].copy()
            safe_state = str(state).replace(" ", "_").replace("/", "-")
            key = f"{out_prefix}/state={safe_state}/price_features.parquet"
            s3.write_parquet_s3(sdf, key)
            total_written += len(sdf)
        log.info("✓ Gold price features written: %d total rows across %d states",
                 total_written, len(states))

    log.info("Gold price feature build complete.")


if __name__ == "__main__":
    main()
