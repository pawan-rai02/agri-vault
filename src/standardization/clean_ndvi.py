"""
AgriVault – Clean NDVI Sentinel-2 Data (Silver Layer)
======================================================
Reads raw NDVI CSV from S3 (already in 0–1 scale from Sentinel-2,
NOT the MODIS ×10000 convention), cleans and forward-fills monthly
observations to daily resolution, writes to standardized/ndvi/.

Raw schema (confirmed from data inspection):
    mandi_id, market_code, mandi_name, district, state,
    latitude, longitude, date (YYYY-MM-DD), ndvi (float, 0–1 range)

The raw data is monthly (one row per mandi per month, ~8 months of 2025).
Forward-filling to daily is done in PySpark using a date spine join.

Input  : s3://agrivault-lake-pawan/raw/ndvi/ndvi_sentinel2_2025.csv
Output : s3://agrivault-lake-pawan/standardized/ndvi/

Run
---
    python -m src.standardization.clean_ndvi
"""

from __future__ import annotations

import logging

from pyspark.sql import functions as F, DataFrame, Window, SparkSession

from src.standardization.spark_session import get_spark, load_config, bucket_uri

log = logging.getLogger(__name__)

# NDVI physically valid range (Sentinel-2 normalized difference)
NDVI_MIN = -0.2
NDVI_MAX = 1.0


def clean_ndvi_base(df: DataFrame) -> DataFrame:
    """
    Clean the raw monthly NDVI DataFrame:
    - Parse date
    - Cast NDVI to double, clamp to valid range
    - Normalise text fields
    - Deduplicate by (mandi_id, date)
    """
    # ── 1. Parse date ──────────────────────────────────────────────────────
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

    # ── 2. Cast and clamp NDVI ─────────────────────────────────────────────
    df = df.withColumn("ndvi", F.col("ndvi").cast("double"))
    df = df.withColumn(
        "ndvi",
        F.when(
            F.col("ndvi").between(NDVI_MIN, NDVI_MAX),
            F.col("ndvi"),
        ).otherwise(None),   # nullify physically invalid values
    )

    # ── 3. Cast coordinates ────────────────────────────────────────────────
    df = df.withColumn("latitude",  F.col("latitude").cast("double"))
    df = df.withColumn("longitude", F.col("longitude").cast("double"))
    df = df.withColumn("market_code", F.col("market_code").cast("long"))

    # ── 4. Normalise text ──────────────────────────────────────────────────
    for col in ("mandi_name", "district", "state"):
        df = df.withColumn(col, F.trim(F.upper(F.col(col))))

    # ── 5. Drop rows missing key identifiers or NDVI ──────────────────────
    df = df.filter(
        F.col("mandi_id").isNotNull()
        & F.col("date").isNotNull()
        & F.col("ndvi").isNotNull()
    )

    # ── 6. Deduplicate ─────────────────────────────────────────────────────
    df = df.dropDuplicates(["mandi_id", "date"])

    return df


def forward_fill_to_daily(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Forward-fill monthly NDVI observations to daily resolution per mandi.

    Strategy:
    1. Build a complete daily date spine for the observed date range
    2. Cross-join each mandi with the date spine
    3. Join NDVI values, then use last-observation-carried-forward (LOCF)
       via a Window with unboundedPreceding to fill nulls

    This produces one row per (mandi_id, date) for every calendar day.
    """
    # ── Get date range from data ───────────────────────────────────────────
    date_bounds = df.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
    ).collect()[0]

    min_date = date_bounds["min_date"]
    max_date = date_bounds["max_date"]

    log.info("Building daily spine from %s to %s", min_date, max_date)

    # ── Build daily date spine ─────────────────────────────────────────────
    date_spine = spark.sql(
        f"SELECT sequence(date'{min_date}', date'{max_date}', interval 1 day) AS date_arr"
    ).select(F.explode(F.col("date_arr")).alias("date"))

    # ── Get unique mandis with their metadata ─────────────────────────────
    mandi_meta = df.select(
        "mandi_id", "market_code", "mandi_name", "district", "state",
        "latitude", "longitude",
    ).dropDuplicates(["mandi_id"])

    # ── Cross-join mandis × dates ─────────────────────────────────────────
    mandi_dates = mandi_meta.crossJoin(date_spine)

    # ── Left-join actual NDVI observations ───────────────────────────────
    ndvi_obs = df.select("mandi_id", "date", "ndvi")
    joined = mandi_dates.join(ndvi_obs, on=["mandi_id", "date"], how="left")

    # ── Forward-fill NDVI using last non-null value ───────────────────────
    w = (
        Window
        .partitionBy("mandi_id")
        .orderBy("date")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    joined = joined.withColumn(
        "ndvi_filled",
        F.last(F.col("ndvi"), ignorenulls=True).over(w),
    )

    # Keep the filled value as ndvi, also keep a flag for observation days
    joined = (
        joined
        .withColumn("is_observed", F.col("ndvi").isNotNull().cast("integer"))
        .withColumn("ndvi", F.col("ndvi_filled"))
        .drop("ndvi_filled")
        .filter(F.col("ndvi").isNotNull())   # drop leading days before first obs
    )

    return joined


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    spark = get_spark("clean-ndvi")

    input_uri  = bucket_uri(cfg, "raw",          "ndvi/ndvi_sentinel2_2025.csv")
    output_uri = bucket_uri(cfg, "standardized", "ndvi")

    log.info("Reading NDVI from %s", input_uri)
    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(input_uri)
    )
    log.info("Raw rows: %d | unique mandis: %d",
             raw.count(), raw.select("mandi_id").distinct().count())

    cleaned = clean_ndvi_base(raw)
    log.info("After base cleaning: %d rows", cleaned.count())

    daily = forward_fill_to_daily(cleaned, spark)
    log.info("After daily forward-fill: %d rows", daily.count())

    log.info("Writing to %s", output_uri)
    (
        daily.write
        .mode("overwrite")
        .partitionBy("state")
        .parquet(output_uri)
    )

    log.info("✓ NDVI standardized (daily) written to S3")
    spark.stop()


if __name__ == "__main__":
    main()
