"""
AgriVault – Clean APMC Market Prices (Silver Layer)
=====================================================
Reads raw APMC CSV from S3, applies cleaning, writes partitioned Parquet
back to S3 under standardized/apmc/.

Input  : s3://agrivault-lake-pawan/raw/apmc/apmc_market_prices.csv
Output : s3://agrivault-lake-pawan/standardized/apmc/  (partitioned by state)

Run
---
    python -m src.standardization.clean_apmc
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import functions as F, DataFrame

from src.standardization.spark_session import get_spark, load_config, bucket_uri

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name normalisation map
# Raw APMC CSV column names → clean snake_case equivalents
# ---------------------------------------------------------------------------
RENAME = {
    # Actual APMC CSV column names → clean snake_case
    "state_name":      "state",
    "state_code":      "state_code",
    "district_name":   "district",
    "district_code":   "district_code",
    "market_center":   "market",
    "market_code":     "market_code",
    "commodity_type":  "commodity_type",
    "commodity":       "commodity",          # already clean name
    "variety":         "variety",
    "origin":          "origin",
    "min_price":       "min_price",
    "max_price":       "max_price",
    "modal_price":     "modal_price",
    "report_date":     "date_raw",           # the actual date column name
    "arrivals_tonnes": "arrivals_tonnes",    # already correct
    "arrivals_unit":   "arrivals_unit",
    "price_unit":      "price_unit",
    "latitude":        "latitude",
    "longitude":       "longitude",
}


def clean_apmc(df: DataFrame) -> DataFrame:
    """Apply all cleaning transforms to the raw APMC DataFrame."""

    # ── 1. Rename to snake_case ────────────────────────────────────────────
    for old, new in RENAME.items():
        if old in df.columns and old != new:
            df = df.withColumnRenamed(old, new)

    # ── 2. Parse date ──────────────────────────────────────────────────────
    # Raw format examples: "15/01/2025", "2025-01-15"
    df = df.withColumn(
        "date",
        F.coalesce(
            F.to_date(F.col("date_raw"), "dd/MM/yyyy"),
            F.to_date(F.col("date_raw"), "yyyy-MM-dd"),
            F.to_date(F.col("date_raw"), "d/M/yyyy"),
        ),
    ).drop("date_raw")

    # ── 3. Cast numeric columns ────────────────────────────────────────────
    for col in ("min_price", "max_price", "modal_price", "arrivals_tonnes",
                "latitude", "longitude"):
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))

    df = df.withColumn("market_code", F.col("market_code").cast("long"))

    # ── 4. Normalise text fields ───────────────────────────────────────────
    for col in ("state", "district", "market", "commodity", "variety",
                "commodity_type", "origin"):
        if col in df.columns:
            df = df.withColumn(col, F.trim(F.upper(F.col(col))))

    # ── 5. Filter invalid rows ─────────────────────────────────────────────
    df = df.filter(
        F.col("date").isNotNull()
        & F.col("modal_price").isNotNull()
        & (F.col("modal_price") > 0)
        & F.col("state").isNotNull()
        & F.col("commodity").isNotNull()
    )

    # ── 6. Clamp unrealistic prices (> ₹1 crore/quintal treated as data error) ─
    df = df.filter(F.col("modal_price") < 1_000_000)

    # ── 7. Deduplicate ─────────────────────────────────────────────────────
    dedup_keys = ["date", "state", "district", "market", "commodity", "variety"]
    existing_dedup = [c for c in dedup_keys if c in df.columns]
    df = df.dropDuplicates(existing_dedup)

    # ── 8. Add year/month partition helpers ───────────────────────────────
    df = (
        df
        .withColumn("year",  F.year("date"))
        .withColumn("month", F.month("date"))
    )

    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    spark = get_spark("clean-apmc")

    input_uri  = bucket_uri(cfg, "raw",          "apmc/apmc_market_prices.csv")
    output_uri = bucket_uri(cfg, "standardized", "apmc")

    log.info("Reading APMC from %s", input_uri)
    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")   # we cast manually
        .option("multiLine", "false")
        .csv(input_uri)
    )
    log.info("Schema: %s", raw.schema.simpleString()[:200])

    cleaned = clean_apmc(raw)

    log.info("Writing to %s", output_uri)
    (
        cleaned.write
        .mode("overwrite")
        .partitionBy("state")
        .parquet(output_uri)
    )

    log.info("✓ APMC standardized written to S3")
    spark.stop()


if __name__ == "__main__":
    main()
