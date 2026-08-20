"""
AgriVault – Clean WDRA Warehouse Receipts (Silver Layer)
=========================================================
Reads all 26 state-level WDRA-*.csv files from S3, unions them,
standardizes column names and types, writes to standardized/wdra/.

Input  : s3://agrivault-lake-pawan/raw/wdra/WDRA-*.csv
Output : s3://agrivault-lake-pawan/standardized/wdra/

Run
---
    python -m src.standardization.clean_wdra
"""

from __future__ import annotations

import logging
import re

from pyspark.sql import functions as F, DataFrame, SparkSession

from src.standardization.spark_session import get_spark, load_config, bucket_uri

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping — WDRA CSVs have inconsistent column names across states.
# We normalise everything to this canonical set.
# ---------------------------------------------------------------------------
CANONICAL_RENAME: dict[str, str] = {
    # Warehouse identity
    "warehouse_name":             "warehouse_name",
    "warehousename":              "warehouse_name",
    "name_of_warehouse":          "warehouse_name",
    "warehouse registration no":  "warehouse_reg_no",
    "warehouse_reg_no":           "warehouse_reg_no",
    "registration_no":            "warehouse_reg_no",
    "reg_no":                     "warehouse_reg_no",

    # Location
    "district":       "district",
    "district_name":  "district",
    "taluk":          "taluk",
    "taluka":         "taluk",
    "village":        "village",

    # Capacity
    "capacity":              "capacity_mt",
    "storage_capacity":      "capacity_mt",
    "capacity_mt":           "capacity_mt",
    "capacity(mt)":          "capacity_mt",
    "total_capacity":        "capacity_mt",

    # Commodity
    "commodity":       "commodity",
    "commodities":     "commodity",
    "commodity_name":  "commodity",

    # Dates
    "date_of_registration":    "registration_date",
    "registration_date":       "registration_date",
    "valid_upto":              "valid_upto",
    "valid upto":              "valid_upto",
    "validity_date":           "valid_upto",

    # Depositor / receipts
    "depositor_name":     "depositor_name",
    "receipt_no":         "receipt_no",
    "quantity_mt":        "quantity_mt",
    "quantity":           "quantity_mt",
    "value_inr":          "value_inr",
}


def normalise_col(name: str) -> str:
    """Lowercase, strip, replace whitespace/special chars → underscores."""
    name = name.strip().lower()
    name = re.sub(r"[\s\-/\\]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = name.strip("_")
    return name


def rename_columns(df: DataFrame) -> DataFrame:
    """Apply canonical column renaming based on normalised name matching."""
    mapping = {col: CANONICAL_RENAME.get(normalise_col(col), normalise_col(col))
               for col in df.columns}
    for old, new in mapping.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def clean_wdra(df: DataFrame) -> DataFrame:
    """Clean and standardize a WDRA DataFrame."""

    df = rename_columns(df)

    # ── Cast numeric columns ───────────────────────────────────────────────
    for col in ("capacity_mt", "quantity_mt", "value_inr"):
        if col in df.columns:
            # Remove commas/spaces that sometimes appear in numbers
            df = df.withColumn(col, F.regexp_replace(F.col(col).cast("string"), "[, ]", ""))
            df = df.withColumn(col, F.col(col).cast("double"))

    # ── Parse dates (various formats seen in WDRA data) ───────────────────
    for date_col in ("registration_date", "valid_upto"):
        if date_col in df.columns:
            df = df.withColumn(
                date_col,
                F.coalesce(
                    F.to_date(F.col(date_col), "dd/MM/yyyy"),
                    F.to_date(F.col(date_col), "dd-MM-yyyy"),
                    F.to_date(F.col(date_col), "yyyy-MM-dd"),
                    F.to_date(F.col(date_col), "d/M/yyyy"),
                    F.to_date(F.col(date_col), "dd-MMM-yyyy"),
                ),
            )

    # ── Normalise text ─────────────────────────────────────────────────────
    for col in ("warehouse_name", "district", "taluk", "commodity", "state"):
        if col in df.columns:
            df = df.withColumn(col, F.trim(F.upper(F.col(col))))

    # ── Filter: must have a warehouse name and state ───────────────────────
    filters = []
    if "warehouse_name" in df.columns:
        filters.append(F.col("warehouse_name").isNotNull())
    if "state" in df.columns:
        filters.append(F.col("state").isNotNull())

    for flt in filters:
        df = df.filter(flt)

    # ── Deduplicate ────────────────────────────────────────────────────────
    dedup_cols = [c for c in ("warehouse_reg_no", "state", "district") if c in df.columns]
    if dedup_cols:
        df = df.dropDuplicates(dedup_cols)

    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    spark = get_spark("clean-wdra")

    # Read all state CSV files in one go using wildcard
    input_uri  = bucket_uri(cfg, "raw",          "wdra/")
    output_uri = bucket_uri(cfg, "standardized", "wdra")

    log.info("Reading WDRA files from %s", input_uri)

    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(f"{input_uri}*.csv")
    )

    # Inject state from file path
    raw = raw.withColumn(
        "source_file", F.input_file_name()
    ).withColumn(
        "state",
        F.regexp_extract(F.col("source_file"), r"WDRA-([^./]+)\.csv", 1),
    ).drop("source_file")

    log.info("Raw rows: %d", raw.count())

    cleaned = clean_wdra(raw)
    log.info("Clean rows: %d", cleaned.count())

    log.info("Writing to %s", output_uri)
    cleaned.write.mode("overwrite").partitionBy("state").parquet(output_uri)

    log.info("✓ WDRA standardized written to S3")
    spark.stop()


if __name__ == "__main__":
    main()
