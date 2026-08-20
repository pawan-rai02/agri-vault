"""
AgriVault – Clean Weather Data (Silver Layer)
==============================================
Reads daily weather parquet from S3, cleans and standardizes,
writes back to S3 under standardized/weather/.

Schema confirmed from raw:
    date (datetime64), temperature_mean, temperature_max, temperature_min,
    precipitation_mm, humidity, wind_speed, latitude, longitude  (all float64)

Input  : s3://agrivault-lake-pawan/raw/weather/weather_daily.parquet
Output : s3://agrivault-lake-pawan/standardized/weather/

Run
---
    python -m src.standardization.clean_weather
"""

from __future__ import annotations

import logging

from pyspark.sql import functions as F, DataFrame

from src.standardization.spark_session import get_spark, load_config, bucket_uri

log = logging.getLogger(__name__)

# Bounds check constants — reasonable ranges for India
_INDIA_LAT  = (6.0, 38.0)
_INDIA_LON  = (68.0, 98.0)
_TEMP_MIN   = -5.0    # °C, extreme cold
_TEMP_MAX   = 55.0    # °C, extreme heat
_PREC_MAX   = 500.0   # mm/day, physically plausible maximum
_HUM_RANGE  = (0.0, 100.0)
_WIND_MAX   = 150.0   # km/h


def clean_weather(df: DataFrame) -> DataFrame:
    """Validate and clean the weather DataFrame."""

    # ── 1. Ensure date is date type ───────────────────────────────────────
    df = df.withColumn("date", F.to_date(F.col("date")))

    # ── 2. Cast all numeric cols (already float64 in parquet, but be explicit) ─
    for col in (
        "temperature_mean", "temperature_max", "temperature_min",
        "precipitation_mm", "humidity", "wind_speed", "latitude", "longitude",
    ):
        df = df.withColumn(col, F.col(col).cast("double"))

    # ── 3. Drop rows with null date or coordinates ────────────────────────
    df = df.filter(
        F.col("date").isNotNull()
        & F.col("latitude").isNotNull()
        & F.col("longitude").isNotNull()
    )

    # ── 4. Validate coordinates are within India bounding box ─────────────
    df = df.filter(
        (F.col("latitude")  >= _INDIA_LAT[0]) & (F.col("latitude")  <= _INDIA_LAT[1])
        & (F.col("longitude") >= _INDIA_LON[0]) & (F.col("longitude") <= _INDIA_LON[1])
    )

    # ── 5. Clamp/nullify physically impossible values ─────────────────────
    df = (
        df
        .withColumn("temperature_mean",
            F.when(
                F.col("temperature_mean").between(_TEMP_MIN, _TEMP_MAX),
                F.col("temperature_mean"),
            )
        )
        .withColumn("temperature_max",
            F.when(
                F.col("temperature_max").between(_TEMP_MIN, _TEMP_MAX),
                F.col("temperature_max"),
            )
        )
        .withColumn("temperature_min",
            F.when(
                F.col("temperature_min").between(_TEMP_MIN, _TEMP_MAX),
                F.col("temperature_min"),
            )
        )
        .withColumn("precipitation_mm",
            F.when(
                (F.col("precipitation_mm") >= 0) & (F.col("precipitation_mm") <= _PREC_MAX),
                F.col("precipitation_mm"),
            ).otherwise(0.0)
        )
        .withColumn("humidity",
            F.when(
                F.col("humidity").between(_HUM_RANGE[0], _HUM_RANGE[1]),
                F.col("humidity"),
            )
        )
        .withColumn("wind_speed",
            F.when(
                (F.col("wind_speed") >= 0) & (F.col("wind_speed") <= _WIND_MAX),
                F.col("wind_speed"),
            )
        )
    )

    # ── 6. Round coordinates to 4dp for join key consistency ─────────────
    df = (
        df
        .withColumn("latitude",  F.round(F.col("latitude"),  4))
        .withColumn("longitude", F.round(F.col("longitude"), 4))
    )

    # ── 7. Deduplicate by date + location ─────────────────────────────────
    df = df.dropDuplicates(["date", "latitude", "longitude"])

    # ── 8. Derived features useful for the model ──────────────────────────
    df = df.withColumn(
        "temp_range",
        F.col("temperature_max") - F.col("temperature_min"),
    )

    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    spark = get_spark("clean-weather")

    input_uri  = bucket_uri(cfg, "raw",          "weather/weather_daily.parquet")
    output_uri = bucket_uri(cfg, "standardized", "weather")

    log.info("Reading weather from %s", input_uri)
    raw = spark.read.parquet(input_uri)
    log.info("Schema: %s", raw.schema.simpleString()[:200])

    cleaned = clean_weather(raw)

    log.info("Writing to %s", output_uri)
    cleaned.write.mode("overwrite").parquet(output_uri)

    log.info("✓ Weather standardized written to S3")
    spark.stop()


if __name__ == "__main__":
    main()
