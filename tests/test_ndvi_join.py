"""
AgriVault – tests/test_ndvi_join.py
=====================================
Unit tests for the NDVI Silver cleaning pipeline:
  - clean_ndvi_base: date parsing, value clamping, null handling, dedup
  - forward_fill_to_daily: spine construction, LOCF, is_observed flag

Tests use in-process PySpark (local[2]) with a tiny synthetic DataFrame
so no S3 access is required.
"""

from __future__ import annotations

import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.standardization.clean_ndvi import clean_ndvi_base, forward_fill_to_daily

# Note: the `spark` fixture is defined in tests/conftest.py (session-scoped)
# and shared across all PySpark tests.


# ---------------------------------------------------------------------------
# Schema matching the raw NDVI CSV
# ---------------------------------------------------------------------------

RAW_SCHEMA = StructType([
    StructField("mandi_id",    StringType(), True),
    StructField("market_code", StringType(), True),
    StructField("mandi_name",  StringType(), True),
    StructField("district",    StringType(), True),
    StructField("state",       StringType(), True),
    StructField("latitude",    StringType(), True),
    StructField("longitude",   StringType(), True),
    StructField("date",        StringType(), True),
    StructField("ndvi",        StringType(), True),
])


def _make_raw(spark: SparkSession, rows: list[tuple]) -> object:
    return spark.createDataFrame(rows, schema=RAW_SCHEMA)


# ---------------------------------------------------------------------------
# Tests: clean_ndvi_base
# ---------------------------------------------------------------------------

class TestCleanNdviBase:

    def test_date_parsed(self, spark):
        """date column must become DateType."""
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "0.35"),
        ])
        cleaned = clean_ndvi_base(df)
        dtype = dict(cleaned.dtypes)["date"]
        assert dtype == "date", f"Expected date, got {dtype}"

    def test_ndvi_cast_to_double(self, spark):
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "0.45"),
        ])
        cleaned = clean_ndvi_base(df)
        dtype = dict(cleaned.dtypes)["ndvi"]
        assert dtype == "double"

    def test_ndvi_out_of_range_nullified(self, spark):
        """NDVI values outside [-0.2, 1.0] should become null and be dropped."""
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "1.5"),   # > 1.0
            ("M002", "101", "Mandi B", "Dist2", "state2",
             "17.5", "78.3", "2025-02-01", "0.30"),  # valid
        ])
        cleaned = clean_ndvi_base(df)
        rows = cleaned.collect()
        mandi_ids = {r.mandi_id for r in rows}
        assert "M001" not in mandi_ids, "Out-of-range NDVI row should be dropped"
        assert "M002" in mandi_ids

    def test_null_ndvi_row_dropped(self, spark):
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", None),
        ])
        cleaned = clean_ndvi_base(df)
        assert cleaned.count() == 0

    def test_null_date_row_dropped(self, spark):
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "bad-date", "0.40"),
        ])
        cleaned = clean_ndvi_base(df)
        assert cleaned.count() == 0

    def test_text_fields_uppercased(self, spark):
        df = _make_raw(spark, [
            ("M001", "100", "mandi a", "district one", "state one",
             "17.5", "78.3", "2025-01-01", "0.35"),
        ])
        cleaned = clean_ndvi_base(df)
        row = cleaned.collect()[0]
        assert row.mandi_name == "MANDI A"
        assert row.district == "DISTRICT ONE"
        assert row.state == "STATE ONE"

    def test_duplicates_removed(self, spark):
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "0.35"),
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "0.40"),  # duplicate (mandi_id, date)
        ])
        cleaned = clean_ndvi_base(df)
        assert cleaned.count() == 1

    def test_negative_ndvi_valid(self, spark):
        """NDVI values in [-0.2, 0) are physically valid (water/soil)."""
        df = _make_raw(spark, [
            ("M001", "100", "Mandi A", "Dist1", "state1",
             "17.5", "78.3", "2025-01-01", "-0.15"),
        ])
        cleaned = clean_ndvi_base(df)
        row = cleaned.collect()[0]
        assert row.ndvi == pytest.approx(-0.15, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: forward_fill_to_daily
# ---------------------------------------------------------------------------

class TestForwardFillToDaily:

    def _base_cleaned(self, spark: SparkSession) -> object:
        """Two mandis with monthly observations — January and March 2025."""
        from pyspark.sql.types import (
            DateType, DoubleType, LongType, StringType, StructField, StructType,
        )

        rows = [
            ("MAHARASHTRA_PUNE_100_PUNE",     100, "PUNE",     "PUNE",     "MAHARASHTRA",
             18.52, 73.85, datetime.date(2025, 1, 1), 0.30),
            ("MAHARASHTRA_PUNE_100_PUNE",     100, "PUNE",     "PUNE",     "MAHARASHTRA",
             18.52, 73.85, datetime.date(2025, 3, 1), 0.45),
            ("KARNATAKA_DHARWAD_200_DHARWAD", 200, "DHARWAD",  "DHARWAD",  "KARNATAKA",
             15.46, 75.01, datetime.date(2025, 1, 1), 0.20),
        ]
        schema = StructType([
            StructField("mandi_id",    StringType(), True),
            StructField("market_code", LongType(),   True),
            StructField("mandi_name",  StringType(), True),
            StructField("district",    StringType(), True),
            StructField("state",       StringType(), True),
            StructField("latitude",    DoubleType(), True),
            StructField("longitude",   DoubleType(), True),
            StructField("date",        DateType(),   True),
            StructField("ndvi",        DoubleType(), True),
        ])
        # Workaround: create from pandas for clean date handling
        import pandas as pd
        pdf = pd.DataFrame(rows, columns=[f.name for f in schema.fields])
        pdf["date"] = pd.to_datetime(pdf["date"]).dt.date
        return spark.createDataFrame(pdf)

    def test_output_has_daily_rows(self, spark):
        """After forward-fill the number of rows per mandi equals day-count in range."""
        cleaned = self._base_cleaned(spark)
        daily = forward_fill_to_daily(cleaned, spark)

        # Date range: 2025-01-01 to 2025-03-01 = 60 days
        # Two mandis → expect 2 × 60 rows = 120 (both mandis have obs on Jan 1)
        count = daily.count()
        assert count == 2 * 60, f"Expected 120 daily rows, got {count}"

    def test_ndvi_filled_between_observations(self, spark):
        """Days between Jan and Mar should carry Jan NDVI via LOCF."""
        cleaned = self._base_cleaned(spark)
        daily = forward_fill_to_daily(cleaned, spark)

        feb = (
            daily
            .filter(
                (F.col("mandi_id") == "MAHARASHTRA_PUNE_100_PUNE")
                & (F.month("date") == 2)
            )
        )
        feb_rows = feb.collect()
        assert len(feb_rows) == 28, f"Expected 28 Feb rows, got {len(feb_rows)}"
        for row in feb_rows:
            assert row.ndvi == pytest.approx(0.30, abs=1e-6), (
                f"Feb NDVI should be forward-filled from Jan (0.30), got {row.ndvi}"
            )

    def test_is_observed_flag(self, spark):
        """is_observed=1 only on days with actual data."""
        cleaned = self._base_cleaned(spark)
        daily = forward_fill_to_daily(cleaned, spark)

        pune = daily.filter(F.col("mandi_id") == "MAHARASHTRA_PUNE_100_PUNE")
        observed = pune.filter(F.col("is_observed") == 1).count()
        # Should have exactly 2 observed days (Jan 1 and Mar 1)
        assert observed == 2, f"Expected 2 observed days, got {observed}"

    def test_no_nulls_in_ndvi_output(self, spark):
        """After forward-fill there should be no null NDVI in the output."""
        cleaned = self._base_cleaned(spark)
        daily = forward_fill_to_daily(cleaned, spark)
        null_count = daily.filter(F.col("ndvi").isNull()).count()
        assert null_count == 0, f"Found {null_count} null NDVI rows after forward-fill"

    def test_mandi_isolation(self, spark):
        """Forward-fill must not bleed NDVI from one mandi to another."""
        cleaned = self._base_cleaned(spark)
        daily = forward_fill_to_daily(cleaned, spark)

        karnataka_jan = (
            daily
            .filter(
                (F.col("mandi_id") == "KARNATAKA_DHARWAD_200_DHARWAD")
                & (F.month("date") == 1)
                & (F.dayofmonth("date") == 1)
            )
            .collect()
        )
        assert len(karnataka_jan) == 1
        assert karnataka_jan[0].ndvi == pytest.approx(0.20, abs=1e-6)
