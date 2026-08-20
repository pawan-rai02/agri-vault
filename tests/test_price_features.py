"""
AgriVault – tests/test_price_features.py
==========================================
Unit tests for the Gold price feature engineering pipeline.

Tests verify:
  - Price lag computation
  - Rolling statistics (mean, std)
  - Price momentum calculation
  - Arrivals rolling features
  - Temporal feature extraction
  - Target forward-looking price computation
  - NDVI join and delta calculation
  - Macro (CPI/WPI) join logic
  - Commodity category mapping

Uses synthetic pandas DataFrames — no S3 or Spark required.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.features.build_price_features import (
    add_arrivals_features,
    add_price_lags,
    add_price_momentum,
    add_rolling_stats,
    add_temporal_features,
    add_targets,
    join_macro,
    join_ndvi,
    join_weather,
)
from src.features.build_risk_features import map_commodity_category


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_apmc() -> pd.DataFrame:
    """3 mandis × 1 commodity × 10 days of price data."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    rows = []
    for mandi in ["M_A", "M_B", "M_C"]:
        for i, d in enumerate(dates):
            rows.append({
                "mandi_id": mandi,
                "commodity": "WHEAT",
                "date": d,
                "modal_price": 100.0 + i * 2 + hash(mandi) % 5,
                "arrivals_tonnes": 10.0 + i,
                "latitude": 18.5,
                "longitude": 73.8,
                "state": "MAHARASHTRA",
                "district": "PUNE",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_weather() -> pd.DataFrame:
    """Matching weather data for the same 10 days, rounded lat/lon."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "lat2": 18.50,
            "lon2": 73.80,
            "temperature_mean": 25.0,
            "precipitation_mm": 2.0,
            "humidity": 60.0,
            "wind_speed": 5.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_ndvi() -> pd.DataFrame:
    """NDVI values for M_A only (to test left-join behavior)."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    rows = [{"mandi_id": "M_A", "date": d, "ndvi": 0.3 + 0.01 * i}
            for i, d in enumerate(dates)]
    return pd.DataFrame(rows)


@pytest.fixture
def simple_cpi() -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2025] * 3,
        "month": [1, 2, 3],
        "food_cpi_index": [120.0, 121.0, 122.0],
    })


@pytest.fixture
def simple_wpi() -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2025] * 3,
        "month": [1, 2, 3],
        "food_wpi_index": [110.0, 111.0, 112.0],
    })


# ---------------------------------------------------------------------------
# Tests: Price Lags
# ---------------------------------------------------------------------------

class TestPriceLags:

    def test_lag_columns_created(self, simple_apmc):
        result = add_price_lags(simple_apmc)
        for lag in (1, 7, 14, 30):
            assert f"price_lag_{lag}d" in result.columns

    def test_lag_1d_shift(self, simple_apmc):
        df = simple_apmc[simple_apmc["mandi_id"] == "M_A"].copy()
        df = add_price_lags(df)
        # Row 0 should be NaN, row 1 should equal row 0's price
        assert pd.isna(df.iloc[0]["price_lag_1d"])
        assert df.iloc[1]["price_lag_1d"] == pytest.approx(df.iloc[0]["modal_price"])

    def test_lags_are_mandi_specific(self, simple_apmc):
        """Lags should not bleed across mandis."""
        df = add_price_lags(simple_apmc)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        mb = df[df["mandi_id"] == "M_B"].sort_values("date")
        # M_A's lag_1d at day 1 should be M_A's price at day 0
        assert ma.iloc[1]["price_lag_1d"] == pytest.approx(ma.iloc[0]["modal_price"])
        # M_B's lag_1d at day 1 should be M_B's price at day 0
        assert mb.iloc[1]["price_lag_1d"] == pytest.approx(mb.iloc[0]["modal_price"])


# ---------------------------------------------------------------------------
# Tests: Rolling Stats
# ---------------------------------------------------------------------------

class TestRollingStats:

    def test_rolling_columns_created(self, simple_apmc):
        result = add_rolling_stats(add_price_lags(simple_apmc))
        for w in (7, 14, 30):
            assert f"price_mean_{w}d" in result.columns
            assert f"price_std_{w}d" in result.columns

    def test_rolling_mean_is_shifted(self, simple_apmc):
        """Rolling mean should use shift(1) to avoid lookahead bias."""
        df = add_price_lags(simple_apmc)
        df = add_rolling_stats(df)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        # price_mean_7d at row 6 should be mean of rows 0..5
        expected = ma.iloc[:6]["modal_price"].mean()
        assert ma.iloc[6]["price_mean_7d"] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: Price Momentum
# ---------------------------------------------------------------------------

class TestPriceMomentum:

    def test_momentum_formula(self, simple_apmc):
        df = add_price_lags(simple_apmc)
        df = add_price_momentum(df)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        # momentum_7d = (price - lag_7d) / lag_7d
        row7 = ma.iloc[7]
        expected = (row7["modal_price"] - row7["price_lag_7d"]) / row7["price_lag_7d"]
        assert row7["price_momentum_7d"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Arrivals Features
# ---------------------------------------------------------------------------

class TestArrivalsFeatures:

    def test_arrivals_mean_7d_created(self, simple_apmc):
        df = add_arrivals_features(simple_apmc)
        assert "arrivals_mean_7d" in df.columns

    def test_arrivals_mean_is_shifted(self, simple_apmc):
        df = add_arrivals_features(simple_apmc)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        # arrivals_mean_7d at row 5 should be mean of rows 0..4
        expected = ma.iloc[:5]["arrivals_tonnes"].mean()
        assert ma.iloc[5]["arrivals_mean_7d"] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: Temporal Features
# ---------------------------------------------------------------------------

class TestTemporalFeatures:

    def test_temporal_columns(self, simple_apmc):
        result = add_temporal_features(simple_apmc)
        assert "day_of_week" in result.columns
        assert "day_of_month" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns

    def test_weekend_flag(self, simple_apmc):
        df = add_temporal_features(simple_apmc)
        # 2025-01-04 is Saturday (day_of_week=5)
        sat = df[df["date"] == pd.Timestamp("2025-01-04")]
        assert sat.iloc[0]["is_weekend"] == 1
        # 2025-01-06 is Monday (day_of_week=0)
        mon = df[df["date"] == pd.Timestamp("2025-01-06")]
        assert mon.iloc[0]["is_weekend"] == 0


# ---------------------------------------------------------------------------
# Tests: Targets
# ---------------------------------------------------------------------------

class TestTargets:

    def test_target_columns_created(self, simple_apmc):
        result = add_targets(simple_apmc)
        for h in (7, 15, 30):
            assert f"target_price_{h}d" in result.columns

    def test_target_7d_is_forward_price(self, simple_apmc):
        df = add_targets(simple_apmc)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        # target_price_7d at row 0 should be row 7's price
        assert ma.iloc[0]["target_price_7d"] == pytest.approx(ma.iloc[7]["modal_price"])

    def test_target_last_rows_are_nan(self, simple_apmc):
        df = add_targets(simple_apmc)
        ma = df[df["mandi_id"] == "M_A"].sort_values("date")
        # Last 7 rows should have NaN target_7d
        assert pd.isna(ma.iloc[-1]["target_price_7d"])
        assert pd.isna(ma.iloc[-2]["target_price_7d"])


# ---------------------------------------------------------------------------
# Tests: Weather Join
# ---------------------------------------------------------------------------

class TestWeatherJoin:

    def test_weather_merge_adds_columns(self, simple_apmc, simple_weather):
        result = join_weather(simple_apmc, simple_weather)
        assert "temperature_mean" in result.columns or "temp_mean_7d" in result.columns

    def test_weather_left_join(self, simple_apmc, simple_weather):
        """APMC rows should be preserved even if weather is missing."""
        result = join_weather(simple_apmc, simple_weather)
        assert len(result) >= len(simple_apmc)


# ---------------------------------------------------------------------------
# Tests: NDVI Join
# ---------------------------------------------------------------------------

class TestNdviJoin:

    def test_ndvi_columns_added(self, simple_apmc, simple_ndvi):
        result = join_ndvi(simple_apmc, simple_ndvi)
        assert "ndvi" in result.columns
        assert "ndvi_delta_30d" in result.columns

    def test_ndvi_only_for_matching_mandis(self, simple_apmc, simple_ndvi):
        """NDVI should only be non-null for M_A (the only mandi in simple_ndvi)."""
        result = join_ndvi(simple_apmc, simple_ndvi)
        mb_rows = result[result["mandi_id"] == "M_B"]
        # M_B has no NDVI data — should be NaN
        assert mb_rows["ndvi"].isna().all()


# ---------------------------------------------------------------------------
# Tests: Macro Join
# ---------------------------------------------------------------------------

class TestMacroJoin:

    def test_cpi_wpi_merged(self, simple_apmc, simple_cpi, simple_wpi):
        df = add_temporal_features(simple_apmc)
        result = join_macro(df, simple_cpi, simple_wpi)
        assert "food_cpi_index" in result.columns
        assert "food_wpi_index" in result.columns

    def test_macro_values_populated(self, simple_apmc, simple_cpi, simple_wpi):
        df = add_temporal_features(simple_apmc)
        result = join_macro(df, simple_cpi, simple_wpi)
        # join_macro adds _year/_month then drops year/month columns
        # Filter by day_of_month == 1 to pick January rows
        jan_rows = result[result["date"].dt.month == 1]
        assert jan_rows["food_cpi_index"].iloc[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Tests: Commodity Category Mapping
# ---------------------------------------------------------------------------

class TestCommodityCategory:

    def test_known_commodities(self):
        assert map_commodity_category("WHEAT") == "Cereal"
        assert map_commodity_category("RICE") == "Cereal"
        assert map_commodity_category("ONION") == "Vegetable"
        assert map_commodity_category("COTTON") == "Fiber"

    def test_unknown_commodity(self):
        assert map_commodity_category("RANDOM_STUFF") == "Other"

    def test_case_insensitive(self):
        assert map_commodity_category("wheat") == "Cereal"
        assert map_commodity_category(" Wheat ") == "Cereal"
