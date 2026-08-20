"""
AgriVault – tests/test_weather_forecast.py
===========================================
Unit tests for the Open-Meteo forward weather forecast client.

Tests verify:
  - Feature extraction from a mocked Open-Meteo response
  - Handling of missing/empty data
  - Graceful failure when API is unreachable
"""

from __future__ import annotations

import pytest

from src.serving.weather_forecast import build_forecast_features, fetch_forecast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_forecast_response() -> dict:
    """Realistic Open-Meteo API response for Lucknow (16 days)."""
    return {
        "latitude": 26.85,
        "longitude": 80.95,
        "timezone": "Asia/Kolkata",
        "daily": {
            "time": [
                "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
                "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08",
                "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12",
                "2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16",
            ],
            "temperature_2m_max": [
                34.0, 35.0, 33.5, 32.0, 31.0, 30.5, 31.5,
                33.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.5, 30.0, 31.0,
            ],
            "temperature_2m_min": [
                26.0, 27.0, 25.5, 24.0, 23.0, 22.5, 23.5,
                25.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.5, 22.0, 23.0,
            ],
            "precipitation_sum": [
                5.0, 0.0, 12.0, 8.0, 0.0, 3.0, 15.0,
                2.0, 0.0, 7.0, 10.0, 0.0, 4.0, 8.0, 1.0, 0.0,
            ],
            "relative_humidity_2m_max": [
                95.0, 90.0, 98.0, 92.0, 88.0, 85.0, 97.0,
                90.0, 87.0, 93.0, 96.0, 85.0, 88.0, 91.0, 86.0, 84.0,
            ],
        },
    }


@pytest.fixture
def minimal_forecast_response() -> dict:
    """Response with only 3 days of data."""
    return {
        "latitude": 26.85,
        "longitude": 80.95,
        "daily": {
            "time": ["2026-08-01", "2026-08-02", "2026-08-03"],
            "temperature_2m_max": [34.0, 35.0, 33.0],
            "temperature_2m_min": [26.0, 27.0, 25.0],
            "precipitation_sum": [5.0, 0.0, 12.0],
            "relative_humidity_2m_max": [95.0, 90.0, 98.0],
        },
    }


# ---------------------------------------------------------------------------
# Tests: build_forecast_features
# ---------------------------------------------------------------------------

class TestBuildForecastFeatures:

    def test_returns_all_keys(self, mock_forecast_response):
        features = build_forecast_features(mock_forecast_response)
        expected_keys = {
            "forecast_temp_7d_avg",
            "forecast_precip_7d_sum",
            "forecast_precip_14d_sum",
            "forecast_humidity_7d_avg",
        }
        assert set(features.keys()) == expected_keys

    def test_temp_7d_avg_computed(self, mock_forecast_response):
        features = build_forecast_features(mock_forecast_response)
        temp = features["forecast_temp_7d_avg"]
        # First 7 days: avg of (max+min)/2
        # Day1: (34+26)/2=30, Day2: (35+27)/2=31, Day3: (33.5+25.5)/2=29.5,
        # Day4: (32+24)/2=28, Day5: (31+23)/2=27, Day6: (30.5+22.5)/2=26.5,
        # Day7: (31.5+23.5)/2=27.5
        # Mean = (30+31+29.5+28+27+26.5+27.5)/7 = 200.5/7 ≈ 28.64
        assert temp is not None
        assert 28.0 < temp < 30.0

    def test_precip_7d_sum(self, mock_forecast_response):
        features = build_forecast_features(mock_forecast_response)
        # First 7 days: 5+0+12+8+0+3+15 = 43
        assert features["forecast_precip_7d_sum"] == 43.0

    def test_precip_14d_sum(self, mock_forecast_response):
        features = build_forecast_features(mock_forecast_response)
        # First 14 days: 43 + 2+0+7+10+0+4+8 = 74
        assert features["forecast_precip_14d_sum"] == 74.0

    def test_humidity_7d_avg(self, mock_forecast_response):
        features = build_forecast_features(mock_forecast_response)
        # First 7 days: (95+90+98+92+88+85+97)/7 = 645/7 ≈ 92.14
        assert features["forecast_humidity_7d_avg"] is not None
        assert 92.0 < features["forecast_humidity_7d_avg"] < 93.0

    def test_minimal_data(self, minimal_forecast_response):
        features = build_forecast_features(minimal_forecast_response)
        # Only 3 days — 14d sum should still work
        assert features["forecast_precip_14d_sum"] == 17.0
        assert features["forecast_temp_7d_avg"] is not None

    def test_empty_response(self):
        features = build_forecast_features({"daily": {}})
        assert features["forecast_temp_7d_avg"] is None
        assert features["forecast_precip_7d_sum"] is None

    def test_none_values_in_daily(self):
        response = {
            "daily": {
                "time": ["2026-08-01"],
                "temperature_2m_max": [None],
                "temperature_2m_min": [None],
                "precipitation_sum": [None],
                "relative_humidity_2m_max": [None],
            }
        }
        features = build_forecast_features(response)
        assert features["forecast_temp_7d_avg"] is None
        assert features["forecast_precip_7d_sum"] is None


# ---------------------------------------------------------------------------
# Tests: fetch_forecast (network)
# ---------------------------------------------------------------------------

class TestFetchForecast:

    def test_fetch_returns_dict_or_none(self):
        """Integration test — should return a dict or None (no crash)."""
        result = fetch_forecast(26.8467, 80.9462, forecast_days=3)
        # Could be None if no network, that's fine
        assert result is None or isinstance(result, dict)

    def test_fetch_bad_coords(self):
        """Invalid coordinates should not crash."""
        result = fetch_forecast(999.0, 999.0, forecast_days=1)
        assert result is None or isinstance(result, dict)
