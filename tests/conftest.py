"""
AgriVault – tests/conftest.py
================================
Shared pytest fixtures used across multiple test files.

Fixtures
--------
    spark           – session-scoped PySpark SparkSession (local[2])
    simple_apmc     – 3 mandis × 1 commodity × 10 days of price data
    simple_weather  – matching weather data for the same 10 days
    simple_ndvi     – NDVI values for M_A only (left-join testing)
    simple_cpi      – monthly CPI index for Jan–Mar 2025
    simple_wpi      – monthly WPI index for Jan–Mar 2025
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# PySpark session (session-scoped — shared across all PySpark tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Create a local PySpark session for testing.

    Session-scoped so it's only created once per test run and reused
    across all tests that need it (e.g. test_ndvi_join.py).

    On Windows, sets PYSPARK_PYTHON so the PySpark worker subprocess
    can find the Python executable (the ``py`` launcher is not on PATH
    as ``python`` by default on some Windows configurations).
    """
    # Ensure PySpark workers can find the Python interpreter
    if "PYSPARK_PYTHON" not in os.environ:
        os.environ["PYSPARK_PYTHON"] = os.environ.get(
            "PYEXEC", os.path.normpath(os.sys.executable)
        )

    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .appName("agrivault-tests")
        .master("local[2]")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Synthetic pandas DataFrames — shared by test_price_features.py
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_apmc():
    """3 mandis × 1 commodity × 10 days of price data."""
    import pandas as pd

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
def simple_weather():
    """Matching weather data for the same 10 days, rounded lat/lon."""
    import pandas as pd

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
def simple_ndvi():
    """NDVI values for M_A only (to test left-join behavior)."""
    import pandas as pd

    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    rows = [{"mandi_id": "M_A", "date": d, "ndvi": 0.3 + 0.01 * i}
            for i, d in enumerate(dates)]
    return pd.DataFrame(rows)


@pytest.fixture
def simple_cpi():
    """Monthly food CPI index for Dec 2024–Mar 2025.

    Includes Dec 2024 so that January rows in simple_apmc can receive
    the lagged (anti-leakage) CPI value after the 1-month shift in
    join_macro.
    """
    import pandas as pd

    return pd.DataFrame({
        "year": [2024, 2025, 2025, 2025],
        "month": [12, 1, 2, 3],
        "food_cpi_index": [119.0, 120.0, 121.0, 122.0],
    })


@pytest.fixture
def simple_wpi():
    """Monthly food WPI index for Dec 2024–Mar 2025.

    Includes Dec 2024 so that January rows in simple_apmc can receive
    the lagged (anti-leakage) WPI value after the 1-month shift in
    join_macro.
    """
    import pandas as pd

    return pd.DataFrame({
        "year": [2024, 2025, 2025, 2025],
        "month": [12, 1, 2, 3],
        "food_wpi_index": [109.0, 110.0, 111.0, 112.0],
    })
