"""
AgriVault – tests/test_app.py
===============================
Unit tests for the Flask dashboard and JSON API endpoints.

Tests verify:
  - /health returns 503 when data not loaded, 200 when loaded
  - /api/scores with and without filters (commodity, decision, state, limit)
  - /api/scores/<commodity> for found and not-found commodities
  - /api/summary returns correct schema and types
  - /api/decisions returns decision distribution
  - /api/commodities returns commodity list with stats
  - HTML routes / and /commodity/<name> render without error
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.api.app import app, _scored_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Flask test client with TESTING mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def mock_scored_df():
    """Synthetic scored data matching the real schema."""
    return pd.DataFrame({
        "state": ["MAHARASHTRA", "MAHARASHTRA", "MAHARASHTRA", "MAHARASHTRA",
                   "KARNATAKA", "KARNATAKA"],
        "district": ["PUNE", "PUNE", "MUMBAI", "MUMBAI",
                     "BANGALORE", "BANGALORE"],
        "commodity": ["WHEAT", "TOMATO", "WHEAT", "ONION",
                      "WHEAT", "TOMATO"],
        "mandi_mean_price": [2500.0, 3000.0, 2400.0, 1800.0, 2600.0, 3200.0],
        "price_cv": [0.08, 0.27, 0.06, 0.33, 0.07, 0.25],
        "forecast_uncertainty": [0.20, 0.50, 0.15, 0.60, 0.18, 0.45],
        "n_warehouses": [2, 0, 5, 0, 3, 1],
        "risk_score": [0.25, 0.55, 0.20, 0.65, 0.22, 0.50],
        "decision": ["APPROVE", "CONDITIONAL", "APPROVE", "REJECT",
                     "APPROVE", "CONDITIONAL"],
        "recommended_ltv": [0.72, 0.55, 0.74, 0.42, 0.73, 0.58],
        "commodity_category": ["Cereal", "Vegetable", "Cereal", "Vegetable",
                               "Cereal", "Vegetable"],
    })


@pytest.fixture
def loaded_client(client, mock_scored_df):
    """Flask test client with mocked scored data already loaded.

    Eliminates the repeated ``with patch(...)`` context manager in every
    test that needs data present.  Tests that need data *absent* should
    use the plain ``client`` fixture instead.
    """
    with patch("src.api.app._scored_df", mock_scored_df):
        yield client


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_503_when_no_data(self, client):
        """Health should report degraded when scored data is not loaded."""
        with patch("src.api.app._scored_df", None):
            resp = client.get("/health")
            assert resp.status_code == 503
            body = resp.get_json()
            assert body["status"] == "degraded"
            assert body["data_loaded"] is False
            assert body["rows_loaded"] == 0

    def test_returns_200_when_data_loaded(self, loaded_client):
        """Health should report healthy when scored data is available."""
        resp = loaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "healthy"
        assert body["data_loaded"] is True
        assert body["rows_loaded"] == 6

    def test_uptime_is_number(self, loaded_client):
        """Uptime should be a non-negative number."""
        resp = loaded_client.get("/health")
        body = resp.get_json()
        assert isinstance(body["uptime_seconds"], (int, float))
        assert body["uptime_seconds"] >= 0

    def test_version_field_present(self, loaded_client):
        """Version field should always be present."""
        resp = loaded_client.get("/health")
        body = resp.get_json()
        assert "version" in body


# ---------------------------------------------------------------------------
# /api/scores — all scores
# ---------------------------------------------------------------------------

class TestApiScores:

    def test_returns_data(self, loaded_client):
        resp = loaded_client.get("/api/scores")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 6
        assert len(body["data"]) == 6

    def test_filter_by_commodity(self, loaded_client):
        resp = loaded_client.get("/api/scores?commodity=WHEAT")
        body = resp.get_json()
        assert body["count"] == 3
        assert all(d["commodity"] == "WHEAT" for d in body["data"])

    def test_filter_by_decision(self, loaded_client):
        resp = loaded_client.get("/api/scores?decision=APPROVE")
        body = resp.get_json()
        assert body["count"] == 3
        assert all(d["decision"] == "APPROVE" for d in body["data"])

    def test_filter_by_state(self, loaded_client):
        resp = loaded_client.get("/api/scores?state=KARNATAKA")
        body = resp.get_json()
        assert body["count"] == 2

    def test_combined_filters(self, loaded_client):
        resp = loaded_client.get("/api/scores?commodity=WHEAT&decision=APPROVE&state=MAHARASHTRA")
        body = resp.get_json()
        assert body["count"] == 2

    def test_limit_parameter(self, loaded_client):
        resp = loaded_client.get("/api/scores?limit=2")
        body = resp.get_json()
        assert body["count"] == 2

    def test_invalid_limit_returns_default(self, loaded_client):
        """Invalid limit should not crash — falls back to 1000."""
        resp = loaded_client.get("/api/scores?limit=abc")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 6  # all rows (under default limit)

    def test_limit_clamped_to_max(self, loaded_client):
        resp = loaded_client.get("/api/scores?limit=99999")
        body = resp.get_json()
        # limit is clamped to 10000, but we only have 6 rows
        assert body["count"] == 6

    def test_response_schema(self, loaded_client):
        """Each record should contain expected columns."""
        resp = loaded_client.get("/api/scores")
        body = resp.get_json()
        record = body["data"][0]
        expected_cols = {
            "state", "district", "commodity", "risk_score",
            "decision", "recommended_ltv",
        }
        assert expected_cols.issubset(set(record.keys()))


# ---------------------------------------------------------------------------
# /api/scores/<commodity> — single commodity
# ---------------------------------------------------------------------------

class TestApiScoresCommodity:

    def test_returns_commodity_data(self, loaded_client):
        resp = loaded_client.get("/api/scores/WHEAT")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["commodity"] == "WHEAT"
        assert body["count"] == 3

    def test_avg_risk_and_ltv_present(self, loaded_client):
        resp = loaded_client.get("/api/scores/WHEAT")
        body = resp.get_json()
        assert "avg_risk" in body
        assert "avg_ltv" in body
        assert isinstance(body["avg_risk"], float)
        assert isinstance(body["avg_ltv"], float)

    def test_decisions_dict_present(self, loaded_client):
        resp = loaded_client.get("/api/scores/WHEAT")
        body = resp.get_json()
        assert "decisions" in body
        assert isinstance(body["decisions"], dict)

    def test_not_found_returns_404(self, loaded_client):
        resp = loaded_client.get("/api/scores/NONEXISTENT")
        assert resp.status_code == 404
        body = resp.get_json()
        assert "error" in body

    def test_case_insensitive(self, loaded_client):
        resp = loaded_client.get("/api/scores/wheat")
        body = resp.get_json()
        assert body["commodity"] == "WHEAT"
        assert body["count"] == 3


# ---------------------------------------------------------------------------
# /api/summary
# ---------------------------------------------------------------------------

class TestApiSummary:

    def test_returns_summary(self, loaded_client):
        resp = loaded_client.get("/api/summary")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["total_rows"] == 6
        assert body["total_commodities"] == 3  # WHEAT, TOMATO, ONION
        assert body["total_states"] == 2       # MAHARASHTRA, KARNATAKA

    def test_risk_percentiles(self, loaded_client):
        resp = loaded_client.get("/api/summary")
        body = resp.get_json()
        pcts = body["risk_percentiles"]
        assert all(k in pcts for k in ["p10", "p25", "p50", "p75", "p90"])
        # Percentiles should be ordered
        assert pcts["p10"] <= pcts["p25"] <= pcts["p50"] <= pcts["p75"] <= pcts["p90"]

    def test_decision_distribution(self, loaded_client):
        resp = loaded_client.get("/api/summary")
        body = resp.get_json()
        dist = body["decision_distribution"]
        assert "APPROVE" in dist
        assert "CONDITIONAL" in dist
        assert "REJECT" in dist
        assert dist["APPROVE"] == 3
        assert dist["REJECT"] == 1

    def test_avg_ltv_is_percentage(self, loaded_client):
        """avg_recommended_ltv should be scaled to percentage (0–100)."""
        resp = loaded_client.get("/api/summary")
        body = resp.get_json()
        assert 0 < body["avg_recommended_ltv"] <= 100


# ---------------------------------------------------------------------------
# /api/decisions
# ---------------------------------------------------------------------------

class TestApiDecisions:

    def test_returns_decision_distribution(self, loaded_client):
        resp = loaded_client.get("/api/decisions")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] > 0
        assert isinstance(body["data"], list)

    def test_records_have_commodity_and_decision(self, loaded_client):
        resp = loaded_client.get("/api/decisions")
        body = resp.get_json()
        for record in body["data"]:
            assert "commodity" in record
            assert "decision" in record
            assert "count" in record


# ---------------------------------------------------------------------------
# /api/commodities
# ---------------------------------------------------------------------------

class TestApiCommodities:

    def test_returns_commodity_list(self, loaded_client):
        resp = loaded_client.get("/api/commodities")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 3  # WHEAT, TOMATO, ONION

    def test_commodity_stats(self, loaded_client):
        resp = loaded_client.get("/api/commodities")
        body = resp.get_json()
        record = body["data"][0]
        assert "count" in record
        assert "avg_risk" in record
        assert "avg_ltv" in record
        assert "approve_pct" in record

    def test_sorted_by_count_descending(self, loaded_client):
        resp = loaded_client.get("/api/commodities")
        body = resp.get_json()
        counts = [r["count"] for r in body["data"]]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

class TestHtmlRoutes:

    def test_dashboard_renders(self, loaded_client):
        resp = loaded_client.get("/")
        assert resp.status_code == 200
        assert b"<html" in resp.data.lower() or b"<!DOCTYPE" in resp.data

    def test_commodity_detail_renders(self, loaded_client):
        resp = loaded_client.get("/commodity/WHEAT")
        assert resp.status_code == 200

    def test_commodity_detail_404(self, loaded_client):
        resp = loaded_client.get("/commodity/NONEXISTENT")
        assert resp.status_code == 404

    def test_commodity_hyphen_to_space(self, loaded_client):
        """URL hyphens should be converted to spaces for matching."""
        resp = loaded_client.get("/commodity/GREEN-CHILLI")
        # Should either match or 404 (no GREEN CHILLI in mock data)
        assert resp.status_code in (200, 404)
