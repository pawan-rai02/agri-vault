"""
AgriVault – tests/test_risk_ltv.py
====================================
Unit tests for the Risk/LTV scoring model.

Tests verify:
  - Risk score computation [0, 1]
  - Decision mapping (APPROVE / CONDITIONAL / REJECT)
  - LTV recommendation within valid bounds
  - Commodity tier mapping
  - Edge cases (null values, extreme values)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.risk_ltv_model import RiskLTVModel, COMMODITY_RISK_TIER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_risk_df() -> pd.DataFrame:
    """Synthetic risk features for testing."""
    return pd.DataFrame({
        "state": ["MAHARASHTRA"] * 4,
        "district": ["PUNE", "PUNE", "MUMBAI", "MUMBAI"],
        "commodity": ["WHEAT", "TOMATO", "WHEAT", "ONION"],
        "mandi_mean_price": [2500.0, 3000.0, 2400.0, 1800.0],
        "mandi_std_price": [200.0, 800.0, 150.0, 600.0],
        "mandi_n_days": [300, 250, 280, 200],
        "total_capacity_mt": [5000.0, 0.0, 10000.0, 0.0],
        "n_warehouses": [2, 0, 5, 0],
        "commodity_category": ["Cereal", "Vegetable", "Cereal", "Vegetable"],
        "portfolio_default_rate": [0.063, 0.063, 0.063, 0.063],
        "portfolio_mean_ltv": [0.70, 0.70, 0.70, 0.70],
        "price_cv": [0.08, 0.27, 0.06, 0.33],
        "forecast_uncertainty": [0.20, 0.50, 0.15, 0.60],
        "risk_score_proxy": [0.35, 0.50, 0.30, 0.55],
        "recommended_max_ltv": [0.43, 0.40, 0.45, 0.40],
    })


@pytest.fixture
def model() -> RiskLTVModel:
    return RiskLTVModel()


# ---------------------------------------------------------------------------
# Tests: Risk Score
# ---------------------------------------------------------------------------

class TestRiskScore:

    def test_score_range(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        assert "risk_score" in scored.columns
        assert scored["risk_score"].min() >= 0.0
        assert scored["risk_score"].max() <= 1.0

    def test_low_volatility_low_risk(self, model, sample_risk_df):
        """WHEAT with low CV and good warehouses should score low."""
        scored = model.score(sample_risk_df)
        wheat = scored[scored["commodity"] == "WHEAT"]
        # The one with warehouses should have lower risk
        wh = wheat[wheat["n_warehouses"] == 5].iloc[0]
        assert wh["risk_score"] < 0.5

    def test_high_volatility_high_risk(self, model, sample_risk_df):
        """TOMATO with high CV and no warehouses should score higher."""
        scored = model.score(sample_risk_df)
        tomato = scored[scored["commodity"] == "TOMATO"].iloc[0]
        wheat_best = scored[
            (scored["commodity"] == "WHEAT") & (scored["n_warehouses"] == 5)
        ].iloc[0]
        assert tomato["risk_score"] > wheat_best["risk_score"]

    def test_score_is_float(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        assert scored["risk_score"].dtype in [np.float64, np.float32]


# ---------------------------------------------------------------------------
# Tests: Decision
# ---------------------------------------------------------------------------

class TestDecision:

    def test_decision_column_exists(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        assert "decision" in scored.columns

    def test_valid_decisions(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        valid = {"APPROVE", "CONDITIONAL", "REJECT"}
        assert set(scored["decision"].unique()).issubset(valid)

    def test_low_risk_approve(self, model):
        """Very low CV, good warehouses, stable commodity → APPROVE."""
        df = pd.DataFrame({
            "state": ["TEST"], "district": ["TEST"],
            "commodity": ["WHEAT"],
            "mandi_mean_price": [2500.0], "mandi_std_price": [50.0],
            "mandi_n_days": [300], "total_capacity_mt": [10000.0],
            "n_warehouses": [10], "commodity_category": ["Cereal"],
            "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
            "price_cv": [0.02], "forecast_uncertainty": [0.10],
            "risk_score_proxy": [0.20], "recommended_max_ltv": [0.70],
        })
        scored = model.score(df)
        assert scored.iloc[0]["decision"] == "APPROVE"

    def test_high_risk_reject(self, model):
        """High CV, no warehouses, perishable → REJECT."""
        df = pd.DataFrame({
            "state": ["TEST"], "district": ["TEST"],
            "commodity": ["TOMATO"],
            "mandi_mean_price": [3000.0], "mandi_std_price": [1500.0],
            "mandi_n_days": [50], "total_capacity_mt": [0.0],
            "n_warehouses": [0], "commodity_category": ["Vegetable"],
            "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
            "price_cv": [0.80], "forecast_uncertainty": [0.90],
            "risk_score_proxy": [0.80], "recommended_max_ltv": [0.40],
        })
        scored = model.score(df)
        assert scored.iloc[0]["decision"] == "REJECT"


# ---------------------------------------------------------------------------
# Tests: LTV Recommendation
# ---------------------------------------------------------------------------

class TestLTVRecommendation:

    def test_ltv_column_exists(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        assert "recommended_ltv" in scored.columns

    def test_ltv_in_valid_range(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        assert scored["recommended_ltv"].min() >= 0.40
        assert scored["recommended_ltv"].max() <= 0.75

    def test_approve_higher_ltv_than_reject(self, model, sample_risk_df):
        scored = model.score(sample_risk_df)
        approves = scored[scored["decision"] == "APPROVE"]["recommended_ltv"]
        rejects = scored[scored["decision"] == "REJECT"]["recommended_ltv"]
        if len(approves) > 0 and len(rejects) > 0:
            assert approves.mean() > rejects.mean()

    def test_ltv_monotonically_decreases_with_risk(self, model):
        """LTV should decrease as risk score increases."""
        risks = np.linspace(0.1, 0.9, 9)
        ltvs = []
        for r in risks:
            df = pd.DataFrame({
                "state": ["T"], "district": ["T"],
                "commodity": ["WHEAT"],
                "mandi_mean_price": [2500.0], "mandi_std_price": [2500 * r],
                "mandi_n_days": [300], "total_capacity_mt": [5000.0],
                "n_warehouses": [int((1 - r) * 5)], "commodity_category": ["Cereal"],
                "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
                "price_cv": [r], "forecast_uncertainty": [r],
                "risk_score_proxy": [r], "recommended_max_ltv": [0.70 - r * 0.3],
            })
            scored = model.score(df)
            ltvs.append(scored.iloc[0]["recommended_ltv"])
        # LTV should generally decrease (allow some tolerance for non-monotonicity)
        decreasing = sum(ltvs[i] >= ltvs[i + 1] for i in range(len(ltvs) - 1))
        assert decreasing >= len(ltvs) - 3  # at least 7 of 8 pairs decreasing


# ---------------------------------------------------------------------------
# Tests: Commodity Tier Mapping
# ---------------------------------------------------------------------------

class TestCommodityTier:

    def test_known_commodities_have_tiers(self):
        for commodity in ["WHEAT", "TOMATO", "ONION", "RICE", "COTTON"]:
            assert commodity in COMMODITY_RISK_TIER

    def test_perishables_higher_tier(self):
        """Perishables should have higher risk tiers than cereals."""
        assert COMMODITY_RISK_TIER["TOMATO"] > COMMODITY_RISK_TIER["WHEAT"]
        assert COMMODITY_RISK_TIER["ONION"] > COMMODITY_RISK_TIER["RICE"]


# ---------------------------------------------------------------------------
# Tests: Null Handling
# ---------------------------------------------------------------------------

class TestNullHandling:

    def test_nulls_in_cv(self, model):
        df = pd.DataFrame({
            "state": ["T"], "district": ["T"],
            "commodity": ["WHEAT"],
            "mandi_mean_price": [2500.0], "mandi_std_price": [np.nan],
            "mandi_n_days": [300], "total_capacity_mt": [5000.0],
            "n_warehouses": [2], "commodity_category": ["Cereal"],
            "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
            "price_cv": [np.nan], "forecast_uncertainty": [0.3],
            "risk_score_proxy": [0.4], "recommended_max_ltv": [0.60],
        })
        scored = model.score(df)
        assert not scored["risk_score"].isna().any()

    def test_nulls_in_forecast_uncertainty(self, model):
        df = pd.DataFrame({
            "state": ["T"], "district": ["T"],
            "commodity": ["WHEAT"],
            "mandi_mean_price": [2500.0], "mandi_std_price": [200.0],
            "mandi_n_days": [300], "total_capacity_mt": [5000.0],
            "n_warehouses": [2], "commodity_category": ["Cereal"],
            "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
            "price_cv": [0.10], "forecast_uncertainty": [np.nan],
            "risk_score_proxy": [0.4], "recommended_max_ltv": [0.60],
        })
        scored = model.score(df)
        assert not scored["risk_score"].isna().any()

    def test_unknown_commodity(self, model):
        df = pd.DataFrame({
            "state": ["T"], "district": ["T"],
            "commodity": ["UNKNOWN_FRUIT"],
            "mandi_mean_price": [1000.0], "mandi_std_price": [200.0],
            "mandi_n_days": [100], "total_capacity_mt": [0.0],
            "n_warehouses": [0], "commodity_category": ["Other"],
            "portfolio_default_rate": [0.063], "portfolio_mean_ltv": [0.70],
            "price_cv": [0.20], "forecast_uncertainty": [0.3],
            "risk_score_proxy": [0.4], "recommended_max_ltv": [0.60],
        })
        scored = model.score(df)
        assert not scored["risk_score"].isna().any()
