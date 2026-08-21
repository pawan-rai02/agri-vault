"""
AgriVault – tests/test_evaluation.py
======================================
Unit tests for the evaluation report module.

Tests verify:
  - Naive persistence baseline produces correct forecasts
  - Pinball loss computation is correct
  - Coverage calculation is correct
  - Risk decision evaluation produces valid metrics
  - Lift calculation makes sense
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import (
    compute_metrics,
    evaluate_risk_decisions,
    naive_persistence_forecast,
)
from src.models.quantile_gbm.loss import pinball_loss


# ---------------------------------------------------------------------------
# Tests: Naive Persistence Baseline
# ---------------------------------------------------------------------------

class TestNaivePersistence:

    def test_forecast_uses_last_train_price(self):
        y_train = np.array([100.0, 102.0, 105.0, 103.0, 108.0])
        y_test = np.array([110.0, 112.0, 109.0])
        preds = naive_persistence_forecast(y_train, y_test, horizon=7)
        assert all(p == pytest.approx(108.0) for p in preds[0.50])

    def test_all_quantiles_same_value(self):
        y_train = np.array([50.0, 55.0, 60.0])
        y_test = np.array([62.0, 58.0])
        preds = naive_persistence_forecast(y_train, y_test, horizon=7)
        # Naive baseline predicts the same value for all quantiles
        assert np.array_equal(preds[0.10], preds[0.50])
        assert np.array_equal(preds[0.50], preds[0.90])

    def test_output_shape(self):
        y_train = np.array([100.0] * 100)
        y_test = np.array([101.0] * 50)
        preds = naive_persistence_forecast(y_train, y_test, horizon=7)
        assert len(preds[0.50]) == 50

    def test_empty_train_uses_nan(self):
        y_train = np.array([])
        y_test = np.array([100.0, 200.0])
        preds = naive_persistence_forecast(y_train, y_test, horizon=7)
        assert np.all(np.isnan(preds[0.50]))


# ---------------------------------------------------------------------------
# Tests: Compute Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_perfect_prediction_zero_loss(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = {0.10: y_true.copy(), 0.50: y_true.copy(), 0.90: y_true.copy()}
        metrics = compute_metrics(y_true, y_pred, label="test")
        assert metrics["pinball_q50"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)

    def test_imperfect_prediction_positive_loss(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = {0.50: np.array([110.0, 190.0, 310.0])}
        metrics = compute_metrics(y_true, y_pred, label="test")
        assert metrics["pinball_q50"] > 0
        assert metrics["rmse"] > 0

    def test_coverage_calculation(self):
        y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        y_pred = {
            0.10: np.array([90.0] * 5),
            0.50: np.array([120.0] * 5),
            0.90: np.array([150.0] * 5),
        }
        metrics = compute_metrics(y_true, y_pred, label="test")
        # All values are within [90, 150] → 100% coverage
        assert metrics["coverage_80pct"] == pytest.approx(100.0)

    def test_partial_coverage(self):
        y_true = np.array([80.0, 100.0, 120.0, 140.0, 160.0])
        y_pred = {
            0.10: np.array([90.0] * 5),
            0.90: np.array([130.0] * 5),
        }
        metrics = compute_metrics(y_true, y_pred, label="test")
        # 80 < 90 (outside), 100 & 120 in [90,130], 140 > 130, 160 > 130 → 40% coverage
        assert metrics["coverage_80pct"] == pytest.approx(40.0)

    def test_mape_calculation(self):
        y_true = np.array([100.0, 200.0])
        y_pred = {0.50: np.array([110.0, 180.0])}
        metrics = compute_metrics(y_true, y_pred, label="test")
        # MAPE = mean(|error|/|true|) * 100
        # = (10/100 + 20/200) / 2 * 100 = (0.1 + 0.1) / 2 * 100 = 10%
        assert metrics["mape"] == pytest.approx(10.0)

    def test_label_appears_in_output(self):
        metrics = compute_metrics(np.array([1.0]), {0.50: np.array([1.0])}, label="my_method")
        assert metrics["method"] == "my_method"


# ---------------------------------------------------------------------------
# Tests: Risk Decision Evaluation
# ---------------------------------------------------------------------------

class TestRiskDecisionEvaluation:

    @pytest.fixture
    def sample_risk_df(self):
        """Create a sample risk feature DataFrame for testing."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "commodity": ["WHEAT"] * n,
            "price_cv": np.random.uniform(0.05, 0.5, n),
            "forecast_uncertainty": np.random.uniform(0.1, 0.8, n),
            "n_warehouses": np.random.randint(0, 10, n),
            "mandi_mean_price": np.random.uniform(1000, 5000, n),
            "mandi_std_price": np.random.uniform(100, 500, n),
            "total_capacity_mt": np.random.uniform(100, 1000, n),
            "commodity_category": ["Cereal"] * n,
            "portfolio_default_rate": [0.06] * n,
            "portfolio_mean_ltv": [0.65] * n,
            "season": np.random.choice(["Kharif", "Rabi", "Zaid"], n),
        })

    def test_returns_valid_metrics(self, sample_risk_df):
        result = evaluate_risk_decisions(sample_risk_df)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "accuracy" in result

    def test_metrics_in_valid_range(self, sample_risk_df):
        result = evaluate_risk_decisions(sample_risk_df)
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
        assert 0 <= result["accuracy"] <= 1

    def test_confusion_matrix_consistency(self, sample_risk_df):
        result = evaluate_risk_decisions(sample_risk_df)
        total = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        assert total == result["n_samples"]

    def test_decision_distribution_present(self, sample_risk_df):
        result = evaluate_risk_decisions(sample_risk_df)
        dist = result["decision_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == result["n_samples"]

    def test_avg_risk_in_range(self, sample_risk_df):
        result = evaluate_risk_decisions(sample_risk_df)
        assert 0 <= result["avg_risk_score"] <= 1
        assert 0.40 <= result["avg_recommended_ltv"] <= 0.75


# ---------------------------------------------------------------------------
# Tests: Lift Calculation Logic
# ---------------------------------------------------------------------------

class TestLiftLogic:

    def test_model_better_than_naive_positive_lift(self):
        """When model outperforms naive, lift should be positive."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        # Model: close to truth
        model_preds = {
            0.10: np.array([95.0, 195.0, 295.0, 395.0, 495.0]),
            0.50: np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            0.90: np.array([105.0, 205.0, 305.0, 405.0, 505.0]),
        }
        model_metrics = compute_metrics(y_true, model_preds, "model")

        # Naive: constant forecast far from truth
        naive_preds = {
            0.10: np.array([50.0] * 5),
            0.50: np.array([50.0] * 5),
            0.90: np.array([50.0] * 5),
        }
        naive_metrics = compute_metrics(y_true, naive_preds, "naive")

        # Lift = (naive - model) / naive * 100 (lower loss = better)
        rmse_lift = (
            (naive_metrics["rmse"] - model_metrics["rmse"])
            / naive_metrics["rmse"] * 100
        )
        assert rmse_lift > 0  # Model should have positive lift

    def test_same_performance_zero_lift(self):
        """When model equals naive, lift should be ~0."""
        y_true = np.array([100.0, 200.0, 300.0])
        preds = {
            0.10: np.array([100.0, 200.0, 300.0]),
            0.50: np.array([100.0, 200.0, 300.0]),
            0.90: np.array([100.0, 200.0, 300.0]),
        }
        m1 = compute_metrics(y_true, preds, "a")
        m2 = compute_metrics(y_true, preds, "b")
        assert m1["pinball_q50"] == pytest.approx(m2["pinball_q50"])
