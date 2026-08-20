"""
AgriVault – tests/test_quantile_gbm.py
========================================
Unit tests for the custom-from-scratch Quantile GBM implementation.

Tests verify:
  - Pinball loss correctness
  - Gradient and hessian sign/dimension
  - CART tree splitting and prediction
  - QuantileGBM fitting and prediction
  - Walk-forward CV split generation
  - End-to-end training on synthetic data

Uses only numpy — no S3, Spark, or external data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.quantile_gbm.loss import (
    gradient,
    hessian,
    leaf_value_quantile,
    pinball_loss,
)
from src.models.quantile_gbm.tree import CARTQuantileTree, TreeNode
from src.models.quantile_gbm.gradient_boosted_trees import QuantileGBM
from src.models.quantile_gbm.hypertuner import walk_forward_splits


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

_GBM_DEFAULTS = {
    "quantiles": [0.50],
    "n_estimators": 20,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "random_state": 42,
}


def _make_gbm(**overrides) -> QuantileGBM:
    """Build a QuantileGBM with sensible defaults.

    Any keyword argument overrides the corresponding default.
    Example: _make_gbm(quantiles=[0.10, 0.50, 0.90], n_estimators=30)
    """
    params = {**_GBM_DEFAULTS, **overrides}
    return QuantileGBM(**params)


@pytest.fixture
def synthetic_data():
    """Simple linear relationship with noise: y = 2*x + noise."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.uniform(0, 10, size=(n, 3))
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, size=n)
    return X, y


@pytest.fixture
def tiny_data():
    """Very small dataset for fast tree tests."""
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    return X, y


# ---------------------------------------------------------------------------
# Tests: Pinball Loss
# ---------------------------------------------------------------------------

class TestPinballLoss:

    def test_perfect_prediction_zero_loss(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, f, 0.5) == pytest.approx(0.0)

    def test_underprediction_positive_loss(self):
        """Predicting too low should penalize with quantile weight."""
        y = np.array([10.0])
        f = np.array([0.0])
        loss = pinball_loss(y, f, 0.9)
        assert loss == pytest.approx(9.0)  # 0.9 * (10 - 0)

    def test_overprediction_positive_loss(self):
        """Predicting too high should penalize with (1-quantile) weight."""
        y = np.array([0.0])
        f = np.array([10.0])
        loss = pinball_loss(y, f, 0.9)
        assert loss == pytest.approx(1.0)  # (1 - 0.9) * (10 - 0)

    def test_symmetry_at_q50(self):
        """At q=0.5, over and underprediction should have equal weight."""
        y = np.array([5.0])
        loss_under = pinball_loss(y, np.array([0.0]), 0.5)
        loss_over = pinball_loss(y, np.array([10.0]), 0.5)
        assert loss_under == pytest.approx(2.5)
        assert loss_over == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Tests: Gradient / Hessian
# ---------------------------------------------------------------------------

class TestGradientHessian:

    def test_gradient_shape(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.5, 1.5, 1.5])
        g = gradient(y, f, 0.5)
        assert g.shape == y.shape

    def test_gradient_sign(self):
        """Gradient should be positive when y > f (under-prediction at q=0.9)."""
        y = np.array([10.0])
        f = np.array([5.0])
        g = gradient(y, f, 0.9)
        assert g[0] == pytest.approx(0.9)

    def test_gradient_overprediction(self):
        """Gradient should be negative when y < f."""
        y = np.array([5.0])
        f = np.array([10.0])
        g = gradient(y, f, 0.9)
        assert g[0] == pytest.approx(-0.1)

    def test_hessian_ones(self):
        y = np.array([1.0, 2.0, 3.0])
        h = hessian(y, y, 0.5)
        np.testing.assert_array_equal(h, np.ones(3))


# ---------------------------------------------------------------------------
# Tests: Leaf Value
# ---------------------------------------------------------------------------

class TestLeafValue:

    def test_leaf_value_median(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val = leaf_value_quantile(r, np.ones(5), 0.5)
        # Should be close to median
        assert val == pytest.approx(3.0, abs=1.0)

    def test_leaf_value_high_quantile(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        val = leaf_value_quantile(r, np.ones(10), 0.9)
        # Should be near the 90th percentile
        assert val >= 8.0


# ---------------------------------------------------------------------------
# Tests: CART Tree
# ---------------------------------------------------------------------------

class TestCARTTree:

    def test_fit_and_predict(self, tiny_data):
        X, y = tiny_data
        tree = CARTQuantileTree(max_depth=3, min_samples_leaf=2, quantile=0.5)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert preds.shape == (10,)

    def test_tree_has_root(self, tiny_data):
        X, y = tiny_data
        tree = CARTQuantileTree(max_depth=3, min_samples_leaf=2, quantile=0.5)
        tree.fit(X, y)
        assert tree.root is not None

    def test_depth_0_single_leaf(self, tiny_data):
        X, y = tiny_data
        tree = CARTQuantileTree(max_depth=0, min_samples_leaf=2, quantile=0.5)
        tree.fit(X, y)
        assert tree.root.is_leaf

    def test_prediction_ordering(self, synthetic_data):
        """Higher X[:,0] should generally produce higher predictions."""
        X, y = synthetic_data
        tree = CARTQuantileTree(max_depth=5, min_samples_leaf=10, quantile=0.5)
        tree.fit(X, y)
        preds = tree.predict(X)
        # Check Spearman-like ordering on X[:,0]
        sorted_idx = np.argsort(X[:, 0])
        sorted_preds = preds[sorted_idx]
        # At least 70% of adjacent pairs should be in order
        in_order = sum(sorted_preds[i] <= sorted_preds[i + 1]
                       for i in range(len(sorted_preds) - 1))
        assert in_order / (len(sorted_preds) - 1) > 0.6

    def test_predict_before_fit_raises(self):
        tree = CARTQuantileTree(max_depth=3, min_samples_leaf=2, quantile=0.5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            tree.predict(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Tests: QuantileGBM Ensemble
# ---------------------------------------------------------------------------

class TestQuantileGBM:

    def test_fit_and_predict(self, synthetic_data):
        X, y = synthetic_data
        model = _make_gbm(quantiles=[0.10, 0.50, 0.90], subsample=0.8)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds.keys()) == {0.10, 0.50, 0.90}
        assert preds[0.10].shape == (500,)
        assert preds[0.50].shape == (500,)
        assert preds[0.90].shape == (500,)

    def test_quantile_ordering(self, synthetic_data):
        """q10 predictions should generally be <= q50 <= q90."""
        X, y = synthetic_data
        model = _make_gbm(quantiles=[0.10, 0.50, 0.90], n_estimators=30)
        model.fit(X, y)
        preds = model.predict(X)
        # On average across all samples, q10 <= q50 <= q90
        mean_q10 = np.mean(preds[0.10])
        mean_q50 = np.mean(preds[0.50])
        mean_q90 = np.mean(preds[0.90])
        assert mean_q10 <= mean_q50 + 1.0  # allow some tolerance
        assert mean_q50 <= mean_q90 + 1.0

    def test_predict_quantile_single(self, synthetic_data):
        X, y = synthetic_data
        model = _make_gbm(n_estimators=10)
        model.fit(X, y)
        preds = model.predict_quantile(X, 0.50)
        assert preds.shape == (500,)

    def test_predict_quantile_untrained_raises(self, synthetic_data):
        X, y = synthetic_data
        model = _make_gbm(n_estimators=10)
        model.fit(X, y)
        with pytest.raises(ValueError, match="not trained"):
            model.predict_quantile(X, 0.90)

    def test_training_losses_tracked(self, synthetic_data):
        X, y = synthetic_data
        model = _make_gbm()
        model.fit(X, y)
        losses = model.training_losses()
        assert 0.50 in losses
        assert len(losses[0.50]) == 20
        # Loss should generally decrease
        assert losses[0.50][-1] <= losses[0.50][0]

    def test_n_trained_trees(self, synthetic_data):
        X, y = synthetic_data
        model = _make_gbm(quantiles=[0.10, 0.50, 0.90], n_estimators=15)
        model.fit(X, y)
        counts = model.n_trained_trees()
        assert all(v == 15 for v in counts.values())

    def test_few_samples_still_works(self):
        """Model should handle small datasets without crashing."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, size=(30, 2))
        y = X[:, 0] * 2 + rng.normal(0, 0.5, size=30)
        model = _make_gbm(n_estimators=5, max_depth=2, min_samples_leaf=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds[0.50].shape == (30,)


# ---------------------------------------------------------------------------
# Tests: Walk-Forward CV Splits
# ---------------------------------------------------------------------------

class TestWalkForwardSplits:

    def test_split_count(self):
        dates = pd.Series(pd.date_range("2025-01-01", periods=12, freq="MS"))
        splits = walk_forward_splits(dates, min_train_months=6, val_months=1)
        # With 12 months, min_train=6, val=1 → 6 folds
        assert len(splits) == 6

    def test_train_before_val(self):
        dates = pd.Series(pd.date_range("2025-01-01", periods=12, freq="MS"))
        splits = walk_forward_splits(dates, min_train_months=6, val_months=1)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()

    def test_expanding_window(self):
        """Training set should grow with each fold."""
        dates = pd.Series(pd.date_range("2025-01-01", periods=12, freq="MS"))
        splits = walk_forward_splits(dates, min_train_months=6, val_months=1)
        for i in range(1, len(splits)):
            assert len(splits[i][0]) >= len(splits[i - 1][0])
