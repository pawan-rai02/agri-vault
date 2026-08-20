"""
AgriVault – Model Registry
============================
Loads all trained Quantile GBM models from S3 into memory once at startup,
so they can be served in milliseconds per request.

Each model file (``models/qgbm_{commodity}_{horizon}d.pkl``) contains a
``QuantileGBM`` instance that handles all quantiles (q10, q50, q90) for
that commodity + horizon combination.

Usage
-----
    from src.serving.model_registry import load_all_models, get_model

    # At startup:
    load_all_models(commodities=["WHEAT", "TOMATO", "ONION"])

    # In request handler:
    model = get_model("WHEAT", horizon=7)
    if model is not None:
        preds = model.predict(X)  # returns {0.10: ..., 0.50: ..., 0.90: ...}

Model naming convention (set by train.py):
    s3://agrivault-lake-pawan/models/qgbm_{commodity}_{horizon}d.pkl
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Tuple

from src.models.quantile_gbm.gradient_boosted_trees import QuantileGBM
from src.storage.s3_client import S3Client

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COMMODITIES = ["WHEAT", "TOMATO", "ONION", "POTATO", "SOYABEAN"]
DEFAULT_HORIZONS = [7, 15, 30]

# ---------------------------------------------------------------------------
# In-memory model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[Tuple[str, int], QuantileGBM] = {}


def _model_key(commodity: str, horizon: int) -> Tuple[str, int]:
    """Normalise and return the cache key."""
    return (commodity.upper(), int(horizon))


def _model_s3_key(commodity: str, horizon: int) -> str:
    """Build the S3 key for a model pickle."""
    safe_name = commodity.upper().replace(" ", "_").replace("/", "-")
    return f"models/qgbm_{safe_name}_{horizon}d.pkl"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all_models(
    commodities: List[str] | None = None,
    horizons: List[int] | None = None,
    s3: S3Client | None = None,
) -> int:
    """Load all trained models from S3 into the in-memory cache.

    Parameters
    ----------
    commodities : list of str, optional
        Commodities to load. Defaults to the 5 trained commodities.
    horizons : list of int, optional
        Forecast horizons (days). Defaults to [7, 15, 30].
    s3 : S3Client, optional
        Pre-initialised S3 client. Created if None.

    Returns
    -------
    int
        Number of models successfully loaded.
    """
    global _MODEL_CACHE

    if commodities is None:
        commodities = DEFAULT_COMMODITIES
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if s3 is None:
        s3 = S3Client()

    loaded = 0
    missing = 0

    for commodity in commodities:
        for horizon in horizons:
            key = _model_s3_key(commodity, horizon)
            cache_key = _model_key(commodity, horizon)

            try:
                raw = s3.read_bytes(key)
                model: QuantileGBM = pickle.loads(raw)
                _MODEL_CACHE[cache_key] = model
                loaded += 1
                log.debug(
                    "Loaded model: %s @%dd  (quantiles=%s)",
                    commodity, horizon, model.quantiles,
                )
            except FileNotFoundError:
                missing += 1
                log.debug("Model not found: s3://%s/%s", s3.bucket, key)
            except Exception as exc:
                log.warning(
                    "Failed to load model %s @%dd: %s", commodity, horizon, exc
                )
                missing += 1

    log.info(
        "Model registry: %d loaded, %d missing/unavailable  "
        "(%d commodities × %d horizons = %d attempted)",
        loaded, missing, len(commodities), len(horizons),
        len(commodities) * len(horizons),
    )
    return loaded


def get_model(commodity: str, horizon: int) -> QuantileGBM | None:
    """Retrieve a loaded model by commodity and horizon.

    Returns None if the model is not in the cache (not trained or failed
    to load).
    """
    return _MODEL_CACHE.get(_model_key(commodity, horizon))


def loaded_model_count() -> int:
    """Number of models currently in the cache (for health checks)."""
    return len(_MODEL_CACHE)


def loaded_models_summary() -> List[dict]:
    """List of loaded models with metadata (for diagnostics)."""
    result = []
    for (commodity, horizon), model in sorted(_MODEL_CACHE.items()):
        result.append({
            "commodity": commodity,
            "horizon_days": horizon,
            "quantiles": model.quantiles,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
        })
    return result


def clear_cache() -> None:
    """Clear the model cache. Useful for testing."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    log.info("Model cache cleared")


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    n = load_all_models()
    print("\n" + "=" * 60)
    print("Model Registry — Load Summary")
    print("=" * 60)
    print(f"Models loaded: {n}")

    summary = loaded_models_summary()
    if summary:
        print("\nLoaded models:")
        for m in summary:
            print(
                f"  {m['commodity']:>10s}  @ {m['horizon_days']:>2d}d  "
                f"quantiles={m['quantiles']}  "
                f"trees={m['n_estimators']}  depth={m['max_depth']}"
            )
    else:
        print("\nNo models found on S3.")
