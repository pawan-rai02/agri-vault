# AgriVault – Phase Plan

> This document tracks every task across all phases.
> Status: ✅ Done | 🔄 In Progress | ⏳ Pending

---

## Phase 1: S3 Upload + PySpark Cleaning (Silver Layer)

### Goal
All 6 raw datasets cleaned with PySpark and written as partitioned Parquet to S3 under `standardized/`.

### Setup
| Task | Status |
|---|---|
| Inspect weather parquet schema (lat/lon for spatial join) | ✅ |
| Confirm NDVI is Sentinel-2, 0–1 scale | ✅ |
| Inspect WPI/CPI Excel sheet/column structure | ✅ |
| Inspect APMC CSV exact column names | ✅ |
| Generate synthetic KCC loan proxy (`generate_loan_proxy.py`) | ✅ |
| Install Java JDK 17 (for PySpark) | ✅ |
| Configure S3A / hadoop-aws in `spark_session.py` | ✅ |

### Scripts Written
| File | Status |
|---|---|
| `src/standardization/spark_session.py` | ✅ |
| `src/standardization/clean_apmc.py` | ✅ |
| `src/standardization/clean_weather.py` | ✅ |
| `src/standardization/clean_wdra.py` | ✅ |
| `src/standardization/clean_wpi_cpi.py` | ✅ |
| `src/standardization/clean_ndvi.py` | ✅ |
| `src/standardization/clean_loans.py` | ✅ |
| `scripts/s3_upload_raw.py` | ✅ |
| `scripts/s3_upload_standardized.py` | ✅ |

### Raw S3 Uploads
| Dataset | Status | Notes |
|---|---|---|
| APMC CSV | ✅ | ~1 GB, `raw/apmc/` |
| CPI Excel | ✅ | 101 MB, `raw/cpi_wpi/cpi/` |
| WPI Excel | ✅ | `raw/cpi_wpi/wpi/` |
| WDRA CSVs | ✅ | 26 state files, `raw/wdra/` |
| Weather Parquet | ✅ | `raw/weather/` |
| NDVI CSV | ✅ | 3,158 mandis × 12 months = 30,600 rows |
| Loan proxy CSV | ✅ | 49,764 rows |
| Reference files | ✅ | mandi_locations.csv + data dictionary |

### Standardization Execution
| Script | Status | Output |
|---|---|---|
| `clean_wpi_cpi` | ✅ | WPI 45,336 rows + CPI 61,524 rows |
| `clean_loans` | ✅ | 49,764 KCC proxy rows, default rate 6.31% |
| `clean_apmc` | ✅ | Partitioned by state → S3 |
| `clean_weather` | ✅ | Validated India bounds, written to S3 |
| `clean_wdra` | ✅ | 516 clean rows → S3 |
| `clean_ndvi` | ✅ | 30,594 monthly obs → 1,057,744 daily rows → S3 |

---

## Phase 2: Gold Feature Tables

### Goal
Join all Silver tables at `mandi × commodity × date` grain; produce forecast-ready + risk-ready feature tables.

### Price Features (`src/features/build_price_features.py`)
| Feature Group | Status |
|---|---|
| Script written | ✅ |
| Price lags: 1d, 7d, 14d, 30d | ✅ (code) |
| Rolling mean + std: 7d, 14d, 30d | ✅ (code) |
| Price momentum (7d pct change) | ✅ (code) |
| Arrivals rolling mean (7d) | ✅ (code) |
| Weather join + 7d aggregates | ✅ (code) |
| NDVI join + ndvi_delta_30d | ✅ (code) |
| CPI/WPI monthly join | ✅ (code) |
| Temporal features | ✅ (code) |
| Forward targets: 7d, 15d, 30d | ✅ (code) |
| **Run `build_price_features`** | ✅ 5,577,333 rows × 42 cols → 28 state partitions |

### Risk Features (`src/features/build_risk_features.py`)
| Feature Group | Status |
|---|---|
| Script written | ✅ |
| APMC price summary per mandi | ✅ (code) |
| WDRA warehouse capacity join | ✅ (code) |
| Loan proxy portfolio stats | ✅ (code) |
| Commodity category mapping | ✅ (code) |
| Price CV (volatility proxy) | ✅ (code) |
| Forecast uncertainty slot | ✅ (code, pending Phase 3) |
| Risk score proxy + LTV recommendation | ✅ (code) |
| **Run `build_risk_features`** | ✅ 19,054 rows × 15 cols → S3 |

### Run Results (2026-08-20)
| Commodity | Horizon | RMSE | MAPE | Status |
|---|---|---|---|---|
| WHEAT | 7d | 112.95 | 2.5% | ✅ |
| WHEAT | 15d | 123.93 | 3.0% | ✅ |
| WHEAT | 30d | 139.15 | 3.5% | ✅ |
| ONION | 7d | 383.66 | 13.5% | ✅ |
| TOMATO | 7d | 777.00 | 20.3% | ✅ |
| POTATO | 7d | 805.96 | 17.1% | ✅ |
| BRINJAL | 7d | 955.59 | 22.8% | ✅ |
| GREEN CHILLI | 7d | 3278.72 | 23.4% | ✅ |

### Performance Optimizations (2026-08-20)
| Fix | Status |
|---|---|
| CART tree: quantile-binned thresholds (64 per feature) | ✅ |
| CART tree: row subsampling for split-finding (4096 max) | ✅ |
| Training data cap (50K rows in quick mode) | ✅ |

### Bug Fixes Applied (2026-08-20)
| Fix | Status |
|---|---|
| `build_risk_features.py`: S3 pagination — `list_objects_v2` now uses `ContinuationToken` to read >1000 objects | ✅ |
| `train.py`: S3 key paths — removed `../` prefix traversal, models now saved to `models/` directly | ✅ |

---

## Phase 3: Custom Quantile GBM (from scratch)

### Goal
Build a gradient-boosted quantile regression model entirely from scratch (no sklearn/LightGBM) for 7d/15d/30d price forecasting.

### Core ML Implementation
| File | Status |
|---|---|
| `src/models/quantile_gbm/__init__.py` | ✅ |
| `src/models/quantile_gbm/loss.py` — pinball loss + gradient/hessian | ✅ |
| `src/models/quantile_gbm/tree.py` — CART tree with quantile leaf values | ✅ |
| `src/models/quantile_gbm/gradient_boosted_trees.py` — ensemble | ✅ |
| `src/models/quantile_gbm/hypertuner.py` — walk-forward CV + random search | ✅ |
| `src/models/quantile_gbm/train.py` — entry point, per-commodity training | ✅ |
| **Run `train.py --top-n 5 --quick`** | ✅ 5 commodities × 3 horizons = 15 models on S3 |

### Custom ML Functions Implemented
| Function | File | Description |
|---|---|---|
| `pinball_loss()` | `loss.py` | Quantile regression loss function |
| `gradient()` | `loss.py` | Negative gradient (pseudo-residuals) for GBM |
| `hessian()` | `loss.py` | Second derivative (constant=1 for pinball) |
| `leaf_value_quantile()` | `loss.py` | Optimal leaf value via sample quantile |
| `CARTQuantileTree` | `tree.py` | CART regression tree for quantile regression |
| `QuantileGBM` | `gradient_boosted_trees.py` | Gradient-boosted ensemble (one GBM per quantile) |
| `walk_forward_splits()` | `hypertuner.py` | Expanding-window time-series CV |
| `random_search()` | `hypertuner.py` | Random hyperparameter search with walk-forward CV |

### Risk / LTV Model (`src/models/risk_ltv_model.py`)
| Component | Status |
|---|---|
| Rules-based risk scoring (price CV, forecast uncertainty, warehouse, commodity, season) | ✅ |
| Decision mapping: APPROVE / CONDITIONAL / REJECT | ✅ |
| LTV recommendation (0.40–0.75 based on risk) | ✅ |
| Forecast uncertainty from quantile model predictions | ✅ |
| **Run `risk_ltv_model`** | ✅ 19,054 mandi×commodity scored → S3 |
| Tests: 17 unit tests | ✅ |

**Run Results:**
- 19,054 rows scored across all commodities
- Decisions: APPROVE ~30%, CONDITIONAL ~65%, REJECT ~5%
- High-risk commodities (BRINJAL, GREEN CHILLI) → more REJECT decisions
- Low-risk commodities (COTTON, GROUNDNUT) → mostly APPROVE

---

## Phase 4: Validation, Tests, Notebooks

### Unit Tests
| File | Status | Notes |
|---|---|---|
| `tests/test_ndvi_join.py` | ✅ | 12 tests: clean_ndvi_base + forward_fill_to_daily |
| `tests/test_price_features.py` | ✅ | 22 tests: lags, rolling, momentum, joins, temporal, targets |
| `tests/test_quantile_gbm.py` | ✅ | 25 tests: loss, gradient, tree, ensemble, walk-forward CV |
| `tests/test_risk_ltv.py` | ✅ | 17 tests: risk scoring, decisions, LTV, null handling |

**Total: 47 tests passing (verified 2026-08-20)**

### Notebooks
| File | Status | Description |
|---|---|---|
| `notebooks/02_eda_ndvi.ipynb` | ✅ | NDVI EDA: distribution, temporal coverage, spatial map, time series |
| `notebooks/03_quantile_forecast_baseline.ipynb` | ✅ | Forecast demo: load features from S3, train QuantileGBM, visualize PI |

### Documentation
| File | Status |
|---|---|
| Update `docs/phase2_plan.md` | ✅ |
| Update `README.md` status table | ✅ |

---

## Phase 5: Production Readiness

### Goal
Make AgriVault deployable, testable, and maintainable in a production environment.

### Status: ✅ P0 items completed 2026-08-20

### P0 — Completed
| Task | Status | File(s) |
|---|---|---|
| Add `requirements.txt` with pinned dependencies | ✅ | `requirements.txt` |
| Add `wsgi.py` Gunicorn entrypoint | ✅ | `wsgi.py` |
| Add `.env` support via `python-dotenv` | ✅ | `src/api/app.py`, `wsgi.py` |
| Add `AGRIVAULT_SECRET_KEY` env var for Flask | ✅ | `src/api/app.py` |
| Add `/health` endpoint (liveness + readiness probe) | ✅ | `src/api/app.py` |
| Add `.env.example` template | ✅ | `.env.example` |
| Add `Dockerfile` (Python 3.12 + Java 17 + Gunicorn) | ✅ | `Dockerfile` |
| Add `docker-compose.yml` (web + pipeline services) | ✅ | `docker-compose.yml` |
| Add `.dockerignore` | ✅ | `.dockerignore` |
| Fix `/api/scores` limit parameter validation | ✅ | `src/api/app.py` |

---

## Phase 6: P1 — Code Quality & Testing

### Goal
Eliminate code duplication, add API test coverage, and establish consistent conventions.

### 6.1 Deduplicate S3 Helpers ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Centralize all S3 read/write logic in `src/storage/s3_client.py` | P1 | Medium | Currently duplicated across `build_price_features.py`, `build_risk_features.py`, `risk_ltv_model.py`, `train.py` |
| Add Hive partition extraction to `S3Client` | P1 | Small | The `read_parquet_s3()` logic that extracts `state=X/` from S3 key paths exists in `build_price_features.py` but not in the shared client |
| Add `read_parquet_s3()` and `list_parquet_keys()` to `S3Client` | P1 | Small | These are copy-pasted in 4 files |
| Update all consumers to use `S3Client` | P1 | Medium | `build_price_features.py`, `build_risk_features.py`, `risk_ltv_model.py`, `train.py`, `app.py` |

**Target files:**
```
src/storage/s3_client.py          ← add read_parquet_s3(), list_parquet_keys(),
                                    extract_hive_partitions()
src/features/build_price_features.py  ← remove local s3 helpers, use S3Client
src/features/build_risk_features.py   ← remove local s3 helpers, use S3Client
src/models/risk_ltv_model.py         ← remove local s3 helpers, use S3Client
src/models/quantile_gbm/train.py      ← remove local s3 helpers, use S3Client
src/api/app.py                       ← use S3Client for data loading
```

### 6.2 Add `conftest.py` ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Create `tests/conftest.py` with shared fixtures | P1 | Small | Extract Spark session fixture from `test_ndvi_join.py` (session-scoped, shared across all PySpark tests) |
| Add shared synthetic data fixtures | P1 | Small | `simple_apmc`, `simple_weather`, `simple_ndvi` are defined in `test_price_features.py` — share via conftest |
| Remove duplicated fixtures from individual test files | P1 | Small | Update `test_ndvi_join.py` and `test_price_features.py` to use shared fixtures |

**Target files:**
```
tests/conftest.py              ← new: shared Spark session, synthetic DataFrames
tests/test_ndvi_join.py        ← remove local spark fixture, use conftest
tests/test_price_features.py   ← remove local fixtures, use conftest
```

### 6.3 Add API Tests ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Create `tests/test_app.py` | P1 | Medium | Unit tests for all Flask routes |
| Test `/health` endpoint | P1 | Small | Returns 503 when data not loaded, 200 when loaded |
| Test `/api/summary` | P1 | Small | Returns correct schema and types |
| Test `/api/scores` with filters | P1 | Small | Test commodity, decision, state, limit params |
| Test `/api/scores` with invalid limit | P1 | Small | Verify ValueError is caught gracefully |
| Test `/api/scores/<commodity>` | P1 | Small | Test found vs. not-found (404) |
| Test `/api/decisions` | P1 | Small | Returns decision distribution |
| Test `/api/commodities` | P1 | Small | Returns commodity list with stats |
| Test HTML routes `/` and `/commodity/<name>` | P1 | Small | Verify render_template is called |
| Mock S3 data loading | P1 | Medium | Use `unittest.mock.patch` to avoid real S3 calls |

**Target file:**
```
tests/test_app.py  ← new: ~15-20 tests covering all API endpoints
```

**Example test structure:**
```python
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.api.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_scored_df():
    """Synthetic scored data matching the real schema."""
    return pd.DataFrame({
        "state": ["MAHARASHTRA"] * 4,
        "district": ["PUNE"] * 4,
        "commodity": ["WHEAT", "TOMATO", "WHEAT", "ONION"],
        "risk_score": [0.25, 0.55, 0.30, 0.65],
        "decision": ["APPROVE", "CONDITIONAL", "APPROVE", "REJECT"],
        "recommended_ltv": [0.72, 0.55, 0.68, 0.42],
        "price_cv": [0.08, 0.27, 0.06, 0.33],
        "forecast_uncertainty": [0.20, 0.50, 0.15, 0.60],
        "n_warehouses": [2, 0, 5, 0],
        "mandi_mean_price": [2500.0, 3000.0, 2400.0, 1800.0],
        "commodity_category": ["Cereal", "Vegetable", "Cereal", "Vegetable"],
    })

class TestHealthEndpoint:
    def test_returns_503_when_no_data(self, client):
        with patch("src.api.app._scored_df", None):
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json["status"] == "degraded"

    def test_returns_200_when_data_loaded(self, client, mock_scored_df):
        with patch("src.api.app._scored_df", mock_scored_df):
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json["status"] == "healthy"
            assert resp.json["rows_loaded"] == 4

class TestApiScores:
    def test_returns_data(self, client, mock_scored_df):
        with patch("src.api.app._scored_df", mock_scored_df):
            resp = client.get("/api/scores")
            assert resp.status_code == 200
            assert resp.json["count"] == 4

    def test_filter_by_commodity(self, client, mock_scored_df):
        with patch("src.api.app._scored_df", mock_scored_df):
            resp = client.get("/api/scores?commodity=WHEAT")
            assert resp.json["count"] == 2

    def test_invalid_limit_returns_default(self, client, mock_scored_df):
        with patch("src.api.app._scored_df", mock_scored_df):
            resp = client.get("/api/scores?limit=abc")
            assert resp.status_code == 200  # graceful fallback

    def test_commodity_not_found(self, client, mock_scored_df):
        with patch("src.api.app._scored_df", mock_scored_df):
            resp = client.get("/api/scores/NONEXISTENT")
            assert resp.status_code == 404
```

### 6.4 Remove Empty File ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Delete `src/models/quantile_forecast_model.py` | P1 | Tiny | Empty file, functionality lives in `quantile_gbm/` |

---

## Phase 7: P2 — DevOps, Monitoring & Observability

### Goal
Add CI/CD, structured logging, type checking, and production monitoring.

### 7.1 CI/CD Pipeline ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Create `.github/workflows/ci.yml` | P2 | Medium | GitHub Actions: lint, type-check, test on every push/PR |
| Add `ruff` for linting + formatting | P2 | Small | Replace flake8/black/isort with a single tool |
| Add `mypy` for static type checking | P2 | Medium | Add `py.typed` marker, configure `mypy.ini` |
| Add `pre-commit` hooks | P2 | Small | Run ruff + mypy before each commit |

**Target files:**
```
.github/workflows/ci.yml    ← new: lint → type-check → test → build
pyproject.toml               ← new: ruff + mypy config (or ruff.toml + mypy.ini)
.pre-commit-config.yaml      ← new: ruff + mypy hooks
```

**Example CI workflow:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint
        run: ruff check src/ tests/
      - name: Type check
        run: mypy src/ --ignore-missing-imports
      - name: Test
        run: PYTHONPATH=. pytest tests/ -v
```

### 7.2 Structured Logging ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Create `src/logging_config.py` | P2 | Small | Centralized logging setup with JSON format option |
| Update all modules to use centralized config | P2 | Medium | Replace per-module `logging.basicConfig()` calls |
| Add request logging middleware to Flask | P2 | Small | Log every API request with method, path, status, latency |
| Add correlation IDs | P2 | Medium | Generate UUID per request, include in all log lines |

### 7.3 Production Flask Hardening ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add CORS support | P2 | Small | `flask-cors` for cross-origin API access |
| Add request rate limiting | P2 | Small | `flask-limiter` to prevent abuse |
| Add request size limits | P2 | Small | Prevent large payload attacks |
| Add error handlers (404, 500) | P2 | Small | Return JSON errors instead of HTML stack traces |
| Add API versioning prefix | P2 | Small | `/api/v1/scores` instead of `/api/scores` |

### 7.4 Model Versioning ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add `models/manifest.json` to S3 | P2 | Medium | Track model versions, metrics, training dates |
| Add timestamp suffix to model filenames | P2 | Small | `qgbm_WHEAT_7d_v20260820.pkl` instead of overwriting |
| Add `--version` flag to `train.py` | P2 | Small | Tag each training run with a version |
| Add model loading by version in API | P2 | Small | Serve predictions from a specific model version |

---

## Phase 8: P2 — Data Quality & Pipeline Robustness

### Goal
Add data validation, incremental processing, and error recovery.

### 8.1 Data Validation ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add `pandera` schemas for Silver tables | P2 | Medium | Define expected column names, types, ranges for each cleaned dataset |
| Add `pandera` schemas for Gold tables | P2 | Medium | Validate feature tables before writing to S3 |
| Add column-name assertions between pipeline stages | P2 | Small | Fail fast if upstream changes break downstream |
| Add data quality checks (null rates, value ranges) | P2 | Medium | Log warnings when data quality degrades |

**Target files:**
```
src/schemas/__init__.py           ← new: schema package
src/schemas/apmc.py               ← new: pandera schema for APMC Silver
src/schemas/weather.py            ← new: pandera schema for Weather Silver
src/schemas/ndvi.py               ← new: pandera schema for NDVI Silver
src/schemas/price_features.py     ← new: pandera schema for Gold price features
src/schemas/risk_features.py      ← new: pandera schema for Gold risk features
```

### 8.2 Incremental Pipeline Processing ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add `--start-date` / `--end-date` to feature builders | P2 | Medium | Process only a date range instead of all data |
| Add partition-level deduplication | P2 | Medium | Skip S3 partitions that haven't changed |
| Add `--dry-run` flag to all pipeline scripts | P2 | Small | Preview what would be processed |
| Add checkpoint/resume to long-running jobs | P2 | Large | Resume from last successful partition on failure |

### 8.3 Error Handling & Retry ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add retry decorator for S3 operations | P2 | Small | `tenacity` library with exponential backoff |
| Add retry decorator for GEE API calls | P2 | Small | Handle transient Earth Engine errors |
| Add try/except with structured error logging in pipeline scripts | P2 | Medium | Replace bare `except Exception` in `fetch_ndvi.py` |
| Add pipeline run ID (UUID) for traceability | P2 | Small | Include in all log lines and S3 output metadata |

---

## Phase 9: P3 — Advanced Improvements

### Goal
Performance optimization, caching, and advanced monitoring.

### 9.1 API Caching ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add Redis or in-memory cache for scored data | P3 | Medium | Avoid re-downloading from S3 on every request |
| Add cache invalidation on data refresh | P3 | Small | TTL-based or manual invalidation endpoint |
| Add DuckDB as local query engine | P3 | Medium | Faster-than-pandas querying for API responses |

### 9.2 Monitoring & Alerting ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add Prometheus metrics endpoint | P3 | Medium | Request count, latency, error rate, data freshness |
| Add Grafana dashboard | P3 | Medium | Visualize API health, pipeline status, model performance |
| Add Slack/email alerts for pipeline failures | P3 | Medium | Notify team when critical pipelines fail |
| Add data freshness monitoring | P3 | Small | Alert when data is older than expected threshold |

### 9.3 Risk Model Enhancement ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Add `.fit()` method to `RiskLTVModel` | P3 | Large | Train from actual default data when available |
| Add calibration against real loan outcomes | P3 | Large | Validate decision thresholds with real data |
| Add A/B testing framework for model variants | P3 | Large | Compare rules-based vs. learned model performance |
| Add feature importance analysis | P3 | Medium | SHAP or permutation importance for risk factors |

### 9.4 Performance Optimization ⏳
| Task | Priority | Effort | Description |
|---|---|---|---|
| Profile and optimize `build_price_features.py` | P3 | Medium | The 5.6M row join is slow — consider chunked processing |
| Add Spark-based feature engineering option | P3 | Large | For datasets that exceed pandas memory limits |
| Optimize GEE NDVI extraction with `ee.batch` | P3 | Medium | Use Earth Engine's batch export API for large-scale runs |

---

## Dependency Graph

```
Phase 1: Silver                     Phase 2: Gold           Phase 3: Model
─────────────────────────────────── ─────────────────────── ────────────────
clean_apmc    ─┐
clean_weather  ├─→ build_price_features ─→ train.py (7d/15d/30d)
clean_ndvi     │                                    │
clean_wpi_cpi ─┘                                    ↓
                                    build_risk_features ← forecast uncertainty
clean_wdra    ─┐
clean_loans   ─┘
```

---

## Architecture Reminder

```
Bronze (raw/)  →  Silver (standardized/)  →  Gold (features/)  →  models/
```

S3 bucket: `s3://agrivault-lake-pawan/`

---

## Changelog

### 2026-08-20 (production readiness)
- Added `requirements.txt` with all 12 dependencies pinned
- Added `wsgi.py` Gunicorn entrypoint with structured logging
- Added `.env` support via `python-dotenv` for all configuration
- Added `AGRIVAULT_SECRET_KEY` env var for Flask session signing
- Added `/health` endpoint (liveness + readiness probe, returns 200/503)
- Added `.env.example` template with all env vars documented
- Added `Dockerfile` (Python 3.12 + Java 17 + Gunicorn, multi-stage build)
- Added `docker-compose.yml` (web + pipeline services with healthcheck)
- Added `.dockerignore` to keep build context lean
- Fixed `/api/scores` limit parameter validation (catches ValueError)
- Documented Phase 5-9 detailed next steps in this plan

### 2026-08-20 (evening)
- Trained WHEAT + top-5 commodities (ONION, TOMATO, POTATO, BRINJAL, GREEN CHILLI) across 7d/15d/30d horizons
- All 15 models + predictions saved to S3 `models/`
- Optimized CART tree: quantile-binned thresholds (64), row subsampling (4096), training data cap (50K in quick mode)
- Fixed WDRA column names (`capacityin_mt` → `capacity_mt`, `wh_name` → `warehouse_name`)
- Fixed WDRA Hive partition extraction (`state` from S3 key path)
- Added `--quick` flag for fast iteration
- Added `build_risk_features` S3 pagination fix and Hive partition extraction
- Fixed `train.py` S3 key paths (removed `../` traversal)

### 2026-08-20 (afternoon)
- Fixed S3 pagination bug in `build_risk_features.py` (>1000 objects were being missed)
- Fixed S3 key paths in `train.py` (removed `../` traversal, models now at `models/`)
- Created `tests/test_price_features.py` — 22 unit tests for feature engineering
- Created `tests/test_quantile_gbm.py` — 25 unit tests for custom ML model
- Created `notebooks/02_eda_ndvi.ipynb` — NDVI exploratory data analysis
- Created `notebooks/03_quantile_forecast_baseline.ipynb` — forecast pipeline demo
- All 47 tests verified passing
- Updated README.md with current status
