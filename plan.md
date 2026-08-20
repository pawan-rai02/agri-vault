# AgriVault — Phase 3: Live Prediction App

## Overview

Transform AgriVault from a batch pipeline (S3 → Spark → pre-scored tables → dashboard) into a live prediction app. The core challenge: compute predictions on-demand in <2 seconds for a crop/location combo a user types in.

**Key insight:** The Quantile GBM takes a 42-column feature vector (price_lag_7d, price_7d_volatility, ndvi_30d_delta, etc.), not (commodity, state). We need a fast feature-serving path.

---

## Part 1: Feature Serving Store ✅ Completed

> **Goal:** Create a nightly-refreshed snapshot of the latest feature vectors, loadable into Flask app memory.

### 1.1 Build `build_serving_snapshot.py` ✅ Completed
- [x] Create `src/features/build_serving_snapshot.py`
- [x] Read Gold `price_features` from S3
- [x] Extract latest row per `(mandi_id, commodity)` using `groupby().head(1)` on desc-sorted data
- [x] Write Parquet to `s3://agrivault-lake-pawan/features/serving_snapshot/latest.parquet`
- [x] Drop target columns (future-leaking) before writing
- [x] Run manually and verify Parquet loads correctly in a notebook
- [x] Test: confirm snapshot has one row per active mandi-commodity pair

### 1.2 Load snapshot into Flask app at startup ✅ Completed
- [x] Add `_serving_snapshot: pd.DataFrame` global to `src/api/app.py`
- [x] Add `_load_serving_snapshot()` helper function with graceful fallback
- [x] Load from S3 Parquet at startup (alongside scored data)
- [x] Expose snapshot status in `/health` endpoint (`serving_snapshot_loaded`, `serving_snapshot_rows`)
- [x] Test: verify snapshot loads and is queryable

### 1.3 Test with live data ✅ Completed
- [x] Verify `load_snapshot()` imports correctly from `build_serving_snapshot.py`
- [x] Verify graceful fallback when snapshot doesn't exist on S3 yet
- [x] Confirm existing tests still pass (snapshot loading is lazy, no breaking changes)

---

## Part 2: Location Resolver ✅ Completed

> **Goal:** Map user input (state, district, market) to a `mandi_id`, with nearest-neighbor fallback.

### 2.1 Create `src/serving/` module ✅ Completed
- [x] Create directory `src/serving/`
- [x] Create `src/serving/__init__.py` with package docstring
- [x] Load `data/reference/mandi_locations.csv` at module level (cached)

### 2.2 Implement `location_resolver.py` ✅ Completed
- [x] Create `src/serving/location_resolver.py`
- [x] Implement `haversine_km(lat1, lon1, lat2, lon2)` using Haversine formula
- [x] Implement `resolve_mandi(state, district=None, market=None)` → `(mandi_id, resolution_type)`
  - Exact match on `mandi_name + state`
  - District + state fallback
  - State-only fallback
  - Returns `"exact_match"`, `"district_fallback"`, `"state_fallback"`, or `"not_found"`
- [x] Implement `nearest_mandi_with_data(lat, lon, available_mandi_ids, top_n=1)` for geo fallback
- [x] CLI smoke test in `__main__`

### 2.3 Test resolver ✅ Completed
- [x] Test with known mandi names (exact match)
- [x] Test with misspelled/deliberately-wrong mandi names (fallback behavior)
- [x] Test edge cases: empty state, no matching district

---

## Part 3: Model Registry ✅ Completed

> **Goal:** Load all trained Quantile GBM models into memory once at startup.

### 3.1 Create `src/serving/model_registry.py` ✅ Completed
- [x] Create `src/serving/model_registry.py`
- [x] Implement `load_all_models(commodities, horizons, s3)` — loads from `models/qgbm_{commodity}_{horizon}d.pkl`
- [x] Cache keyed by `(commodity, horizon)` — each model handles all quantiles (q10, q50, q90)
- [x] Implement `get_model(commodity, horizon)` → model or None
- [x] Implement `loaded_model_count()` for health check
- [x] Implement `loaded_models_summary()` for diagnostics
- [x] Implement `clear_cache()` for testing
- [x] CLI smoke test in `__main__`

### 3.2 Test model loading ✅ Completed
- [x] Model format confirmed: `QuantileGBM` instances pickled to `.pkl`
- [x] Key pattern: `qgbm_{commodity}_{horizon}d.pkl`
- [x] Graceful fallback: missing models logged but don't crash startup

### 3.3 Integrate into Flask startup ✅ Completed
- [x] Import and call `load_all_models()` in `app.py` at startup
- [x] Expose `models_loaded` count in `/health` endpoint
- [x] Graceful error handling: if models fail to load, app still starts

---

## Part 4: `/api/predict` Endpoint ✅ Completed

> **Goal:** New POST endpoint that takes user input and returns forecast, risk score, decision, and LTV recommendation.

### 4.1 Create `src/api/predict.py` ✅ Completed
- [x] Create `src/api/predict.py`
- [x] Create Flask Blueprint named `"predict"`
- [x] Implement `POST /api/predict` endpoint:
  - Parse JSON body: `commodity`, `state`, `district`, `market`, `requested_loan_amount`, `quantity_kg`, `warehouse_grade`
  - Validate required fields (`commodity`, `state`)
  - Resolve mandi via `resolve_mandi()` with 4-level fallback
  - Look up feature vector from serving snapshot
  - Run forecast for horizons [7, 15, 30] × quantiles [0.10, 0.50, 0.90]
  - Run `RiskLTVModel.score()` with live features
  - Return JSON with all required fields
- [x] Build feature vector from snapshot columns
- [x] Compute risk score, decision, and recommended LTV
- [x] Calculate recommended loan amount (if quantity + loan amount provided)
- [x] Return detailed explanation with risk components
- [x] Include request_id and duration_ms for debugging

### 4.2 Register blueprint in `app.py` ✅ Completed
- [x] Import and register `predict_bp` in `src/api/app.py`
- [x] Existing tests still pass (no breaking changes)

### 4.3 Test endpoint ✅ Completed
- [x] Endpoint validates input and returns proper HTTP status codes
- [x] Handles missing snapshot gracefully (503)
- [x] Handles missing mandi data (404)
- [x] Handles missing commodity at mandi (404 with available commodities list)
- [x] Logs predictions to S3 for monitoring

---

## Part 5: Fallback Tiers for Unmodeled Commodities ✅ Completed

> **Goal:** Handle commodities without a trained model gracefully — not silently fail.

### 5.1 Implement tier logic ✅ Completed
- [x] Define three tiers:
  - **Tier 1:** Trained model available → full QuantileGBM forecast band + risk/LTV
  - **Tier 2:** No model but price stats available → historical percentile fallback (`price_mean_30d ± 1.28σ`, scaled by horizon)
  - **Tier 3:** No data at all → `"insufficient_data"` response (HTTP 404)
- [x] `forecast_method` field included in every response (`"quantile_gbm"` / `"historical_percentile_fallback"` / `"insufficient_data"`)

### 5.2 Implement historical fallback ✅ Completed
- [x] Percentile-based fallback in `_run_forecast()` when `get_model()` returns None
- [x] Uses `price_mean_30d` and `price_std_30d` from the serving snapshot
- [x] Band width scales with horizon: `scale = 1.0 + (horizon/30) * 0.5`
- [x] Labeled clearly as fallback in response

### 5.3 Handle Tier 3 (no data) ✅ Completed
- [x] Returns `{"error": "insufficient_data"}` with HTTP 404
- [x] All three tiers tested via `_run_forecast()` logic

---

## Part 6: Prediction Form (Frontend) ✅ Completed

> **Goal:** A user-facing form that submits to `/api/predict` and renders results with charts.

### 6.1 Create `predict.html` template ✅ Completed
- [x] Create `src/api/templates/predict.html`
- [x] Form fields: commodity, state, district, market, quantity, warehouse grade (A/B/C), requested loan amount
- [x] Submit button → POST to `/api/predict` via JavaScript fetch
- [x] Add `GET /predict` route in `app.py`

### 6.2 Add JavaScript for API call + chart rendering ✅ Completed
- [x] Handle form submission with `fetch()` POST
- [x] Parse JSON response and populate results
- [x] Render forecast band as canvas chart (high/median/low lines + shaded confidence band)
- [x] Display risk score (color-coded), decision (badge), recommended LTV, max loan amount
- [x] Show `resolution_type` info when using fallback mandi
- [x] Risk breakdown table with all components

### 6.3 Add navigation link ✅ Completed
- [x] Add "Predict LTV" nav link to `dashboard.html` header
- [x] Add nav link to `commodity.html` header
- [x] Style consistently with existing pages (same green gradient, card styles, badges)

### 6.4 Test frontend ✅ Completed
- [x] Form validates required fields (commodity, state)
- [x] Error box displays API errors clearly
- [x] Results section hidden until prediction succeeds

---

## Part 7: Prediction Logging ✅ Completed

> **Goal:** Log every prediction request+response to S3 for monitoring and future drift detection.

### 7.1 Implement logging in `predict.py` ✅ Completed
- [x] `_log_prediction()` function in `predict.py` (lines 211–238)
- [x] Writes JSON to `s3://agrivault-lake-pawan/logs/predictions/YYYY-MM-DD/{request_id}.json`
- [x] Includes: request_id, timestamp (ISO), duration_ms, full request body, mandi_id, resolution_type, commodity, forecast_method, risk_score, decision, recommended_ltv_pct
- [x] Wrapped in try/except — never crashes the prediction request
- [x] Called at the end of every `/api/predict` request

---

## Part 8: Nightly Snapshot Refresh ✅ Completed

> **Goal:** Automate the serving snapshot to refresh daily after the Gold pipeline completes.

### 8.1 Create scheduling script ✅ Completed
- [x] Create `scripts/refresh_serving_snapshot.ps1` (Windows Task Scheduler)
- [x] Runs `python -m src.features.build_serving_snapshot`
- [x] Logs output to `logs/snapshot_refresh_YYYY-MM-DD.log`
- [x] Error handling with `$ErrorActionPreference = "Stop"`
- [x] Duration tracking and exit code reporting
- [x] Follows same pattern as existing `run_spark.ps1`

### 8.2 Task Scheduler setup instructions ✅ Completed
- [x] Script includes Task Scheduler configuration in header comments
- [x] Action: `PowerShell.exe -ExecutionPolicy Bypass -File "D:\agri-vault\scripts\refresh_serving_snapshot.ps1"`
- [x] Trigger: Daily at 03:00 AM (after Gold pipeline)

---

## Part 9 (Optional): Deployment ⬜ Pending

> **Goal:** For public demo, containerize and deploy to a cloud service.

### 9.1 Dockerize the app ⬜ Pending
- [ ] Update `Dockerfile` for the new serving path
- [ ] Ensure S3 credentials work via environment variables / IAM role
- [ ] Test Docker build and run locally

### 9.2 Deploy to EC2 or Elastic Beanstalk ⬜ Pending
- [ ] Provision small EC2 instance (free-tier eligible)
- [ ] Deploy with gunicorn: `gunicorn -w 4 src.api.app:app`
- [ ] Put behind Nginx reverse proxy
- [ ] Configure IAM role for S3 access
- [ ] Verify health endpoint returns 200

### 9.3 Security review ⬜ Pending
- [ ] Ensure no hardcoded AWS credentials
- [ ] Environment variables or IAM roles only
- [ ] HTTPS via Let's Encrypt or AWS certificate

---

## Summary

| Part | Description | Status |
|------|-------------|--------|
| 1 | Feature Serving Store | ✅ Completed |
| 2 | Location Resolver | ✅ Completed |
| 3 | Model Registry | ✅ Completed |
| 4 | `/api/predict` Endpoint | ✅ Completed |
| 5 | Fallback Tiers | ✅ Completed |
| 6 | Prediction Form (Frontend) | ✅ Completed |
| 7 | Prediction Logging | ✅ Completed |
| 8 | Nightly Snapshot Refresh | ✅ Completed |
| 9 | Deployment (Optional) | ⬜ Pending |

**Legend:** ✅ Completed | ⬜ Pending | 🔄 In Progress

---

*Last updated: 2026-08-20 (Parts 1-8 completed)*
*Generated with Codebuff 🤖*
