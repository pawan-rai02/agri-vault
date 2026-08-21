# AgriVault 🌾

AgriVault is an agricultural commodity risk and lending analytics platform designed to support post-harvest financing decisions. It combines mandi prices, NASA satellite weather, NDVI vegetation health, CPI/WPI macro indicators, warehouse infrastructure data, and loan-risk signals into a reproducible Bronze → Silver → Gold data and ML pipeline — served through a live prediction API with real-time weather enrichment.

## Objectives

- Build mandi-wise and commodity-wise short-term price forecasts for 7/15/30-day horizons.
- Produce uncertainty-aware forecasts using a custom Quantile Gradient Boosted Model.
- Fetch live NASA weather data by GPS coordinates for real-time prediction enrichment.
- Resolve user locations to the nearest mandi using geo-based distance matching.
- Feed forecasts, uncertainty, agricultural conditions, market information, warehouse grade, and loan signals into a risk/LTV model.
- Support decisions such as APPROVE, CONDITIONAL, or REJECT with recommended loan-to-value ratios.
- Expose results through a Flask dashboard, JSON API, and interactive prediction form.

## Architecture

```text
External sources
    ├── APMC / Agmarknet        (mandi prices)
    ├── CPI / WPI               (macro indicators)
    ├── WDRA                    (warehouse infrastructure)
    ├── Weather                 (historical daily weather)
    ├── NASA POWER API          (live weather by lat/lon)
    ├── Google Earth Engine     (Sentinel-2 NDVI)
    └── Loan-risk proxy         (synthetic default signals)
            │
            ▼
AWS S3: agrivault-lake-pawan
    ├── raw/                    Bronze: source/collected data
    ├── standardized/           Silver: cleaned/typed/deduplicated
    ├── features/               Gold: model-ready features + serving snapshot
    ├── models/                 Trained models + predictions
    └── reference/              Mandi master + data dictionaries
            │
            ▼
Price forecasting → Risk/LTV model → Decision support → Dashboard/API
                                                         ├── Live Prediction (POST /api/predict)
                                                         ├── NASA weather enrichment
                                                         └── Geo-based mandi resolution
```

## Repository Structure

```text
agri-vault/
├── configs/
│   ├── aws_config.yaml              # AWS CLI profile + S3 bucket
│   └── gee_config.yaml              # Google Earth Engine project
├── data/
│   ├── raw/                         # Bronze (local copies)
│   │   ├── apmc/
│   │   ├── cpi_wpi/{cpi,wpi}/
│   │   ├── wdra/
│   │   ├── weather/
│   │   ├── ndvi/
│   │   └── loans/
│   ├── standardized/                # Silver (local copies)
│   │   ├── apmc/  cpi/  wpi/  wdra/  weather/  ndvi/  loans/
│   ├── features/                    # Gold (local copies)
│   │   ├── price_features/
│   │   └── risk_features/
│   ├── models/
│   │   └── training_summary.csv
│   └── reference/
│       ├── mandi_locations.csv      # 3,158 physical mandis
│       └── apmc/
│           └── apmc_data_dictionary.xlsx
├── notebooks/
│   ├── 02_eda_ndvi.ipynb
│   └── 03_quantile_forecast_baseline.ipynb
├── scripts/
│   ├── gee_fetch_ndvi.py           # GEE NDVI extraction runner
│   ├── generate_loan_proxy.py      # Synthetic loan risk proxy
│   ├── refresh_serving_snapshot.ps1 # Daily snapshot refresh
│   ├── run_spark.ps1               # PySpark standardization runner
│   ├── s3_upload_raw.py            # Upload raw data to S3
│   └── s3_upload_standardized.py   # Upload standardized data to S3
├── src/
│   ├── api/                        # Flask dashboard + prediction API
│   │   ├── app.py                  # App factory, routes, startup
│   │   ├── predict.py              # POST /api/predict endpoint
│   │   └── templates/
│   │       ├── dashboard.html      # KPIs, decision distribution, rankings
│   │       ├── commodity.html      # Commodity detail page
│   │       ├── predict.html        # Interactive LTV prediction form
│   │       └── 404.html
│   ├── features/                   # Gold feature engineering
│   │   ├── build_price_features.py # 9 feature groups, 5.6M rows
│   │   ├── build_risk_features.py  # 7 risk feature groups
│   │   └── build_serving_snapshot.py # Daily snapshot for live API
│   ├── ingestion/
│   │   └── fetch_ndvi.py           # GEE Sentinel-2 NDVI fetcher
│   ├── models/
│   │   ├── risk_ltv_model.py       # Risk/LTV scoring (6 signals)
│   │   └── quantile_gbm/           # Custom from-scratch GBM
│   │       ├── loss.py             # Pinball loss + gradient/hessian
│   │       ├── tree.py             # CART tree for quantile regression
│   │       ├── gradient_boosted_trees.py  # QuantileGBM ensemble
│   │       ├── hypertuner.py       # Walk-forward CV + random search
│   │       └── train.py            # Per-commodity training entry point
│   ├── reference/
│   │   └── build_mandi_locations.py
│   ├── serving/                    # Live prediction support
│   │   ├── model_registry.py       # Loads trained models from S3
│   │   ├── location_resolver.py    # Mandi resolution (text + geo)
│   │   └── nasa_weather.py         # NASA POWER API weather client
│   ├── standardization/            # PySpark Silver cleaning
│   │   ├── spark_session.py
│   │   ├── clean_apmc.py
│   │   ├── clean_weather.py
│   │   ├── clean_ndvi.py
│   │   ├── clean_wdra.py
│   │   ├── clean_wpi_cpi.py
│   │   └── clean_loans.py
│   └── storage/
│       └── s3_client.py            # S3 read/write wrapper (boto3)
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_app.py                 # API + dashboard route tests
│   ├── test_ndvi_join.py
│   ├── test_price_features.py
│   ├── test_quantile_gbm.py
│   └── test_risk_ltv.py
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # web + pipeline services
├── wsgi.py                         # Gunicorn entry point
├── requirements.txt
└── .env.example
```

## Data Sources

### APMC (Mandi Prices)

```text
data/raw/apmc/apmc_market_prices.csv    (~1 GB, 5.4M+ rows, year 2025)
```

Key columns: `report_date`, `state_name`, `district_name`, `market_center`, `market_code`, `latitude`, `longitude`, `commodity`, `variety`, `arrivals_tonnes`, `min_price`, `max_price`, `modal_price`

The APMC dataset already contains latitude/longitude — no external geocoding required.

### Mandi Master

3,158 unique physical mandis after deduplication (market code alone is not unique — code 1765 is reused by Dharashiv and Murum).

```text
data/reference/mandi_locations.csv
Schema: mandi_id, market_code, mandi_name, district, state, latitude, longitude
```

Generated by: `py src/reference/build_mandi_locations.py`

### NASA POWER Weather

```text
data/raw/weather/weather_daily.parquet
```

Parameters: Temperature (T2M), Precipitation (PRECTOTCORR), Humidity (RH2M), Wind Speed (WS2M)

### NDVI (Sentinel-2 via Google Earth Engine)

```text
data/raw/ndvi/ndvi_sentinel2_2025.csv
```

Source: `COPERNICUS/S2_SR_HARMONIZED` → monthly median composites → mandi-level extraction with 500m buffer.

### CPI / WPI / WDRA / Loans

```text
data/raw/cpi_wpi/{cpi,wpi}/wpi_monthly.xlsx
data/raw/wdra/state-wise CSV files
data/raw/loans/loan_risk_proxy.csv
```

## AWS S3 Lake

Bucket: `s3://agrivault-lake-pawan/`

```text
raw/
├── apmc/  cpi_wpi/{cpi,wpi}/  wdra/  weather/  ndvi/  loans/
standardized/
├── apmc/  cpi/  wpi/  wdra/  weather/  ndvi/  loans/
features/
├── price_features/              # Gold: 5.6M rows × 42 cols
├── risk_features/               # Gold: 19K rows × 15 cols
└── serving_snapshot/            # Latest feature vector per mandi×commodity
models/
├── qgbm_{commodity}_{horizon}d.pkl          # Trained Quantile GBM models
├── qgbm_{commodity}_predictions.parquet     # Model predictions
└── risk_ltv_model.pkl                       # Risk/LTV scoring model
reference/
├── mandi_locations.csv
└── apmc/apmc_data_dictionary.xlsx
```

## Live Prediction Pipeline

The prediction endpoint (`POST /api/predict`) is the core of the decision-support system:

```text
User input (commodity + location)
    │
    ├── Text-based mandi resolution (state/district/market → mandi_id)
    ├── Geo-based fallback (lat/lon → nearest mandi via haversine)
    │
    ├── Serving snapshot lookup (latest features per mandi×commodity)
    ├── NASA POWER API → live weather (temp, precip, humidity, wind)
    │
    ├── Quantile GBM forecast (7d/15d/30d bands)
    ├── Historical percentile fallback (when no model available)
    │
    └── Risk/LTV scoring (6 signals → APPROVE / CONDITIONAL / REJECT)
```

### Request

```json
POST /api/predict
{
    "commodity": "WHEAT",
    "latitude": 26.8467,
    "longitude": 80.9462,
    "quantity_kg": 500,
    "warehouse_grade": "A",
    "requested_loan_amount": 1000000
}
```

`state` is optional when `latitude` + `longitude` are provided — the system auto-resolves to the nearest mandi.

### Response

```json
{
    "commodity": "WHEAT",
    "mandi_id": "UTTAR_PRADESH_LUCKNOW_334_LUCKNOW",
    "resolution_type": "geo_nearest",
    "mandi_distance_km": 0.0,
    "used_snapshot": true,
    "nasa_weather_used": true,
    "forecast": {
        "7d":  { "low": 2578.62, "median": 2591.67, "high": 2604.71 },
        "15d": { "low": 2577.06, "median": 2591.67, "high": 2606.27 },
        "30d": { "low": 2574.14, "median": 2591.67, "high": 2609.19 }
    },
    "forecast_method": "historical_percentile_fallback",
    "risk_score": 0.0983,
    "decision": "APPROVE",
    "recommended_ltv_pct": 72.2,
    "recommended_loan_amount": 931380.0,
    "modal_price": 2580.0,
    "explanation": {
        "components": {
            "price_cv": 0.0249,
            "forecast_uncertainty": 0.0323,
            "n_warehouses": 2,
            "warehouse_grade": "A",
            "commodity_tier": 0.2,
            "season": "Rabi"
        },
        "feature_importance": {
            "price_cv": 6.0,
            "forecast_uncertainty": 28.7,
            "warehouse_gap": 27.1,
            "warehouse_grade": 0.0,
            "commodity_tier": 19.1,
            "season_risk": 0.0
        }
    }
}
```

## Feature Engineering

### Price Features (Gold Layer)

Joins at: `mandi × commodity × date` using APMC + Weather + NDVI + CPI + WPI

| Feature Group | Details |
|---|---|
| Price lags | 1d, 7d, 14d, 30d |
| Rolling stats | 7d/14d/30d mean and std-dev |
| Price momentum | 7d percentage change |
| Arrivals | raw + 7d rolling mean |
| Weather | 7d rolling temp, precipitation, humidity |
| NDVI | daily forward-filled + 30d delta |
| Macro | food CPI index + food WPI index (monthly) |
| Temporal | day-of-week, day-of-month, month, is_weekend |
| Targets | forward-looking prices at 7d, 15d, 30d |

Output: 5.6M rows × 42 columns → S3

### Risk Features (Gold Layer)

7 feature groups combining mandi infrastructure, price statistics, portfolio defaults, and commodity characteristics. Output: 19K rows × 15 columns.

### Serving Snapshot

Daily snapshot built from the latest row per `(mandi_id, commodity)` in the Gold price_features table. Provides a fast-loading feature vector for the live prediction API.

```powershell
PYTHONPATH=. python -m src.features.build_serving_snapshot
```

## Forecasting Model

**Custom Quantile GBM** — built from scratch, no sklearn, no LightGBM.

```text
src/models/quantile_gbm/
├── loss.py                 # Pinball loss with analytical gradient/hessian
├── tree.py                 # CART tree for quantile regression
├── gradient_boosted_trees.py  # QuantileGBM ensemble (q10, q50, q90)
├── hypertuner.py           # Walk-forward CV + random search
└── train.py                # Per-commodity training entry point
```

- Trains one model per commodity × horizon (5 commodities × 3 horizons = 15 models)
- Walk-forward cross-validation for hyperparameter tuning
- Produces prediction intervals (q10, q50, q90) for uncertainty-aware forecasts

## Risk / LTV Model

```text
src/models/risk_ltv_model.py
```

Combines 6 risk signals with weighted scoring:

| Signal | Weight | Description |
|---|---|---|
| Price volatility (CV) | 25% | Coefficient of variation of recent prices |
| Forecast uncertainty | 20% | Prediction interval width from Quantile GBM |
| Warehouse coverage | 15% | Number of WDRA warehouses in mandi area |
| Warehouse grade | 10% | Storage quality: A (premium) / B (standard) / C (poor) |
| Commodity perishability | 20% | Hardcoded tier: perishables → higher risk |
| Season risk | 10% | Kharif (monsoon-dependent) vs Rabi (stable) |

Decision thresholds:

```text
APPROVE       — risk < 0.35, LTV 65–75%
CONDITIONAL   — risk 0.35–0.60, LTV 50–65%
REJECT        — risk ≥ 0.60, LTV 40–50%
```

Feature importance (% contribution to total risk) is computed and returned with every prediction.

## Dashboard / API

Flask application serving risk scores, lending decisions, and interactive prediction.

### HTML Pages

| Route | Description |
|---|---|
| `GET /` | Dashboard: KPIs, decision distribution, commodity rankings, state summary |
| `GET /commodity/<name>` | Commodity detail: state breakdown, top districts |
| `GET /predict` | Interactive LTV prediction form with lat/lon + NASA weather |

### JSON API

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Health check (snapshot status, model count) |
| `GET /api/summary` | GET | Overall statistics |
| `GET /api/commodities` | GET | All commodities with risk stats |
| `GET /api/scores` | GET | Filterable risk scores (`?commodity=WHEAT&decision=APPROVE`) |
| `GET /api/scores/<commodity>` | GET | Scores for one commodity |
| `GET /api/decisions` | GET | Decision distribution by commodity |
| `POST /api/predict` | POST | Live prediction with forecast + risk scoring |

### Run

```powershell
# Development
PYTHONPATH=. python -m src.api.app

# Production (Gunicorn)
gunicorn wsgi:app --bind 0.0.0.0:8000 --workers 2 --timeout 120
```

## Docker

```powershell
# Build
docker build -t agrivault .

# Run API
docker run --env-file .env -p 8000:8000 agrivault

# Run a pipeline step
docker compose run pipeline python -m src.standardization.clean_apmc
docker compose run pipeline python -m src.features.build_price_features
docker compose run pipeline python -m src.models.quantile_gbm.train --top-n 5 --quick

# Full stack
docker compose up --build
```

## Google Earth Engine

Project: `silver-aurora-416511`

```powershell
earthengine authenticate
earthengine set_project silver-aurora-416511
py scripts/gee_fetch_ndvi.py --batch-size 25
```

Source: `COPERNICUS/S2_SR_HARMONIZED` → cloud masking (SCL classes 3,8,9,10,11) → monthly median composites → mandi-level extraction with 500m buffer at 10m scale.

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS
cp .env.example .env
# Edit .env with your AWS credentials

# 3. Start the dashboard + API
& { $env:PYTHONPATH = "."; py -m src.api.app }

# Open http://127.0.0.1:5000

# 4. Try a prediction
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"commodity":"WHEAT","latitude":26.8467,"longitude":80.9462}'
```

## All Commands

```powershell
# ── Silver Standardization (PySpark) ──
.\scripts\run_spark.ps1 src.standardization.clean_apmc
.\scripts\run_spark.ps1 src.standardization.clean_weather
.\scripts\run_spark.ps1 src.standardization.clean_ndvi
.\scripts\run_spark.ps1 src.standardization.clean_wdra
.\scripts\run_spark.ps1 src.standardization.clean_wpi_cpi
.\scripts\run_spark.ps1 src.standardization.clean_loans

# ── Gold Features (pandas/PyArrow → S3) ──
PYTHONPATH=. python -m src.features.build_price_features
PYTHONPATH=. python -m src.features.build_risk_features
PYTHONPATH=. python -m src.features.build_serving_snapshot

# ── Model Training ──
PYTHONPATH=. python -m src.models.quantile_gbm.train --commodity WHEAT
PYTHONPATH=. python -m src.models.quantile_gbm.train --top-n 5 --quick
PYTHONPATH=. python -m src.models.risk_ltv_model

# ── NDVI Extraction ──
py scripts/gee_fetch_ndvi.py --batch-size 25

# ── Mandi Master ──
py src/reference/build_mandi_locations.py

# ── Upload to S3 ──
PYTHONPATH=. python scripts/s3_upload_raw.py
PYTHONPATH=. python scripts/s3_upload_standardized.py

# ── API / Dashboard ──
PYTHONPATH=. python -m src.api.app

# ── Evaluation Report ──
PYTHONPATH=. python -m src.evaluation.evaluate --no-s3
PYTHONPATH=. python -m src.evaluation.evaluate --commodity ONION

# ── Tests ──
python -m pytest tests/ -v
```

## Engineering Decisions

1. **Never use `market_code` alone as mandi ID** — code 1765 is reused by two different mandis.
2. **Process the ~1 GB APMC file in chunks** rather than loading entirely into RAM.
3. **NDVI is mandi-level, not row-level** — one NDVI value per mandi per month, not per price observation.
4. **Keep raw NDVI missing values missing** — do not fabricate values in Bronze.
5. **Batch GEE spatial reductions** to avoid Earth Engine memory limits.
6. **NASA weather enriches predictions when the snapshot is stale** — live API fetches real-time data by lat/lon.
7. **Warehouse grade affects risk scoring** — grade A reduces risk, grade C increases it.
8. **Geo-based mandi resolution** — when text lookup fails, find the nearest mandi by haversine distance.
9. **Separate acquisition, cleaning, feature engineering, and modeling** in the repository.

## Model Refresh Cadence

Models should be retrained periodically to avoid staleness as new APMC data lands:

| Step | Command | Frequency |
|---|---|---|
| Silver standardization | `python -m src.standardization.clean_apmc` (etc.) | Weekly |
| Gold price features | `python -m src.features.build_price_features` | Weekly |
| Gold risk features | `python -m src.features.build_risk_features` | Weekly |
| Serving snapshot | `python -m src.features.build_serving_snapshot` | Daily |
| Quantile GBM retrain | `python -m src.models.quantile_gbm.train --top-n 5` | Weekly |
| Risk/LTV retrain | `python -m src.models.risk_ltv_model` | Monthly |

Example PowerShell automation:

```powershell
# Weekly retrain pipeline (Windows Task Scheduler)
$env:PYTHONPATH = "D:\agri-vault"
python -m src.standardization.clean_apmc
python -m src.features.build_price_features
python -m src.features.build_risk_features
python -m src.features.build_serving_snapshot
python -m src.models.quantile_gbm.train --top-n 5
python -m src.models.risk_ltv_model
```

## Project Status

| Component | Status |
|---|---|
| **Data Pipeline** | |
| Raw data on S3 (APMC, CPI, WPI, WDRA, Weather, NDVI, Loans) | ✅ |
| 3,158 mandi master generated | ✅ |
| Silver standardization (7 datasets, PySpark → S3) | ✅ |
| Gold price features (5.6M rows × 42+ cols → S3) | ✅ |
| Gold risk features (19K rows × 15 cols → S3) | ✅ |
| Serving snapshot (daily feature vectors for live API) | ✅ |
| **Phase 4 Features** | |
| Fixed model fallback (quantile_gbm now used instead of fallback) | ✅ |
| Open-Meteo forward weather forecast (16-day ahead) | ✅ |
| Spatial features (state avg price, agro-climatic zone, major market distance) | ✅ |
| Agro-climatic zone reference (ICAR 15 zones) | ✅ |
| Risk/LTV weights fitted from loan-proxy data (logistic regression) | ✅ |
| WPI/CPI leakage fix (1-month lag for publication delay) | ✅ |
| API key auth + rate limiting | ✅ |
| NDVI anomaly (multi-year MODIS baseline) | ⏳ Requires multi-year data pull |
| Multi-year APMC history (2021–2025) | ⏳ Requires data pull from Agmarknet |
| Model evaluation report (pinball loss, naive baseline comparison) | ✅ Script ready, run with `--no-s3` or with S3 access |
| **Models** | |
| Custom Quantile GBM (from scratch, no sklearn) | ✅ |
| Walk-forward CV hypertuner | ✅ |
| 15 models trained (5 commodities × 3 horizons) | ✅ |
| Risk/LTV scoring model (6 signals, data-fitted weights, feature importance) | ✅ |
| **Live Prediction** | |
| POST /api/predict endpoint | ✅ |
| Geo-based mandi resolution (haversine) | ✅ |
| NASA POWER API live weather enrichment | ✅ |
| Open-Meteo forward weather forecast | ✅ |
| Warehouse grade → risk impact | ✅ |
| Feature importance in risk breakdown | ✅ |
| API key auth + rate limiting | ✅ |
| **Interface** | |
| Flask dashboard (KPIs, decisions, rankings) | ✅ |
| Interactive LTV prediction form | ✅ |
| Docker + Gunicorn production setup | ✅ |
| NDVI EDA notebook | ✅ |
| Forecast baseline notebook | ✅ |
| **Validation** | |
| Unit tests (135 passing, including 17 evaluation tests) | ✅ |

---

**Project:** AgriVault
**Data Architecture:** Bronze → Silver → Gold → Models → Dashboard
**Primary Lake:** AWS S3
**Satellite Source:** Google Earth Engine / Sentinel-2
**Live Weather:** NASA POWER API (historical) + Open-Meteo (forecast)
**Forecasting:** Custom Quantile GBM (from scratch)
**Risk/LTV:** 6-signal weighted scoring (logistic-regression fitted) → APPROVE / CONDITIONAL / REJECT
**Interface:** Flask dashboard + JSON API + Docker
