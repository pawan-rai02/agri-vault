# AgriVault 🌾

Agricultural commodity risk and lending analytics platform for post-harvest financing. Combines mandi prices, satellite weather, NDVI vegetation health, macro indicators, warehouse data, and loan-risk signals into a Bronze → Silver → Gold data pipeline, served through a live prediction API.

## Objectives

- Short-term price forecasts (7/15/30-day horizons) per mandi × commodity
- Uncertainty-aware predictions via custom Quantile Gradient Boosted Model
- Live NASA weather enrichment by GPS coordinates
- Geo-based mandi resolution (nearest market by haversine distance)
- Risk scoring and LTV recommendations (APPROVE / CONDITIONAL / REJECT)
- Flask dashboard, JSON API, and interactive prediction form

## Architecture

```
External Sources
├── APMC / Agmarknet          Mandi prices (~5.4M rows)
├── NASA POWER API            Historical + live weather
├── Open-Meteo                Forward weather forecasts (16-day)
├── Google Earth Engine       Sentinel-2 NDVI vegetation index
├── CPI / WPI                 Macro price indices
├── WDRA                      Warehouse infrastructure
├── MODIS                     Multi-year NDVI baseline
└── Loan-risk proxy           Synthetic default signals
        │
        ▼
AWS S3 Lake (agrivault-lake-pawan)
├── raw/              Bronze — source/collected data
├── standardized/     Silver — cleaned, typed, deduplicated
├── features/         Gold — model-ready features + serving snapshot
├── models/           Trained models + predictions
└── reference/        Mandi master + data dictionaries
        │
        ▼
Price Forecasting → Risk/LTV Model → Decision Support → Dashboard/API
```

## Data Pipeline

### Bronze → Silver (PySpark)

Each module reads raw CSV/Parquet from S3, applies validation and cleaning, and writes partitioned Parquet back.

| Module | Data | Key Cleaning Steps |
|--------|------|--------------------|
| `clean_apmc.py` | Mandi prices | Date parsing, price clamping (≥₹0, <₹1Cr), deduplication |
| `clean_weather.py` | Daily weather | Coordinate validation (India bbox), physical bounds clamping |
| `clean_ndvi.py` | Sentinel-2 NDVI | Cloud masking, monthly median composites |
| `clean_wdra.py` | WDRA warehouses | District normalization, capacity casting |
| `clean_wpi_cpi.py` | CPI/WPI indices | Excel parsing, monthly aggregation |
| `clean_loans.py` | Loan risk proxy | Synthetic default signal cleaning |

### Silver → Gold (pandas/PyArrow)

Two feature tables, built without Spark dependency:

**Price Features** — 5.6M rows × 42+ columns at `mandi × commodity × date` grain:

| Group | Features |
|-------|----------|
| Price lags | 1d, 7d, 14d, 30d |
| Rolling stats | 7d/14d/30d mean and std-dev |
| Price momentum | 7d percentage change |
| Arrivals | raw tonnes + 7d rolling mean |
| Weather | 7d rolling temp, precipitation, humidity |
| NDVI | Daily forward-filled + 30d delta |
| MODIS anomaly | Z-score vs 4-year baseline, stress/surplus flags |
| Macro | Food CPI + food WPI index (monthly, 1-month lagged) |
| Spatial | State avg price, nearest mandi avg, agro-climatic zone, distance to major market |
| Temporal | Day-of-week, day-of-month, month, is_weekend |
| Targets | Forward-looking prices at 7d, 15d, 30d |

**Risk Features** — 19K rows × 15 columns combining mandi infrastructure, price statistics, portfolio defaults, and commodity characteristics.

### Serving Snapshot

Daily snapshot of the latest row per `(mandi_id, commodity)` from the Gold price_features table. Powers the live prediction API with fast feature lookups.

## Models

### Quantile GBM (from scratch)

```
src/models/quantile_gbm/
├── loss.py                     Pinball loss + analytical gradient/hessian
├── tree.py                     CART tree for quantile regression
├── gradient_boosted_trees.py   QuantileGBM ensemble (q10, q50, q90)
├── hypertuner.py               Walk-forward CV + random search
└── train.py                    Per-commodity training entry point
```

- One model per commodity × horizon (5 commodities × 3 horizons = 15 models)
- Walk-forward cross-validation for hyperparameter tuning
- Produces prediction intervals (q10, q50, q90) for uncertainty-aware forecasts
- No sklearn, no LightGBM — fully custom implementation

### Risk / LTV Model

Combines 6 risk signals with weighted scoring:

| Signal | Weight | Description |
|--------|--------|-------------|
| Price volatility (CV) | 25% | Coefficient of variation of recent prices |
| Forecast uncertainty | 20% | Prediction interval width from Quantile GBM |
| Warehouse coverage | 15% | Number of WDRA warehouses in mandi area |
| Warehouse grade | 10% | Storage quality: A (premium) / B (standard) / C (poor) |
| Commodity perishability | 20% | Tier-based risk (perishables → higher risk) |
| Season risk | 10% | Kharif (monsoon-dependent) vs Rabi (stable) |

Decision thresholds:

```
APPROVE       — risk < 0.35,  LTV 65–75%
CONDITIONAL   — risk 0.35–0.60, LTV 50–65%
REJECT        — risk ≥ 0.60,  LTV 40–50%
```

## Live Prediction Pipeline

```
User Input (commodity + location)
├── Text mandi resolution (state/district/market → mandi_id)
├── Geo fallback (lat/lon → nearest mandi via haversine)
├── Serving snapshot lookup (latest features per mandi×commodity)
├── NASA POWER API → live historical weather
├── Open-Meteo → forward weather forecast (16-day)
├── Quantile GBM forecast (7d/15d/30d bands)
├── Historical percentile fallback (when no model available)
└── Risk/LTV scoring (6 signals → decision + recommended LTV)
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

### Response

```json
{
    "commodity": "WHEAT",
    "mandi_id": "UTTAR_PRADESH_LUCKNOW_334_LUCKNOW",
    "resolution_type": "geo_nearest",
    "forecast": {
        "7d":  { "low": 2578.62, "median": 2591.67, "high": 2604.71 },
        "15d": { "low": 2577.06, "median": 2591.67, "high": 2606.27 },
        "30d": { "low": 2574.14, "median": 2591.67, "high": 2609.19 }
    },
    "forecast_method": "quantile_gbm",
    "risk_score": 0.0983,
    "decision": "APPROVE",
    "recommended_ltv_pct": 72.2,
    "recommended_loan_amount": 931380.0,
    "modal_price": 2580.0,
    "explanation": { "..." }
}
```

## Dashboard / API

### HTML Pages

| Route | Description |
|-------|-------------|
| `GET /` | KPIs, decision distribution, commodity rankings, state summary |
| `GET /commodity/<name>` | Commodity detail: state breakdown, top districts |
| `GET /predict` | Interactive LTV prediction form |

### JSON API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Health check (snapshot status, model count) |
| `GET /api/summary` | GET | Overall statistics |
| `GET /api/commodities` | GET | All commodities with risk stats |
| `GET /api/scores` | GET | Filterable risk scores (`?commodity=WHEAT&decision=APPROVE`) |
| `GET /api/scores/<commodity>` | GET | Scores for one commodity |
| `GET /api/decisions` | GET | Decision distribution by commodity |
| `POST /api/predict` | POST | Live prediction with forecast + risk scoring |

## Repository Structure

```
agri-vault/
├── configs/
│   ├── aws_config.yaml              AWS CLI profile + S3 bucket
│   └── gee_config.yaml              Google Earth Engine project
├── data/
│   ├── raw/                         Bronze (local copies)
│   ├── standardized/                Silver (local copies)
│   ├── features/                    Gold (local copies)
│   ├── models/                      Training summaries
│   └── reference/                   Mandi master, data dictionaries
├── notebooks/
│   ├── 02_eda_ndvi.ipynb
│   └── 03_quantile_forecast_baseline.ipynb
├── scripts/
│   ├── run_spark.ps1                PySpark standardization runner
│   ├── s3_upload_raw.py             Upload raw data to S3
│   ├── s3_upload_standardized.py    Upload standardized data to S3
│   ├── gee_fetch_ndvi.py            GEE NDVI extraction
│   ├── generate_loan_proxy.py       Synthetic loan risk proxy
│   ├── refresh_serving_snapshot.ps1 Daily snapshot refresh
│   ├── fetch_multiyear_data.ps1     Multi-year data pull
│   └── join_and_upload_ndvi_anomaly.ps1
├── src/
│   ├── api/                         Flask app + prediction endpoint
│   │   ├── app.py                   App factory, routes, startup
│   │   ├── predict.py               POST /api/predict
│   │   └── templates/               dashboard, commodity, predict, 404
│   ├── features/                    Gold feature engineering
│   │   ├── build_price_features.py  Price feature table (5.6M rows)
│   │   ├── build_risk_features.py   Risk feature table (19K rows)
│   │   └── build_serving_snapshot.py Daily snapshot for live API
│   ├── ingestion/
│   │   └── fetch_ndvi.py            GEE Sentinel-2 NDVI fetcher
│   ├── models/
│   │   ├── risk_ltv_model.py        Risk/LTV scoring (6 signals)
│   │   └── quantile_gbm/            Custom from-scratch GBM
│   ├── reference/
│   │   └── build_mandi_locations.py Mandi master generator
│   ├── serving/                     Live prediction support
│   │   ├── model_registry.py        Loads trained models from S3
│   │   ├── location_resolver.py     Mandi resolution (text + geo)
│   │   ├── nasa_weather.py          NASA POWER API client
│   │   └── weather_forecast.py      Open-Meteo forecast client
│   ├── standardization/             PySpark Silver cleaning
│   └── storage/
│       └── s3_client.py             S3 read/write wrapper (boto3)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── wsgi.py
├── requirements.txt
└── .env.example
```

## AWS S3 Lake

Bucket: `s3://agrivault-lake-pawan/`

```
raw/                  Source data (APMC, weather, NDVI, CPI, WPI, WDRA, loans)
standardized/         Cleaned Silver tables (partitioned by state where applicable)
features/
├── price_features/   Gold: 5.6M rows × 42+ cols (partitioned by state)
├── risk_features/    Gold: 19K rows × 15 cols
└── serving_snapshot/ Latest feature vector per mandi×commodity
models/
├── qgbm_*_*.pkl               Trained Quantile GBM models
├── qgbm_*_predictions.parquet Model predictions
└── risk_ltv_model.pkl         Risk/LTV scoring model
reference/
├── mandi_locations.csv         3,158 physical mandis
└── apmc/apmc_data_dictionary.xlsx
logs/predictions/               Prediction audit logs
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS
cp .env.example .env
# Edit .env with your AWS credentials

# 3. Start the dashboard + API
PYTHONPATH=. python -m src.api.app
# Open http://127.0.0.1:5000

# 4. Try a prediction
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"commodity":"WHEAT","latitude":26.8467,"longitude":80.9462}'
```

## All Commands

```bash
# ── Silver Standardization (PySpark) ──
python -m src.standardization.clean_apmc
python -m src.standardization.clean_weather
python -m src.standardization.clean_ndvi
python -m src.standardization.clean_wdra
python -m src.standardization.clean_wpi_cpi
python -m src.standardization.clean_loans

# ── Gold Features (pandas/PyArrow → S3) ──
PYTHONPATH=. python -m src.features.build_price_features
PYTHONPATH=. python -m src.features.build_risk_features
PYTHONPATH=. python -m src.features.build_serving_snapshot

# ── Model Training ──
PYTHONPATH=. python -m src.models.quantile_gbm.train --commodity WHEAT
PYTHONPATH=. python -m src.models.quantile_gbm.train --top-n 5 --quick
PYTHONPATH=. python -m src.models.risk_ltv_model

# ── NDVI Extraction ──
python scripts/gee_fetch_ndvi.py --batch-size 25

# ── Mandi Master ──
python src/reference/build_mandi_locations.py

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

## Docker

```bash
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

```bash
earthengine authenticate
earthengine set_project silver-aurora-416511
python scripts/gee_fetch_ndvi.py --batch-size 25
```

Source: `COPERNICUS/S2_SR_HARMONIZED` → cloud masking (SCL classes 3,8,9,10,11) → monthly median composites → mandi-level extraction with 500m buffer at 10m scale.

## Model Refresh Cadence

| Step | Command | Frequency |
|------|---------|-----------|
| Silver standardization | `python -m src.standardization.clean_*` | Weekly |
| Gold price features | `python -m src.features.build_price_features` | Weekly |
| Gold risk features | `python -m src.features.build_risk_features` | Weekly |
| Serving snapshot | `python -m src.features.build_serving_snapshot` | Daily |
| Quantile GBM retrain | `python -m src.models.quantile_gbm.train --top-n 5` | Weekly |
| Risk/LTV retrain | `python -m src.models.risk_ltv_model` | Monthly |

## Engineering Decisions

1. **Never use `market_code` alone as mandi ID** — code 1765 is reused by two different mandis (Dharashiv and Murum).
2. **Process the ~1 GB APMC file in chunks** rather than loading entirely into RAM.
3. **NDVI is mandi-level, not row-level** — one NDVI value per mandi per month, not per price observation.
4. **Keep raw NDVI missing values missing** — do not fabricate values in Bronze.
5. **Batch GEE spatial reductions** to avoid Earth Engine memory limits.
6. **NASA weather enriches predictions when the snapshot is stale** — live API fetches real-time data by lat/lon.
7. **Open-Meteo provides forward forecasts** — 16-day ahead weather for prediction enrichment.
8. **Warehouse grade affects risk scoring** — grade A reduces risk, grade C increases it.
9. **Geo-based mandi resolution** — when text lookup fails, find the nearest mandi by haversine distance.
10. **CPI/WPI lagged by 1 month** — accounts for publication delay (CPI ~12th, WPI ~14th of following month).

## Project Status

| Component | Status |
|-----------|--------|
| Raw data on S3 (APMC, CPI, WPI, WDRA, Weather, NDVI, Loans) | ✅ |
| 3,158 mandi master generated | ✅ |
| Silver standardization (7 datasets, PySpark → S3) | ✅ |
| Gold price features (5.6M rows × 42+ cols → S3) | ✅ |
| Gold risk features (19K rows × 15 cols → S3) | ✅ |
| Serving snapshot (daily feature vectors for live API) | ✅ |
| Custom Quantile GBM (from scratch, no sklearn) | ✅ |
| Walk-forward CV hypertuner | ✅ |
| 15 models trained (5 commodities × 3 horizons) | ✅ |
| Risk/LTV scoring (6 signals, data-fitted weights) | ✅ |
| POST /api/predict endpoint | ✅ |
| Geo-based mandi resolution (haversine) | ✅ |
| NASA POWER API live weather enrichment | ✅ |
| Open-Meteo forward weather forecast | ✅ |
| Warehouse grade → risk impact | ✅ |
| Feature importance in risk breakdown | ✅ |
| API key auth + rate limiting | ✅ |
| Flask dashboard (KPIs, decisions, rankings) | ✅ |
| Interactive LTV prediction form | ✅ |
| Docker + Gunicorn production setup | ✅ |
| Unit tests (135 passing) | ✅ |
| NDVI anomaly (multi-year MODIS baseline) | ⏳ Requires multi-year data pull |
| Multi-year APMC history (2021–2025) | ⏳ Requires data pull from Agmarknet |

---

**Data Architecture:** Bronze → Silver → Gold → Models → Dashboard
**Primary Lake:** AWS S3
**Satellite Source:** Google Earth Engine / Sentinel-2
**Live Weather:** NASA POWER API (historical) + Open-Meteo (forecast)
**Forecasting:** Custom Quantile GBM (from scratch)
**Risk/LTV:** 6-signal weighted scoring → APPROVE / CONDITIONAL / REJECT
**Interface:** Flask dashboard + JSON API + Docker
