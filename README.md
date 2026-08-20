# AgriVault 🌾

AgriVault is an agricultural commodity risk and lending analytics platform designed to support post-harvest financing decisions. It combines mandi prices, weather, satellite vegetation health (NDVI), CPI/WPI, warehouse information, and loan-risk signals into a reproducible Bronze → Silver → Gold data and ML pipeline.

## Objectives

- Build mandi-wise and commodity-wise short-term price forecasts for 7/15/30-day horizons.
- Produce uncertainty-aware forecasts using quantile regression.
- Feed forecasts, uncertainty, agricultural conditions, market information, and loan signals into a risk/LTV model.
- Support decisions such as APPROVE, REJECT, or CONDITIONAL APPROVAL WITH ADJUSTED LTV.
- Eventually expose the results through a decision-support dashboard/API.

## Architecture

```text
External sources
    ├── APMC / Agmarknet
    ├── CPI / WPI
    ├── WDRA
    ├── Weather
    ├── Google Earth Engine / Sentinel-2
    └── Loan-risk proxy
            │
            ▼
AWS S3: agrivault-lake-pawan
    ├── raw/             # Bronze: source/collected data
    ├── standardized/    # Silver: cleaned/typed/deduplicated
    ├── features/        # Gold: model-ready features
    ├── models/          # Trained models + predictions
    └── reference/       # master/reference data
            │
            ▼
Price forecasting → Risk/LTV model → Decision support → Dashboard/API
```

## Repository

```text
D:\agri-vault
├── configs
│   ├── aws_config.yaml
│   └── gee_config.yaml
├── data
│   ├── raw
│   │   ├── apmc
│   │   ├── cpi_wpi
│   │   │   ├── cpi
│   │   │   └── wpi
│   │   ├── wdra
│   │   ├── weather
│   │   ├── ndvi
│   │   └── loans
│   ├── reference
│   │   ├── mandi_locations.csv
│   │   └── apmc
│   │       └── apmc_data_dictionary.xlsx
│   ├── standardized
│   │   ├── apmc
│   │   ├── cpi
│   │   ├── wpi
│   │   ├── wdra
│   │   ├── weather
│   │   ├── ndvi
│   │   └── loans
│   └── features
│       ├── price_features
│       └── risk_features
├── docs
│   ├── srs.md
│   └── phase2_plan.md
├── notebooks
│   ├── 02_eda_ndvi.ipynb
│   └── 03_quantile_forecast_baseline.ipynb
├── scripts
│   ├── gee_fetch_ndvi.py
│   ├── s3_upload_raw.py
│   └── s3_upload_standardized.py
├── src
│   ├── api/                    # Flask dashboard + API
│   │   ├── app.py
│   │   └── templates/
│   ├── features/               # Gold feature engineering
│   │   ├── build_price_features.py
│   │   └── build_risk_features.py
│   ├── models/                 # ML models
│   │   ├── risk_ltv_model.py
│   │   └── quantile_gbm/       # Custom from-scratch GBM
│   ├── standardization/        # PySpark Silver cleaning
│   │   ├── spark_session.py
│   │   ├── clean_apmc.py
│   │   ├── clean_weather.py
│   │   ├── clean_ndvi.py
│   │   ├── clean_wdra.py
│   │   ├── clean_wpi_cpi.py
│   │   └── clean_loans.py
│   ├── reference/
│   └── ingestion/
└── tests
    ├── test_ndvi_join.py
    ├── test_price_features.py
    ├── test_quantile_gbm.py
    └── test_risk_ltv.py
```

## Data

### APMC

Local raw data:

```text
data/raw/apmc/
├── apmc_market_prices.csv
└── apmc_market_prices_source.zip
```

The APMC CSV is about 1 GB and contains more than 5.4 million rows. It currently covers 2025.

Important columns:

```text
id, report_date, state_name, state_code,
district_name, district_code, market_center,
market_code, latitude, longitude,
commodity_type, commodity, variety, origin,
arrivals_tonnes, arrivals_unit,
min_price, max_price, modal_price, price_unit
```

The dataset already contains latitude/longitude, so external geocoding was not required.

### CPI / WPI

```text
data/raw/cpi_wpi/
├── cpi/cpi_monthly.xlsx
└── wpi/wpi_monthly.xlsx
```

Used as macroeconomic context.

### WDRA

```text
data/raw/wdra/
└── state-wise WDRA CSV files
```

Used for warehouse-related risk features.

### Weather

```text
data/raw/weather/weather_daily.parquet
```

Used for agricultural and price features.

### Loans

```text
data/raw/loans/loan_risk_proxy.csv
```

Planned input to the risk/LTV layer.

## Mandi master

The APMC data contains 3,157 unique `market_code` values, but one code was reused:

```text
1765 → Dharashiv
1765 → Murum
```

Therefore `market_code` alone is not a valid physical-mandi identifier.

After deduplication using market code + mandi + district + state + coordinates, we found:

```text
3,158 unique physical mandi/location records
```

The reference file is:

```text
data/reference/mandi_locations.csv
```

Schema:

```text
mandi_id
market_code
mandi_name
district
state
latitude
longitude
```

`mandi_id` is generated from state + district + market code + market center.

Generator:

```text
src/reference/build_mandi_locations.py
```

Run:

```powershell
py src\reference\build_mandi_locations.py
```

## AWS S3 lake

Bucket:

```text
s3://agrivault-lake-pawan/
```

Target structure:

```text
raw/
├── apmc/
├── cpi_wpi/cpi/
├── cpi_wpi/wpi/
├── wdra/
├── weather/
├── ndvi/
└── loans/

standardized/
├── apmc/
├── cpi/
├── wpi/
├── wdra/
├── weather/
├── ndvi/
└── loans/

features/
├── price_features/
└── risk_features/

models/
├── qgbm_*_*.pkl              # Trained quantile GBM models
├── qgbm_*_predictions.parquet  # Model predictions
└── risk_ltv_model.pkl         # Risk/LTV scoring model

reference/
├── mandi_locations.csv
└── apmc_data_dictionary.xlsx
```

AWS CLI was verified as version 2.35.5. AWS region is `ap-south-1`.

Raw APMC, CPI, WPI, WDRA, weather, and reference dictionary data have been uploaded and verified. NDVI is being generated; standardized/features layers are planned downstream.

Empty S3 prefixes initially use `.keep` objects because S3 does not have real directories.

## Google Earth Engine setup

Earth Engine Python API:

```powershell
py -m pip install earthengine-api
```

Authentication:

```powershell
earthengine authenticate
```

Google Cloud / Earth Engine project:

```text
silver-aurora-416511
```

Configured with:

```powershell
earthengine set_project silver-aurora-416511
```

Python initialization was verified:

```powershell
py -c "import ee; ee.Initialize(project='silver-aurora-416511'); print('GEE initialized successfully')"
```

The Earth Engine CLI `ls` message about a missing assets folder is not an authentication failure; no project asset folder had been created.

## NDVI pipeline

Source:

```text
COPERNICUS/S2_SR_HARMONIZED
```

NDVI:

```text
NDVI = (B8 - B4) / (B8 + B4)
```

where B8 is NIR and B4 is Red.

Current configuration:

```yaml
project_id: "silver-aurora-416511"
dataset: "COPERNICUS/S2_SR_HARMONIZED"
start_date: "2025-01-01"
end_date: "2026-01-01"
cloud_percentage: 30
buffer_meters: 500
scale_meters: 10
output_dir: "data/raw/ndvi"
```

Because APMC currently covers 2025 only, NDVI is aligned to calendar year 2025.

### Cloud masking

Sentinel-2 SCL classes masked:

```text
3  cloud shadow
8  medium-probability cloud
9  high-probability cloud
10 cirrus
11 snow/ice
```

Scene-level cloud threshold is 30%.

### Extraction design

The first implementation queried individual images per mandi and was successfully tested on Rayadurg. It returned five January 2025 observations with NDVI values around 0.20–0.25.

Scaling that approach failed with:

```text
EEException: User memory limit exceeded
```

The problem was that the collection was too broad and the workflow performed too many individual operations.

The pipeline was redesigned to:

```text
Sentinel-2 scenes
    ↓
spatial filtering
    ↓
SCL cloud masking
    ↓
monthly median NDVI composite
    ↓
reduceRegions over mandi buffers
    ↓
mandi × month NDVI
```

Mandis are processed in batches to control Earth Engine memory.

A 50-mandi test with batch size 25 completed successfully:

```text
Mandis:       50
Months:       12
Observations: 496
NDVI range:   0.1358 → 0.7059
Mean NDVI:    0.3035
```

Some mandi-month combinations had no valid observation after cloud masking. These should remain missing in raw data and be handled explicitly during later feature engineering.

Production output:

```text
data/raw/ndvi/ndvi_sentinel2_2025.csv
```

Schema:

```text
mandi_id
market_code
mandi_name
district
state
latitude
longitude
date
ndvi
```

Production command:

```powershell
py scripts\gee_fetch_ndvi.py --batch-size 25
```

The 50-mandi test output must not be treated as production data.

## Code organization

### Reference

```text
src/reference/build_mandi_locations.py
```

APMC → physical mandi master.

### Ingestion

```text
src/ingestion/fetch_ndvi.py
```

GEE/Sentinel-2 → raw NDVI.

Execution wrapper:

```text
scripts/gee_fetch_ndvi.py
```

### Cleaning

```text
src/cleaning/clean_ndvi.py
```

Planned NDVI Silver-layer cleaning.

### Standardization

```text
src/standardization/
```

Planned source-specific standardization.

### API / Dashboard

```text
src/api/
├── __init__.py
├── app.py                  # Flask app with API + dashboard
└── templates/
    ├── dashboard.html      # Main dashboard (KPIs, tables, charts)
    ├── commodity.html      # Commodity detail page
    └── 404.html            # Not found page
```

Flask application serving:
- HTML dashboard with KPIs, decision distribution, commodity rankings
- JSON API for risk scores, summaries, and decision data

### Features

```text
src/features/
├── build_price_features.py    # Gold price features (9 feature groups)
├── build_risk_features.py     # Gold risk features (7 feature groups)
└── __init__.py
```

Gold-layer feature construction — reads Silver from S3, writes Gold to S3.

### Models

```text
src/models/
├── quantile_forecast_model.py       # (legacy placeholder)
├── risk_ltv_model.py                # Risk/LTV scoring model
└── quantile_gbm/                    # Custom from-scratch implementation
    ├── __init__.py
    ├── loss.py                       # Pinball loss + gradient/hessian
    ├── tree.py                       # CART tree for quantile regression
    ├── gradient_boosted_trees.py     # QuantileGBM ensemble
    ├── hypertuner.py                 # Walk-forward CV + random search
    └── train.py                      # Entry point, per-commodity training
```

**Quantile GBM:** Custom gradient-boosted quantile regression — no sklearn, no LightGBM.
Trains one GBM per quantile (q10, q50, q90) with walk-forward CV.

**Risk/LTV Model:** Rules-based scoring combining price volatility, forecast uncertainty,
warehouse coverage, commodity perishability, and season risk.
Outputs: APPROVE / CONDITIONAL / REJECT with recommended LTV (40–75%).

## Feature engineering (implemented)

The price feature table joins at:

```text
mandi × commodity × date
```

using:

```text
APMC + Weather + NDVI + CPI + WPI
```

Implemented feature groups:

- **Price lags:** 1d, 7d, 14d, 30d
- **Rolling stats:** 7d/14d/30d mean and std-dev
- **Price momentum:** 7d percentage change
- **Arrivals:** raw + 7d rolling mean
- **Weather:** 7d rolling temp, precipitation, humidity
- **NDVI:** daily forward-filled + 30d delta
- **Macro:** food CPI index + food WPI index (monthly)
- **Temporal:** day-of-week, day-of-month, month, is_weekend
- **Targets:** forward-looking prices at 7d, 15d, 30d

## Forecasting (implemented)

Model:

```text
Custom Quantile GBM (from scratch, no sklearn/LightGBM)
```

Horizons:

```text
7 days
15 days
30 days
```

The goal is to provide both a point forecast and an uncertainty band.

Notebook:

```text
notebooks/03_quantile_forecast_baseline.ipynb
```

## Risk / LTV model (implemented)

```text
src/models/risk_ltv_model.py
```

Combines 5 risk signals:

| Signal | Weight | Source |
|---|---|---|
| Price volatility (CV) | 30% | Risk features |
| Forecast uncertainty | 25% | Quantile model predictions (q90–q10) |
| Warehouse coverage | 15% | WDRA data |
| Commodity perishability | 20% | Hardcoded tier map |
| Season risk | 10% | Kharif / Rabi / Zaid |

Decision outputs:

```text
APPROVE       — risk < 0.35, LTV 65–75%
CONDITIONAL   — risk 0.35–0.60, LTV 50–65%
REJECT        — risk > 0.60, LTV 40–50%
```

## Dashboard / API (implemented)

```text
src/api/app.py
```

Flask application serving risk scores and lending decisions.

### HTML Pages
- `GET /` — Dashboard overview (KPIs, decision distribution, commodity rankings, state summary)
- `GET /commodity/<name>` — Commodity detail (state breakdown, top districts)

### JSON API
- `GET /api/summary` — Overall statistics
- `GET /api/commodities` — All commodities with risk stats
- `GET /api/scores?commodity=WHEAT&decision=APPROVE` — Filterable risk scores
- `GET /api/scores/<commodity>` — Scores for one commodity
- `GET /api/decisions` — Decision distribution by commodity

### Run

```powershell
# Start dashboard
PYTHONPATH=. python -m src.api.app

# Open in browser
# http://127.0.0.1:5000
```

## Engineering decisions

1. **Do not use market_code alone as mandi ID.**
2. **Process the ~1 GB APMC file in chunks rather than loading it entirely into RAM.**
3. **Do not calculate NDVI for every APMC price row.** NDVI is mandi-level/geospatial information.
4. **Keep raw NDVI missing values missing.** Do not fabricate values in Bronze.
5. **Do not blindly query the global Sentinel-2 collection.** Spatial filtering and monthly composites are required.
6. **Batch GEE spatial reductions** to avoid Earth Engine memory limits.
7. **Keep acquisition, cleaning, feature engineering, and modeling separated** in the repository.

## Useful commands

```powershell
# Repository
tree /F
git status

# AWS
aws sts get-caller-identity
aws configure get region
aws s3 ls "s3://agrivault-lake-pawan/" --recursive

# GEE
earthengine set_project silver-aurora-416511
py -c "import ee; ee.Initialize(project='silver-aurora-416511'); print('GEE initialized successfully')"

# Mandi master
py src\reference\build_mandi_locations.py

# NDVI extraction
py scripts\gee_fetch_ndvi.py --batch-size 25

# Silver standardization (PySpark)
.\scripts\run_spark.ps1 src.standardization.clean_apmc
.\scripts\run_spark.ps1 src.standardization.clean_weather
.\scripts\run_spark.ps1 src.standardization.clean_ndvi
.\scripts\run_spark.ps1 src.standardization.clean_wdra
.\scripts\run_spark.ps1 src.standardization.clean_wpi_cpi
.\scripts\run_spark.ps1 src.standardization.clean_loans

# Gold features (pandas/PyArrow → S3)
PYTHONPATH=. python -m src.features.build_price_features
PYTHONPATH=. python -m src.features.build_risk_features

# Train models
PYTHONPATH=. python -m src.models.quantile_gbm.train --commodity WHEAT
PYTHONPATH=. python -m src.models.quantile_gbm.train --top-n 5 --quick
PYTHONPATH=. python -m src.models.risk_ltv_model

# Dashboard
PYTHONPATH=. python -m src.api.app

# Tests
PYTHONPATH=. python -m pytest tests/ -v
```

## Current status

| Component | Status |
|---|---|
| **Data Pipeline** | |
| Raw data uploaded (APMC, CPI, WPI, WDRA, Weather, NDVI, Loans) | ✅ |
| 3,158 mandi master generated | ✅ |
| Silver standardization (all 6 datasets, PySpark → S3) | ✅ |
| Gold price features (5.6M rows × 42 cols → S3) | ✅ |
| Gold risk features (19K rows × 15 cols → S3) | ✅ |
| **Models** | |
| Custom Quantile GBM (from scratch, no sklearn) | ✅ |
| Walk-forward CV hypertuner | ✅ |
| 15 models trained (5 commodities × 3 horizons) | ✅ |
| Risk/LTV scoring model (19K scored, 3 decisions) | ✅ |
| **Interface** | |
| Flask dashboard + JSON API | ✅ |
| NDVI EDA notebook | ✅ |
| Forecast baseline notebook | ✅ |
| **Validation** | |
| Unit tests (66 passing) | ✅ |

## Pipeline status

```text
✅ NDVI extraction + S3 upload
✅ Silver standardization (all 6 datasets)
✅ Gold price features (5.6M rows)
✅ Gold risk features (19K rows)
✅ Custom Quantile GBM (from scratch)
✅ Model training (5 commodities, 15 models)
✅ Risk/LTV model (19K scored)
✅ Dashboard + API (Flask)
✅ Unit tests (66 passing)
✅ Notebooks (EDA + forecast baseline)
```

## Project philosophy

AgriVault is being built as a reproducible data and ML system rather than a collection of one-off notebooks:

```text
Source
  ↓
Bronze / Raw
  ↓
Silver / Standardized
  ↓
Gold / Features
  ↓
Forecasting
  ↓
Risk / LTV
  ↓
Decision support
```

External acquisition is scriptable, transformations belong in `src/`, notebooks are for exploration/experimentation, raw data is preserved, and model-ready datasets are separated from source data.

---

**Project:** AgriVault  
**Data architecture:** Bronze → Silver → Gold → Models → Dashboard  
**Primary lake:** AWS S3  
**Satellite source:** Google Earth Engine / Sentinel-2  
**Forecasting:** Custom Quantile GBM (from scratch, no sklearn)  
**Risk/LTV:** Rules-based scoring → APPROVE / CONDITIONAL / REJECT  
**Interface:** Flask dashboard + JSON API
