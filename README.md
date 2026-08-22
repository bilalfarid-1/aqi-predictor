# 🌍 Pearls AQI Predictor • AeroSense Precision Platform

> **10Pearls Shine Internship • Cohort 9 Data Science Final Project**  
> An automated, production-grade 3-day atmospheric Air Quality Index (AQI) forecasting system powered by Hopsworks Feature Store, Direct Multi-Horizon Machine Learning, TreeSHAP explainability, and an AeroSense Precision executive dashboard.

---

## 🏗️ System Architecture

```
[ Open-Meteo Air Quality & Weather API ]
                  │
                  ▼ (Hourly Ingestion)
[ Feature Engineering Engine (81 Features) ]
  • Cyclical Time Encodings (sin/cos)
  • Multi-Lag Dynamics (1h to 48h)
  • Rolling Statistics & Volatility (6h-48h)
  • Ventilation & Dispersion Indices
                  │
                  ▼
[ Hopsworks Cloud Feature Store ] (Dual-tier Parquet fallback)
                  │
                  ▼ (Daily Retraining & Versioning)
[ Direct Multi-Horizon Forecaster (+24h, +48h, +72h) ]
  • XGBoost Regressor (Champion V2 • R²: 0.941)
  • Random Forest Regressor
  • LightGBM Regressor
  • Ridge Linear Baseline
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
[ TreeSHAP Explainability ]  [ Hazard Alert Dispatcher (>150 AQI) ]
       │                     │
       └──────────┬──────────┘
                  ▼
[ FastAPI High-Performance Serving Suite ]
                  ▼
[ AeroSense Precision Web Dashboard (http://localhost:8000) ]
```

---

## 🎯 Final Project Deliverables

| Deliverable | Description | Location in Repo |
| :--- | :--- | :--- |
| **1. End-to-End Prediction System** | Direct multi-horizon ML forecasting (+24h, +48h, +72h) with 81 engineered time-series features. | [`src/models/train.py`](src/models/train.py), [`src/feature_engineering.py`](src/feature_engineering.py) |
| **2. Automated Scalable Pipeline** | Hourly feature pipeline, 14,472 records backfill, daily retraining, and Hopsworks Feature Store integration. | [`pipelines/`](pipelines/), [`.github/workflows/`](.github/workflows/) |
| **3. Interactive Web Dashboard** | AeroSense Precision UI with 72h spline forecast, 6-axis pollutant radar, and real-time telemetry. | [`app/`](app/) |
| **4. Comprehensive Technical Report** | Detailed project report covering methodology, mathematical formulas, and benchmarks. | [`REPORT.md`](REPORT.md) |

---

## 📊 Cross-Validation Performance Benchmarks

Models were evaluated on 14,472 backfilled hourly records:

| Horizon | Model | MAE (AQI pts) | RMSE (AQI pts) | R² Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | **XGBoost Regressor** | **6.42** | **9.18** | **0.941** | 🏆 Champion |
| +24h (Day 1) | Random Forest | 7.15 | 10.42 | 0.924 | Candidate |
| +24h (Day 1) | LightGBM | 6.88 | 9.85 | 0.932 | Candidate |
| +24h (Day 1) | Ridge Linear Baseline | 12.30 | 16.80 | 0.812 | Baseline |
| **+48h (Day 2)** | **XGBoost Regressor** | **9.84** | **13.75** | **0.887** | 🏆 Champion |
| **+72h (Day 3)** | **XGBoost Regressor** | **13.20** | **18.40** | **0.824** | 🏆 Champion |

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/bilalfarid-1/aqi-predictor.git
cd aqi-predictor

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials (Hopsworks API key, Telegram bot token if alerting is enabled)
```

### 3. Run Historical Backfill (Optional / Local Cache)

```bash
python pipelines/01_backfill.py
```

### 4. Launch the AeroSense Precision Web Application

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

* **Web Application:** `http://localhost:8000`
* **Interactive API Documentation (Swagger):** `http://localhost:8000/docs`

---

## 📁 Repository Structure

```
aqi-predictor/
├── app/                                 # Production Serving Suite
│   ├── api.py                           # FastAPI REST microservice
│   └── static/                          # AeroSense Precision Frontend
│       ├── index.html                   # Claymorphic single-page app
│       └── app.js                       # Reactive client logic & SHAP prefetch
│
├── src/                                 # Core Data Science & ML Engine
│   ├── data_ingestion.py                # Open-Meteo live API integration
│   ├── feature_engineering.py           # 81 engineered time-series features
│   ├── feature_store.py                 # Hopsworks Feature Store SDK
│   ├── explainability.py                # TreeSHAP attribution engine
│   ├── alerts.py                        # Hazardous AQI (>150) alert dispatcher
│   ├── config.py                        # EPA AQI breakpoints & geolocations
│   └── models/
│       ├── train.py                     # Multi-horizon forecaster (XGBoost, RF, LightGBM, Ridge)
│       ├── evaluate.py                  # Evaluation engine (RMSE, MAE, R², MAPE)
│       └── registry.py                  # Model artifact registry
│
├── pipelines/                           # CI/CD Automated Pipelines
│   ├── 01_backfill.py                   # Historical backfill pipeline
│   ├── 02_feature_pipeline.py           # Hourly automated feature pipeline
│   └── 03_training_pipeline.py          # Daily automated model retraining
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Complete EDA notebook
│
├── .github/workflows/                   # Automated GitHub Actions
│   ├── feature_pipeline.yml             # Runs hourly (cron: 0 * * * *)
│   └── training_pipeline.yml            # Runs daily (cron: 0 2 * * *)
│
├── requirements.txt                     # Pinned project dependencies
├── .env.example                         # Environment configuration template
├── .gitignore                           # Git ignore rules for secrets and caches
├── REPORT.md                            # Comprehensive technical final report
└── README.md                            # Project overview & architecture
```

---

## 🔒 Security & Privacy

* **Zero Secrets Committed:** All sensitive API keys and tokens are loaded via environment variables (`.env`) and excluded via `.gitignore`.
* **Reproducible Builds:** Pinned dependencies ensure consistent runtime across local environments and GitHub Actions runners.

---

## 📄 License & Acknowledgments

This project was built as part of the **10Pearls Shine Internship Program (Cohort 9)**.  
*Author:* **Muhammad Bilal Farid**
