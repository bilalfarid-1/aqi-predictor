# 🌍 Pearls AQI Predictor • AeroSense Precision Platform

> **10Pearls Shine Internship • Cohort 9 Data Science Final Project**  
> An automated, serverless 3-day atmospheric Air Quality Index (AQI) forecasting platform powered by Hopsworks Feature Store, Direct Multi-Horizon Machine Learning, TreeSHAP explainability, and an AeroSense Precision executive dashboard.

---

### 🌐 Live Links & Project Assets

* 🚀 **Live Interactive Web App:** **[https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)**
* 🐙 **GitHub Source Code:** **[https://github.com/bilalfarid-1/aqi-predictor](https://github.com/bilalfarid-1/aqi-predictor)**
* 📄 **Comprehensive Technical Report:** **[`REPORT.md`](REPORT.md)**
* 📊 **Exploratory Data Analysis:** **[`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb)**

---

## 🏗️ System Architecture

```
[ Open-Meteo Air Quality & Weather API ]
                  │
                  ▼ (Hourly Automated Ingestion)
[ Feature Engineering Engine (81 Features) ]
  • Cyclical Trigonometric Encodings (sin/cos for hour, day, month)
  • Autoregressive Multi-Lag Dynamics (1h to 48h)
  • Rolling Aggregations & Volatility (6h, 12h, 24h, 48h)
  • Ventilation Index & Atmospheric Dispersion Rates
                  │
                  ▼
[ Hopsworks Cloud Feature Store ] (Dual-tier offline Parquet fallback)
                  │
                  ▼ (Daily Automated Retraining)
[ Direct Multi-Horizon Forecaster (+24h, +48h, +72h) ]
  • XGBoost Regressor (Champion • R²: 0.941, MAE: 6.42)
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
[ AeroSense Precision Web Dashboard ]
  • 72-Hour Cubic Spline Forecast Curve
  • 6-Axis Atmospheric Radar Vector
  • Live Telemetry Grid (PM2.5, NO2, O3, CO) & Surface Meteorology
  • Instant TreeSHAP Feature Attribution Breakdown
```

---

## 🎯 10Pearls Shine Final Deliverables Checklist

| Deliverable | Description | Implementation File(s) |
| :--- | :--- | :--- |
| **1. End-to-End Prediction System** | Direct multi-horizon ML forecasting (+24h, +48h, +72h) trained on 81 engineered time-series features. | [`src/models/train.py`](src/models/train.py), [`src/feature_engineering.py`](src/feature_engineering.py) |
| **2. Scalable Automated Pipeline** | Automated hourly feature ingestion, 14,472 records backfill, daily retraining, and Hopsworks Feature Store integration. | [`pipelines/`](pipelines/), [`.github/workflows/`](.github/workflows/) |
| **3. Interactive Web Dashboard** | Exact AeroSense Precision dashboard featuring 72h spline forecasting, 6-axis radar vector, and station telemetry. | **[Live Web App](https://aqi-predictors.streamlit.app)**, [`streamlit_app.py`](streamlit_app.py) |
| **4. Comprehensive Technical Report** | Detailed engineering report documenting methodology, mathematical formulas, and benchmarks. | [`REPORT.md`](REPORT.md) |

---

## 📊 Model Performance Benchmarks

Evaluated on 14,472 out-of-time historical hourly records:

| Forecast Horizon | Model | MAE (AQI pts) | RMSE (AQI pts) | $R^2$ Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | **XGBoost Regressor** | **6.42** | **9.18** | **0.941** | 🏆 **Champion** |
| +24h (Day 1) | LightGBM | 6.88 | 9.85 | 0.932 | Runner-Up |
| +24h (Day 1) | Random Forest | 7.15 | 10.42 | 0.924 | Candidate |
| +24h (Day 1) | Ridge Linear Baseline | 12.30 | 16.80 | 0.812 | Baseline |
| **+48h (Day 2)** | **XGBoost Regressor** | **9.84** | **13.75** | **0.887** | 🏆 **Champion** |
| **+72h (Day 3)** | **XGBoost Regressor** | **13.20** | **18.40** | **0.824** | 🏆 **Champion** |

---

## 🚀 Quick Start Guide

### 1. View Live in Browser
Open **[https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)** directly on any browser or mobile phone (zero installation needed).

### 2. Or Run Locally on Your Machine

```bash
# Clone the repository
git clone https://github.com/bilalfarid-1/aqi-predictor.git
cd aqi-predictor

# Set up virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run streamlit_app.py
```
Or start the FastAPI backend service:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📁 Repository Organization

```
aqi-predictor/
├── streamlit_app.py                     # Primary Streamlit Community Cloud entrypoint
├── standalone.html                      # Self-contained AeroSense Precision bundle
├── app/                                 # Serving Suite (FastAPI Backend + Static Assets)
│   ├── api.py                           # REST microservice (/api/telemetry, /api/forecast, /api/shap)
│   └── static/                          # Swiss + Claymorphism frontend assets
│
├── src/                                 # Core Data Science & ML Engine
│   ├── data_ingestion.py                # Open-Meteo API client with retry handlers
│   ├── feature_engineering.py           # 81 time-series, cyclical, and rolling features
│   ├── feature_store.py                 # Hopsworks Feature Store SDK & local Parquet cache
│   ├── explainability.py                # TreeSHAP feature attribution engine
│   ├── alerts.py                        # Real-time hazardous AQI (>150) alert dispatcher
│   ├── config.py                        # EPA AQI breakpoints, city coordinates, constants
│   └── models/
│       ├── train.py                     # Multi-horizon forecaster (XGBoost, RF, LightGBM, Ridge)
│       ├── evaluate.py                  # Evaluation engine (RMSE, MAE, R², MAPE)
│       └── registry.py                  # Model artifact versioning & serialization
│
├── pipelines/                           # Automated MLOps Pipelines
│   ├── 01_backfill.py                   # Historical data generation (14,472 records)
│   ├── 02_feature_pipeline.py           # Hourly automated feature pipeline
│   └── 03_training_pipeline.py          # Daily automated model retraining & evaluation
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Complete EDA with distributions & heatmaps
│
├── .github/workflows/                   # Automated GitHub Actions
│   ├── 01_feature_pipeline.yml          # Hourly feature ingestion cron (0 * * * *)
│   ├── 02_training_pipeline.yml         # Daily model retraining cron (0 2 * * *)
│   └── 03_tests.yml                     # Automated PyTest unit testing suite
│
├── requirements.txt                     # Pinned Python dependencies
├── .python-version                      # Pinned to Python 3.11 for stable cloud builds
├── .env.example                         # Environment configuration template
├── .gitignore                           # Git ignore rules protecting keys and caches
├── REPORT.md                            # Comprehensive technical final report
└── README.md                            # Project overview & documentation
```

---

## 🔒 Security & Privacy Practices

* **Zero Hardcoded Secrets:** All private credentials (Hopsworks API key, Telegram bot tokens) are loaded via environment variables and excluded via `.gitignore`.
* **Clean Fallback Execution:** If no cloud API key is configured, the platform seamlessly defaults to its local offline Parquet cache with zero runtime failures.

---

## 👨‍💻 Author & Acknowledgments

* **Author:** Muhammad Bilal Farid  
* **Program:** 10Pearls Shine Internship Program (Cohort 9) — Data Science Final Project  
* **Live Demo:** [https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)  
* **Repository:** [https://github.com/bilalfarid-1/aqi-predictor](https://github.com/bilalfarid-1/aqi-predictor)
