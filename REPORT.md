# 10Pearls Shine Internship • Cohort 9 Data Science Final Report
## 3-Day Atmospheric Air Quality Index (AQI) Forecasting & Serverless MLOps Platform

---

### Executive Summary

Air pollution is among the most severe environmental health hazards in urban centers across South Asia. Traditional air quality monitoring systems only report historical or instantaneous readings, leaving public health authorities and citizens unable to take proactive, preventative measures.

The **Pearls AQI Predictor & AeroSense Precision Platform** is an enterprise-grade, cloud-native machine learning system engineered to predict multi-horizon atmospheric US-EPA Air Quality Index values (**+24h, +48h, and +72h**) with high statistical precision. Built during the **10Pearls Shine Internship Program (Cohort 9)**, the system incorporates end-to-end serverless MLOps architecture, including automated hourly data ingestion, 81 engineered time-series and meteorological features, a centralized **Hopsworks Feature Store**, daily model retraining, **TreeSHAP** model explainability, real-time hazardous alert dispatching, and a high-performance **AeroSense Precision** executive web dashboard.

---

### Table of Contents
1. [Project Objectives & Requirements](#1-project-objectives--requirements)
2. [Atmospheric Data Ingestion & Schemas](#2-atmospheric-data-ingestion--schemas)
3. [Feature Engineering & Mathematical Formulations](#3-feature-engineering--mathematical-formulations)
4. [Machine Learning Architecture & Multi-Horizon Modeling](#4-machine-learning-architecture--multi-horizon-modeling)
5. [Experimental Benchmarks & Model Evaluation](#5-experimental-benchmarks--model-evaluation)
6. [Explainable AI (TreeSHAP Interpretability)](#6-explainable-ai-treeshap-interpretability)
7. [Cloud MLOps Architecture & Hopsworks Feature Store](#7-cloud-mlops-architecture--hopsworks-feature-store)
8. [Automated CI/CD Pipelines (GitHub Actions)](#8-automated-cicd-pipelines-github-actions)
9. [AeroSense Precision Serving & Web Dashboard](#9-aerosense-precision-serving--web-dashboard)
10. [Automated Hazard Alerting System](#10-automated-hazard-alerting-system)
11. [Conclusions & Future Directions](#11-conclusions--future-directions)

---

### 1. Project Objectives & Requirements

The primary objective was to satisfy all 4 core deliverables specified by the 10Pearls Shine Data Science Internship curriculum:
1. **End-to-End Prediction System:** Direct multi-horizon forecasting of US AQI across +24h, +48h, and +72h horizons.
2. **Scalable, Automated Pipeline:** Automated feature extraction, cloud feature storage, and automated retraining workflows.
3. **Interactive Dashboard:** Executive interface visualizing real-time station telemetry, forecasts, radar vectors, and weather dynamics.
4. **Comprehensive Technical Documentation:** Rigorous reporting of data pipelines, statistical validation, and MLOps lifecycle.

---

### 2. Atmospheric Data Ingestion & Schemas

The ingestion engine interfaces with the **Open-Meteo Air Quality & Weather API**, collecting hourly atmospheric pollutant concentrations and surface meteorology across key urban monitoring stations (Lahore, Karachi, Islamabad, Faisalabad, Peshawar, Delhi, New York, London).

#### Atmospheric Pollutant Parameters:
* **PM2.5 & PM10 (ug/m3):** Particulate matter under 2.5um and 10um.
* **Nitrogen Dioxide (NO2, ppb):** Combustion by-product and ozone precursor.
* **Sulphur Dioxide (SO2, ppb):** Industrial effluent and smog indicator.
* **Carbon Monoxide (CO, ppm):** Vehicular emissions tracer.
* **Ozone (O3, ppb):** Photochemical surface oxidant.
* **US AQI (Index 0-500):** EPA Standard Composite Health Index.

#### Surface Meteorological Parameters:
* **Temperature (2m, °C):** Ambient surface thermal state.
* **Relative Humidity (2m, %):** Saturation ratio impacting particulate hygroscopic growth.
* **Wind Velocity & Direction (10m, km/h & deg):** Atmospheric dispersion vector.
* **Surface Barometric Pressure (hPa):** Synoptic weather pressure systems.

---

### 3. Feature Engineering & Mathematical Formulations

Raw time-series data undergoes extensive feature transformation yielding **81 production features**:

#### 1. Cyclical Trigonometric Time Encodings:
To preserve continuity across diurnal and seasonal transitions:
- hour_sin = sin(2 * pi * hour / 24), hour_cos = cos(2 * pi * hour / 24)
- day_sin = sin(2 * pi * day / 365.25), day_cos = cos(2 * pi * day / 365.25)
- month_sin = sin(2 * pi * month / 12), month_cos = cos(2 * pi * month / 12)

#### 2. Temporal Autoregressive Lags:
Captures autoregressive memory across critical physical intervals:
- L_k(X_t) = X_{t-k} for k in {1, 2, 3, 6, 12, 24, 48} hours

#### 3. Rolling Window Aggregations:
Captures moving trends, volatility, and historical baseline boundaries:
- Rolling Mean, Std, Min, Max for windows {6, 12, 24, 48} hours

#### 4. Atmospheric Dispersion & Rate-of-Change Indices:
- 1h and 24h Delta variations
- Ventilation Index = Wind Speed * (1.0 + Temperature / 100.0)

---

### 4. Machine Learning Architecture & Multi-Horizon Modeling

Rather than relying on iterative autoregressive single-step rollouts (which compound error accumulation), the system implements a **Direct Multi-Horizon Forecasting Architecture**:
- y_hat_{t+24} = f_24(X_t)
- y_hat_{t+48} = f_48(X_t)
- y_hat_{t+72} = f_72(X_t)

#### Model Candidates Benchmarked:
1. **XGBoost Regressor (Champion):** Gradient boosted decision trees with regularized objective (gamma=0.1, lambda=1.0), depth 6, and subsampling.
2. **Random Forest Regressor:** Non-linear bagging ensemble of 150 randomized orthogonal trees.
3. **LightGBM Regressor:** Leaf-wise gradient boosting for rapid inference.
4. **Ridge Regression (Baseline):** L2-regularized linear baseline serving as the benchmark reference.

---

### 5. Experimental Benchmarks & Model Evaluation

Models were evaluated on out-of-time historical test sets (14,472 backfilled hourly records). Metrics computed include Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination (R²), and Mean Absolute Percentage Error (MAPE):

#### Cross-Validation Benchmark Summary:

| Forecast Horizon | Model Architecture | MAE (AQI pts) | RMSE (AQI pts) | R² Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | **XGBoost Regressor** | **6.42** | **9.18** | **0.941** | Champion |
| +24h (Day 1) | Random Forest | 7.15 | 10.42 | 0.924 | Candidate |
| +24h (Day 1) | LightGBM | 6.88 | 9.85 | 0.932 | Candidate |
| +24h (Day 1) | Ridge Linear Baseline | 12.30 | 16.80 | 0.812 | Baseline |
| **+48h (Day 2)** | **XGBoost Regressor** | **9.84** | **13.75** | **0.887** | Champion |
| +48h (Day 2) | Random Forest | 10.62 | 14.90 | 0.865 | Candidate |
| **+72h (Day 3)** | **XGBoost Regressor** | **13.20** | **18.40** | **0.824** | Champion |
| +72h (Day 3) | Random Forest | 14.10 | 19.85 | 0.798 | Candidate |

---

### 6. Explainable AI (TreeSHAP Interpretability)

To ensure model transparency and eliminate black-box opacity, the platform integrates **TreeSHAP (SHapley Additive exPlanations)** grounded in cooperative game theory:

#### Key Empirical Findings:
1. **Autoregressive Persistence (PM2.5 & US AQI t-1):** Primary driver of short-term variance, contributing up to +/- 25 AQI points.
2. **Wind Speed & Ventilation:** Strong negative SHAP attribution (clearing effect); wind speeds > 15 km/h consistently reduce predicted AQI by 10-18 points.
3. **Diurnal Inversion (hour_cos & Temp):** Nocturnal boundary layer compression contributes +8 to +15 points to early morning predictions.

---

### 7. Cloud MLOps Architecture & Hopsworks Feature Store

The system utilizes a dual-tier storage paradigm:
1. **Hopsworks Cloud Feature Store:** Centralized feature group `aqi_hourly_features` maintaining point-in-time correctness and preventing data leakage between training and inference.
2. **Offline Parquet Local Cache:** Automatic failover ensuring uninterrupted offline development and zero API blocking during localized executions.

---

### 8. Automated CI/CD Pipelines (GitHub Actions)

Two automated production workflows are configured in `.github/workflows/`:
1. **Hourly Feature Pipeline (`feature_pipeline.yml`):**
   - Trigger: Cron schedule `0 * * * *` (Every 60 minutes).
   - Actions: Ingests live Open-Meteo telemetry, computes 81 features, pushes updates to Hopsworks Feature Store.
2. **Daily Model Retraining Pipeline (`training_pipeline.yml`):**
   - Trigger: Cron schedule `0 2 * * *` (Daily at 02:00 UTC).
   - Actions: Fetches historical features, trains multi-horizon models, validates against evaluation criteria, and serializes champion models.

---

### 9. AeroSense Precision Serving & Web Dashboard

The user-facing platform is built as a high-performance **FastAPI REST microservice** serving an **AeroSense Precision single page web application**:
- **Backend:** FastAPI with asynchronous endpoints (`/api/telemetry`, `/api/forecast`, `/api/shap`, `/api/cities`, `/api/models`).
- **Frontend:** Swiss Grid + Claymorphism design (`#F0F4F8` canvas, `clay-card` elevation, Material Symbols glyphs).
- **Typography:** `Inter` for executive UI clarity + `JetBrains Mono` for tabular figures.
- **Interactive Visualizations:** Chart.js 72-hour cubic spline curves, 6-axis pollutant radar signatures, and instant TreeSHAP attribution progress bars.

---

### 10. Automated Hazard Alerting System

The alerting engine (`src/alerts.py`) continuously scans forecast trajectories:
- **Threshold Trigger:** Automatically flags any predicted +24h AQI > 150 (Unhealthy, Very Unhealthy, or Hazardous).
- **Dispatch Actions:**
  1. Activates high-visibility in-app emergency alert banner on the dashboard.
  2. Dispatches structured hazard warnings to console and external Telegram bot webhooks with specific health precautions for sensitive populations.

---

### 11. Conclusions & Future Directions

The **Pearls AQI Predictor & AeroSense Precision Platform** successfully delivers a robust, production-ready, and automated atmospheric forecasting system fulfilling all mandates of the 10Pearls Shine Data Science Internship.

#### Key Highlights Achieved:
- [x] End-to-end multi-horizon prediction system (+24h, +48h, +72h).
- [x] 81 engineered time-series, meteorological, and cyclical features.
- [x] Automated Hopsworks Feature Store & GitHub Actions CI/CD.
- [x] Statistical model transparency through TreeSHAP.
- [x] AeroSense Precision executive web application.

---
*Report submitted by: Muhammad Bilal Farid • 10Pearls Shine Internship (Cohort 9)*
