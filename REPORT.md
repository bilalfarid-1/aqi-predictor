# 10Pearls Shine Internship • Cohort 9 Data Science Final Project
## 3-Day Atmospheric Air Quality Index (AQI) Forecasting & Serverless MLOps Platform

**Author:** Muhammad Bilal Farid  
**Program:** 10Pearls Shine Internship Program (Cohort 9) — Data Science Track  
**Live Application:** [https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)  
**GitHub Repository:** [https://github.com/bilalfarid-1/aqi-predictor](https://github.com/bilalfarid-1/aqi-predictor)  
**Submission Date:** August 2026 (Final Release v2.0.0)

---

### Project Overview & Real-World Motivation

Urban centers across Pakistan and South Asia face severe seasonal smog and hazardous particulate pollution every year. Cities like Lahore, Karachi, and Peshawar regularly breach dangerous air quality thresholds, yet most public platforms only show what the air was like hours ago or at best right now. By the time a school, hospital, or commuter realizes the AQI has crossed 200, the exposure has already happened.

The goal of this internship capstone project was to build a system that looks forward instead of backward: an automated, serverless machine learning platform that predicts ground-level US-EPA Air Quality Index values for the next three days (**+24h, +48h, and +72h**) with clear statistical explanations behind every forecast.

Rather than building a static script or a one-off model notebook, I designed this as an end-to-end MLOps pipeline that continuously pulls live atmospheric data, updates its feature store, trains multi-horizon models, explains prediction drivers via TreeSHAP, and presents the results through a clean, responsive web interface.

---

### Key Deliverables Checklist (10Pearls Shine Requirements)

All four core requirements specified by the 10Pearls Shine internship curriculum have been implemented and verified:

| # | Internship Deliverable | Implementation Details | Verified Location |
| :---: | :--- | :--- | :--- |
| **1** | **End-to-End AQI Prediction System** | High-resolution Open-Meteo ingestion, 81 engineered time-series features, Direct Multi-Horizon Forecasters (+24h, +48h, +72h), and cross-validation benchmarks. | [`src/models/train.py`](src/models/train.py), [`src/feature_engineering.py`](src/feature_engineering.py) |
| **2** | **Scalable, Automated Pipeline** | Centralized Hopsworks Feature Store with offline parquet fallback, 14,472 records historical backfill, and GitHub Actions CI/CD workflows for hourly ingestion and daily retraining. | [`pipelines/`](pipelines/), [`.github/workflows/`](.github/workflows/) |
| **3** | **Interactive Web Dashboard** | Modern AeroSense Precision dashboard featuring 72-hour forecast splines, 6-axis pollutant radar signatures, surface meteorology, and TreeSHAP waterfall tables. | **[Live Web App](https://aqi-predictors.streamlit.app)**, [`streamlit_app.py`](streamlit_app.py), [`app/api.py`](app/api.py) |
| **4** | **Comprehensive Technical Report** | Detailed documentation of pipeline architecture, mathematical formulations, model selection rationale, and production lessons learned. | [`REPORT.md`](REPORT.md), [`README.md`](README.md) |

---

### 1. Data Collection & Atmospheric Ingestion

Air quality is strongly coupled with ground meteorology. Predicting pollution without knowing surface wind speed or temperature inversions leads to poor results. The ingestion engine (`src/data_ingestion.py`) connects directly to the Open-Meteo Air Quality and Weather APIs, capturing hourly readings across eight major urban monitoring centers (Lahore, Karachi, Islamabad, Faisalabad, Peshawar, Delhi, New York, and London).

#### Captured Pollutants:
* **PM2.5 & PM10 ($\mu	ext{g/m}^3$):** Fine and coarse inhalable particulate matter. PM2.5 is the primary health hazard in regional winter smog.
* **Nitrogen Dioxide (NO2, $	ext{ppb}$):** High-temperature combustion effluent (vehicles, thermal power plants).
* **Sulphur Dioxide (SO2, $	ext{ppb}$):** Industrial emissions and heavy oil combustion.
* **Carbon Monoxide (CO, $	ext{ppm}$):** Incomplete vehicular combustion tracer.
* **Ozone (O3, $	ext{ppb}$):** Photochemical ground-level oxidant formed under intense sunlight.
* **US-EPA AQI (0–500 scale):** Standard composite index calculated using official EPA piecewise linear concentration breakpoints.

#### Surface Meteorological Dynamics:
* **Temperature at 2m ($^\circ	ext{C}$):** Governs atmospheric boundary layer thermal structure.
* **Relative Humidity at 2m ($\%$):** Directly impacts particulate hygroscopic growth and atmospheric haze density.
* **Wind Velocity & Direction at 10m ($	ext{km/h}$ & degrees):** Determines lateral advection and pollutant dispersal rates.
* **Surface Barometric Pressure ($	ext{hPa}$):** High-pressure systems create stagnant air masses that trap smog near the ground.

---

### 2. Feature Engineering: Turning Raw Telemetry into Predictive Signals

Raw sensor readings alone are insufficient for multi-day time-series forecasting. I engineered **81 distinct features** (`src/feature_engineering.py`) to give the models temporal awareness and atmospheric physical context:

#### A. Cyclical Time Encodings (Fixing Midnight Discontinuity)
Standard numerical representations of time (e.g., hour $0$ to $23$) create an artificial mathematical cliff between 11:59 PM and 12:00 AM. Using sine and cosine transformations ensures continuity across diurnal cycles and seasonal transitions:
$$	ext{hour}_{\sin} = \sin\left(rac{2\pi \cdot 	ext{hour}}{24}ight), \quad 	ext{hour}_{\cos} = \cos\left(rac{2\pi \cdot 	ext{hour}}{24}ight)$$
$$	ext{month}_{\sin} = \sin\left(rac{2\pi \cdot 	ext{month}}{12}ight), \quad 	ext{month}_{\cos} = \cos\left(rac{2\pi \cdot 	ext{month}}{12}ight)$$

#### B. Autoregressive Temporal Lags
Pollution exhibits strong memory. I extracted historical lag features across physically meaningful intervals:
$$L_k(X_t) = X_{t-k} \quad 	ext{for } k \in \{1, 2, 3, 6, 12, 24, 48\} 	ext{ hours}$$
A 24-hour lag captures yesterday's pollution at the exact same hour, while 1-hour and 2-hour lags provide immediate inertia.

#### C. Multi-Window Rolling Statistics & Volatility
To help the algorithms distinguish brief localized sensor spikes from sustained atmospheric degradation, rolling aggregations were calculated over 6h, 12h, 24h, and 48h windows:
$$\mu_{w}(X_t) = rac{1}{w}\sum_{i=0}^{w-1} X_{t-i}, \quad \sigma_{w}(X_t) = \sqrt{rac{1}{w}\sum_{i=0}^{w-1} (X_{t-i} - \mu_{w})^2}$$
$$\min_{w}(X_t) = \min(X_{t-w+1}, \dots, X_t), \quad \max_{w}(X_t) = \max(X_{t-w+1}, \dots, X_t)$$

#### D. Atmospheric Ventilation & Rate-of-Change Indicators
Stagnant air traps pollutants. I formulated a proxy **Ventilation Index** combining wind speed and surface temperature:
$$	ext{Ventilation Index} = 	ext{Wind Speed}_{10	ext{m}} 	imes \left(1.0 + rac{	ext{Temperature}_{2	ext{m}}}{100.0}ight)$$
$$\Delta 	ext{AQI}_{1	ext{h}} = 	ext{AQI}_t - 	ext{AQI}_{t-1}, \quad \Delta 	ext{AQI}_{24	ext{h}} = 	ext{AQI}_t - 	ext{AQI}_{t-24}$$

---

### 3. Model Architecture & Multi-Horizon Strategy

A major design decision in time-series forecasting is whether to use **Recursive Forecasting** (predict $t+1$, append to data, predict $t+2$, and repeat) or **Direct Forecasting** (train dedicated models for each specific horizon).

In preliminary tests, recursive rollouts accumulated compounding errors rapidly; by hour 72, predictions frequently drifted into unrealistic extremes. I adopted a **Direct Multi-Horizon Forecasting Architecture**:
* **Model 1 ($	heta_{24}$):** Dedicated forecaster for **Day 1 (+24h)**
* **Model 2 ($	heta_{48}$):** Dedicated forecaster for **Day 2 (+48h)**
* **Model 3 ($	heta_{72}$):** Dedicated forecaster for **Day 3 (+72h)**

Each model optimizes its loss function specifically for that forecast distance using currently observed physical conditions.

#### Evaluated Algorithms:
1. **XGBoost Regressor (Champion Model):** Extreme gradient boosted trees with regularized objectives ($\gamma=0.1, \lambda=1.0$), max depth 6, and column subsampling. Best balance between speed, stability, and handling non-linear atmospheric relationships.
2. **Random Forest Regressor:** Bagging ensemble of 150 randomized orthogonal trees; stable against outliers but slower to infer.
3. **LightGBM Regressor:** Leaf-wise tree growth algorithm; rapid training speed and competitive performance.
4. **Ridge Regression (Baseline):** L2-regularized linear model providing an interpretable reference baseline.

---

### 4. Experimental Benchmarks & Evaluation Results

The models were evaluated against an out-of-time test dataset spanning 14,472 backfilled hourly records (2025–2026). Standard evaluation metrics were calculated across all horizons:

$$	ext{RMSE} = \sqrt{rac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}, \quad 	ext{MAE} = rac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|, \quad R^2 = 1 - rac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - ar{y})^2}$$

#### Cross-Validation Summary Table:

| Forecast Horizon | Model Architecture | MAE (AQI pts) | RMSE (AQI pts) | $R^2$ Score | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | **XGBoost Regressor** | **6.42** | **9.18** | **0.941** | 🏆 **Champion** |
| +24h (Day 1) | LightGBM | 6.88 | 9.85 | 0.932 | Runner-Up |
| +24h (Day 1) | Random Forest | 7.15 | 10.42 | 0.924 | Solid Baseline |
| +24h (Day 1) | Ridge Linear Baseline | 12.30 | 16.80 | 0.812 | Linear Baseline |
| **+48h (Day 2)** | **XGBoost Regressor** | **9.84** | **13.75** | **0.887** | 🏆 **Champion** |
| +48h (Day 2) | Random Forest | 10.62 | 14.90 | 0.865 | Solid Baseline |
| **+72h (Day 3)** | **XGBoost Regressor** | **13.20** | **18.40** | **0.824** | 🏆 **Champion** |
| +72h (Day 3) | Random Forest | 14.10 | 19.85 | 0.798 | Solid Baseline |

**Practical Takeaways:**
* XGBoost achieved an $R^2$ of **0.941** on Day 1 with an average error of only **6.42 AQI points**, making the forecast reliable for daily public health guidance.
* Performance gracefully degrades as the horizon expands to Day 3 ($R^2 = 0.824$, $	ext{MAE} = 13.20$), which is consistent with atmospheric chaos dynamics.

---

### 5. Explainable AI: Opening the Black Box with TreeSHAP

Predicting high pollution is useful, but explaining *why* it will occur is critical for public trust. I integrated **TreeSHAP (SHapley Additive exPlanations)** grounded in cooperative game theory (`src/explainability.py`).

Every forecast decomposes into an additive sum of feature attributions:
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
Where $\phi_0$ is the global expected AQI baseline, and $\phi_i(x)$ is the exact numerical impact of feature $i$.

#### Practical Interpretability Insights Observed:
1. **Autoregressive Persistence:** The 1-hour lag and current PM2.5 concentrations consistently accounted for the largest baseline contribution ($\pm 15$ to $25$ points).
2. **The Clearing Effect of Wind:** Wind speeds above $15	ext{ km/h}$ generated negative SHAP values between $-10$ and $-18$ AQI points, directly reflecting physical atmospheric dispersion.
3. **Nocturnal Trapping:** In early morning forecasts, high humidity combined with low temperatures ($	ext{hour}_{\cos}$ near peaks) consistently added $+10$ to $+16$ points, capturing winter ground inversions where emissions cannot disperse upward.

---

### 6. MLOps Architecture & Cloud CI/CD Automation

Building a maintainable machine learning system requires robust data versioning and scheduled orchestration:

```
[ Open-Meteo API ] ──(Hourly Ingestion)──> [ Feature Pipeline (81 Features) ]
                                                        │
                                                        ▼
                                         [ Hopsworks Feature Store ]
                                                        │
                                            (Daily Model Retraining)
                                                        ▼
                                         [ Champion Model Registry ]
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
         [ FastAPI REST Microservice ]                                   [ Streamlit Community Cloud ]
         (Local: http://localhost:8000)                                  (https://aqi-predictors.streamlit.app)
```

1. **Dual-Tier Feature Store:** Integrated the **Hopsworks Feature Store SDK** (`src/feature_store.py`) to prevent train-serve skew. I also implemented an automatic offline Parquet cache layer so that network drops or free-tier quota limits never disrupt local model inference or testing.
2. **Hourly Feature Ingestion (`.github/workflows/01_feature_pipeline.yml`):** Runs on cron `0 * * * *` every 60 minutes via GitHub Actions, pulling live atmospheric readings and publishing updated feature vectors.
3. **Daily Model Retraining (`.github/workflows/02_training_pipeline.yml`):** Runs on cron `0 2 * * *` at 02:00 UTC daily, retraining multi-horizon candidates on rolling window data and validating performance before updating the active model registry.
4. **Hazard Alerting Dispatcher (`src/alerts.py`):** Automatically flags when a +24h forecast exceeds the hazardous threshold ($>150$ AQI) and formats clinical emergency advisories for public channels.

---

### 7. User Interface & Live Deployment

A major design goal was providing zero visual ambiguity and instantaneous insights. Rather than a basic default dashboard, the platform implements the **AeroSense Precision** design system:
* **Visual Identity:** Soft light blue-grey background (`#F0F4F8`) with custom claymorphic elevation shadows (`clay-card`).
* **Typography Hierarchy:** `Inter` for crisp UI labels and navigation, paired with `JetBrains Mono` for tabular data numbers and the prominent `140px` hero AQI metric.
* **Telemetry & Visualizations:** Live 4-card pollutant grid with colored vertical status stripes, a 72-hour cubic spline forecast curve, a 6-axis atmospheric radar vector, and a dedicated TreeSHAP attribution breakdown view.
* **Production Deployment:** Live and accessible 24/7 on Streamlit Community Cloud at **[https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)** with zero installation needed.

---

### 8. Production Challenges & Lessons Learned

1. **Cloud Environment Python Compatibility:**
   * *Problem:* Streamlit Community Cloud recently defaulted to an experimental Python 3.14 environment on new deployments, which broke wheels for packages like `pandas` and `hopsworks`.
   * *Solution:* Explicitly pinned Python 3.11 using a root `.python-version` configuration file and optimized `requirements.txt` for serverless inference, resulting in clean 25-second cloud builds.
2. **Preventing Error Accumulation in Multi-Step Forecasting:**
   * *Problem:* Autoregressive single-step rollouts caused forecast errors to compound by Day 3.
   * *Solution:* Transitioning to a Direct Multi-Horizon approach produced stable, well-bounded predictions for each target day independently.
3. **Client-Side Responsiveness:**
   * *Problem:* Re-fetching and re-computing SHAP values on every dropdown click introduced a 1.5-second lag.
   * *Solution:* Pre-computed regional telemetry payloads in client memory, enabling instant, sub-second station switching without jarring page refreshes.

---

### 9. Conclusion & Next Steps

The **Pearls AQI Predictor & AeroSense Precision Platform** fulfills all requirements of the 10Pearls Shine Internship Program. It demonstrates that reliable, explainable machine learning can be deployed using modern serverless MLOps practices at zero infrastructure cost.

#### Future Planned Enhancements:
* Integrate satellite aerosol optical depth (AOD) feeds from NASA MODIS/Sentinel-5P for broader rural coverage.
* Add multi-city spatial graph neural networks (GNNs) to model cross-border atmospheric transport between adjacent industrial basins.
* Implement mobile SMS and automated push notifications for high-risk respiratory patients.

---
*Report submitted by: Muhammad Bilal Farid • 10Pearls Shine Internship (Cohort 9)*  
*Repository: [https://github.com/bilalfarid-1/aqi-predictor](https://github.com/bilalfarid-1/aqi-predictor)*  
*Live Demo: [https://aqi-predictors.streamlit.app](https://aqi-predictors.streamlit.app)*
