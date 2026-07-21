# 🌍 AQI Predictor: Automated Air Quality Forecasting

A **professional, production-ready** machine learning system that predicts Air Quality Index (AQI) for the next 3 days using an end-to-end automated pipeline.

Built during the **10Pearls Shine Internship Program** with a focus on professional ML practices, cloud-native architecture, and automated CI/CD.

---

## 🎯 What This Project Does

This system **automatically**:
1. ✅ **Fetches real-time AQI & weather data** every hour from public APIs (AQICN, OpenWeatherMap)
2. ✅ **Engineers features** and stores them in a cloud Feature Store (Hopsworks)
3. ✅ **Trains ML models** daily to predict AQI for the next 3 days
4. ✅ **Runs pipelines** on a schedule using GitHub Actions (zero manual intervention)
5. ✅ **Displays forecasts** on an interactive Streamlit dashboard with explainability charts

**Zero human intervention needed.** Set it up once, and it runs forever. 🚀

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, NumPy |
| **ML Models** | scikit-learn (Random Forest, Ridge Regression) |
| **Feature Store** | Hopsworks |
| **Model Explainability** | SHAP |
| **Automation** | GitHub Actions |
| **Dashboard** | Streamlit, Plotly |
| **APIs** | AQICN, OpenWeatherMap |
| **Deep Learning** | TensorFlow (optional) |

---

## 📁 Project Structure

```
aqi-predictor/
├── src/                          # Shared utilities & configuration
│   ├── config.py                 # Centralized configuration management
│   ├── logger.py                 # Professional logging setup
│   └── utils.py                  # Helper functions (WIP)
│
├── pipelines/                    # Production ML pipelines
│   ├── feature_pipeline.py        # Fetch data, engineer features, store in Hopsworks
│   ├── backfill.py               # Generate historical training data
│   └── training_pipeline.py       # Train models, evaluate, save to registry
│
├── app/                          # Web dashboard
│   └── dashboard.py              # Streamlit app with predictions & explainability
│
├── notebooks/                    # EDA & exploration (NOT production code)
│   └── eda.ipynb                 # Data exploration & analysis
│
├── tests/                        # Unit tests
│   └── test_feature_pipeline.py  # Example tests
│
├── .github/workflows/            # GitHub Actions CI/CD
│   ├── feature_pipeline.yml      # Runs every hour
│   └── training_pipeline.yml     # Runs every day
│
├── requirements.txt              # All Python dependencies (pinned versions)
├── .env.example                  # Template for environment variables
├── .gitignore                    # Keeps secrets out of GitHub
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone & Set Up

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your actual API keys:
#   - AQICN_API_KEY (from https://aqicn.org/api/)
#   - OPENWEATHER_API_KEY (from https://openweathermap.org/api)
#   - HOPSWORKS_API_KEY (from https://hopsworks.ai)
```

### 3. Run the Feature Pipeline (Fetch Data)

```bash
python -m pipelines.feature_pipeline
```

This fetches real AQI data from your city and stores it in Hopsworks.

### 4. View the Dashboard

```bash
streamlit run app/dashboard.py
```

Open your browser to `http://localhost:8501` to see live predictions.

---

## 📊 Project Roadmap (6 Sprints)

| Sprint | Dates | Status | Focus |
|--------|-------|--------|-------|
| Pre-Sprint | Jul 18–20 | ✅ | Environment setup, API keys |
| **Sprint 1** | Jul 21–27 | 🚧 | Tool familiarization (Python, Pandas, APIs) |
| Sprint 2 | Jul 28–Aug 3 | ⬜ | Feature pipeline development |
| Sprint 3 | Aug 4–10 | ⬜ | Historical backfill + EDA |
| Sprint 4 | Aug 11–17 | ⬜ | ML training pipeline |
| Sprint 5 | Aug 18–24 | ⬜ | CI/CD automation + dashboard |
| Sprint 6 | Aug 25–Sep 2 | ⬜ | Polish, report, submission |

---

## 🧪 Testing

Run unit tests to verify everything works:

```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage report
```

---

## 📚 Key Concepts Explained

### What is a Feature Store?
Instead of just using CSV files, we use **Hopsworks** — a cloud platform that stores ML features. Think of it as a professional database that automatically versions features, prevents data leakage, and serves data to both training and inference.

### Why GitHub Actions?
Our pipelines run automatically on a schedule without any manual intervention:
- Feature pipeline: Every hour (keeps data fresh)
- Training pipeline: Every day (model stays up-to-date)

### What is SHAP?
SHAP gives us explainability — we can show which features (PM2.5, temperature, hour of day) matter most for predictions. This is crucial for building trust in ML systems.

---

## 📈 Performance Metrics

The model is evaluated using:
- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE** — Mean Absolute Error (average prediction error)
- **R²** — How well the model explains variance in data (closer to 1.0 is better)

Current metrics (TBD): To be filled in during Sprint 4.

---

## 🤝 Contributing

This is a learning project, but professional practices are followed:
- Create a branch for each feature: `git checkout -b feature/your-feature`
- Write meaningful commit messages: `git commit -m "feat: add feature X"`
- Test before pushing: `pytest tests/`
- Create a pull request for review

---

## ⚠️ Important Notes

### Never Commit API Keys
- Create a `.env` file from `.env.example`
- Add `.env` to `.gitignore` (already done)
- GitHub will scan for leaked keys — keep this repo clean

### Data Leakage Prevention
- Always split train/test data **before** any preprocessing
- The backfill script handles this automatically

---

## 📞 Support & Questions

For questions or issues:
1. Check the `SETUP.md` file for detailed setup instructions
2. Review the individual pipeline docstrings: `python -m pipelines.feature_pipeline --help`
3. Check GitHub Actions logs for automated pipeline issues

---

## 📄 License

This project is part of the 10Pearls Shine Internship Program.

---

## 🙏 Acknowledgments

- Built as part of the **10Pearls Shine Internship Program**
- Mentor & team guidance throughout the project
- Open-source communities: pandas, scikit-learn, Streamlit, Hopsworks

---

**Last Updated:** July 2026  
**Status:** 🚧 In Development (Sprint 1)
