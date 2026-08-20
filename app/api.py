"""
Pearls AQI Predictor - FastAPI Production Backend & Serving Engine
Exposes high-performance endpoints for real-time telemetry, 72-hour forecasts, and SHAP explainability.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.config import CITIES, DEFAULT_CITY, get_aqi_category, FORECAST_HOURS, AQI_CATEGORIES
from src.data_ingestion import fetch_combined_data
from src.feature_engineering import create_features
from src.models.train import MultiHorizonForecaster
from src.explainability import compute_shap_explanations
from src.alerts import send_hazard_alert

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="Automated 72-hour atmospheric air quality forecasting and explainability microservice.",
    version="2.0.0"
)

# Enable CORS for seamless local and remote consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global forecaster cache
forecaster_cache = {}

def get_or_load_forecaster(model_type: str = "xgboost"):
    if model_type not in forecaster_cache:
        try:
            forecaster = MultiHorizonForecaster.load()
        except Exception:
            forecaster = MultiHorizonForecaster(model_type=model_type)
            # Create a sample baseline fit if not loaded
            sample_df = fetch_combined_data(city=DEFAULT_CITY, past_days=7, forecast_days=4)
            sample_feats = create_features(sample_df, is_training=True)
            forecaster.train(sample_feats)
            forecaster.save()
        forecaster_cache[model_type] = forecaster
    return forecaster_cache[model_type]


@app.get("/api/cities")
def list_cities():
    """Returns available monitoring locations."""
    return {
        "cities": [
            {"name": name, "country": info["country"], "lat": info["lat"], "lon": info["lon"]}
            for name, info in CITIES.items()
        ]
    }


@app.get("/api/models")
def list_models():
    """Returns available inference model engines."""
    return {
        "models": [
            {"id": "xgboost", "label": "XGBoost Regressor (V2 Champion)", "is_default": True},
            {"id": "randomforest", "label": "Random Forest Ensemble", "is_default": False},
            {"id": "lightgbm", "label": "LightGBM Gradient Boost", "is_default": False},
            {"id": "ridge", "label": "Ridge Linear Baseline", "is_default": False},
        ]
    }


@app.get("/api/telemetry/{city}")
def get_telemetry(city: str):
    """Fetches real-time ground telemetry, current AQI, pollutants, and weather vectors."""
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    raw_df = fetch_combined_data(city=city, past_days=7, forecast_days=4)
    if raw_df.empty:
        raise HTTPException(status_code=503, detail="Atmospheric telemetry service unavailable.")

    feature_df = create_features(raw_df, is_training=False)
    latest_row = feature_df.iloc[[-1]]

    current_aqi = float(latest_row["us_aqi"].values[0]) if "us_aqi" in latest_row.columns else 80.0
    cat = get_aqi_category(current_aqi)

    # Status theme mappings
    status_themes = {
        "Good": {"bg": "#ECFDF5", "border": "#A7F3D0", "text": "#065F46", "dot": "#10B981", "band": 0, "aurora": "rgba(16, 185, 129, 0.12)"},
        "Moderate": {"bg": "#FEFCE8", "border": "#FEF08A", "text": "#854D0E", "dot": "#F59E0B", "band": 1, "aurora": "rgba(245, 158, 11, 0.12)"},
        "Unhealthy for Sensitive Groups": {"bg": "#FFF7ED", "border": "#FED7AA", "text": "#9A3412", "dot": "#F97316", "band": 2, "aurora": "rgba(249, 115, 22, 0.12)"},
        "Unhealthy": {"bg": "#FEF2F2", "border": "#FECACA", "text": "#991B1B", "dot": "#EF4444", "band": 3, "aurora": "rgba(239, 68, 68, 0.13)"},
        "Very Unhealthy": {"bg": "#FAF5FF", "border": "#E9D5FF", "text": "#6B21A8", "dot": "#A855F7", "band": 4, "aurora": "rgba(168, 85, 247, 0.14)"},
        "Hazardous": {"bg": "#450A0A", "border": "#7F1D1D", "text": "#FEF2F2", "dot": "#DC2626", "band": 5, "aurora": "rgba(127, 29, 29, 0.16)"},
    }
    theme = status_themes.get(cat["label"], status_themes["Moderate"])

    pollutants = {
        "pm2_5": float(latest_row.get("pm2_5", pd.Series([45])).values[0]),
        "pm10": float(latest_row.get("pm10", pd.Series([80])).values[0]),
        "no2": float(latest_row.get("nitrogen_dioxide", pd.Series([25])).values[0]),
        "so2": float(latest_row.get("sulphur_dioxide", pd.Series([10])).values[0]),
        "co": float(latest_row.get("carbon_monoxide", pd.Series([300])).values[0]) / 10.0,
        "o3": float(latest_row.get("ozone", pd.Series([40])).values[0]),
    }

    weather = {
        "temperature": float(latest_row.get("temperature_2m", pd.Series([28.0])).values[0]),
        "humidity": float(latest_row.get("relative_humidity_2m", pd.Series([55.0])).values[0]),
        "wind_speed": float(latest_row.get("wind_speed_10m", pd.Series([8.5])).values[0]),
        "pressure": float(latest_row.get("surface_pressure", pd.Series([1012.0])).values[0]),
    }

    return {
        "city": city,
        "timestamp": datetime.now().isoformat(),
        "current_aqi": round(current_aqi, 1),
        "category": cat,
        "theme": theme,
        "pollutants": pollutants,
        "weather": weather,
    }


@app.get("/api/forecast/{city}")
def get_forecast(city: str, model: str = Query("xgboost")):
    """Computes 72-hour direct multi-horizon predictions with point variance deltas."""
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    raw_df = fetch_combined_data(city=city, past_days=7, forecast_days=4)
    if raw_df.empty:
        raise HTTPException(status_code=503, detail="Atmospheric telemetry service unavailable.")

    feature_df = create_features(raw_df, is_training=False)
    latest_row = feature_df.iloc[[-1]]
    current_aqi = float(latest_row["us_aqi"].values[0]) if "us_aqi" in latest_row.columns else 80.0

    forecaster = get_or_load_forecaster(model)
    preds = forecaster.predict(latest_row)

    f24 = float(preds.get(24, [current_aqi])[0])
    f48 = float(preds.get(48, [current_aqi])[0])
    f72 = float(preds.get(72, [current_aqi])[0])

    is_hazardous = f24 > 150
    if is_hazardous:
        send_hazard_alert(city, f24, horizon_hours=24)

    return {
        "city": city,
        "model": model,
        "current_aqi": round(current_aqi, 1),
        "forecast": {
            "t0": {"label": "Now (T-0)", "value": round(current_aqi, 1), "delta": 0.0},
            "t24": {"label": "Day 1 (+24h)", "value": round(f24, 1), "delta": round(f24 - current_aqi, 1)},
            "t48": {"label": "Day 2 (+48h)", "value": round(f48, 1), "delta": round(f48 - current_aqi, 1)},
            "t72": {"label": "Day 3 (+72h)", "value": round(f72, 1), "delta": round(f72 - current_aqi, 1)},
        },
        "is_hazardous": is_hazardous,
        "hazard_message": f"Projected +24h AQI ({f24:.0f}) exceeds safe health thresholds in {city}. Sensitive groups and respiratory patients should restrict outdoor activity." if is_hazardous else None,
    }


@app.get("/api/shap/{city}")
def get_shap_explanation(city: str, model: str = Query("xgboost")):
    """Computes TreeSHAP feature attributions for tomorrow's forecast."""
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    raw_df = fetch_combined_data(city=city, past_days=7, forecast_days=4)
    feature_df = create_features(raw_df, is_training=False)
    latest_row = feature_df.iloc[[-1]]

    forecaster = get_or_load_forecaster(model)
    shap_info = compute_shap_explanations(forecaster, latest_row, horizon=24)
    return shap_info or {"top_drivers": []}


@app.get("/health")
def health_check():
    """Healthcheck probe for deployment platforms."""
    return {"status": "online", "version": "2.0.0-prod", "timestamp": datetime.now().isoformat()}


# Static files mount
static_path = Path(__file__).resolve().parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
def serve_index():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Pearls AQI API Active. Visit /docs for Swagger documentation."})
