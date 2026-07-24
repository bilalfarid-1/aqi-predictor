"""
Configuration settings, constants, and endpoints for Pearls AQI Predictor.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR = BASE_DIR / "models_cache"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# API Keys & Secrets
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Supported Global Cities with Coordinates & Timezones
CITIES = {
    "Lahore": {"lat": 31.5204, "lon": 74.3587, "country": "PK", "tz": "Asia/Karachi"},
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "country": "PK", "tz": "Asia/Karachi"},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "country": "PK", "tz": "Asia/Karachi"},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350, "country": "PK", "tz": "Asia/Karachi"},
    "Peshawar": {"lat": 34.0151, "lon": 71.5249, "country": "PK", "tz": "Asia/Karachi"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "country": "IN", "tz": "Asia/Kolkata"},
    "New York": {"lat": 40.7128, "lon": -74.0060, "country": "US", "tz": "America/New_York"},
    "London": {"lat": 51.5074, "lon": -0.1278, "country": "GB", "tz": "Europe/London"},
}

DEFAULT_CITY = "Lahore"

# US EPA AQI Categories & Thresholds
AQI_CATEGORIES = [
    {
        "min": 0,
        "max": 50,
        "label": "Good",
        "color": "#00e400",
        "textColor": "#000000",
        "desc": "Air quality is satisfactory, and air pollution poses little or no risk."
    },
    {
        "min": 51,
        "max": 100,
        "label": "Moderate",
        "color": "#ffff00",
        "textColor": "#000000",
        "desc": "Air quality is acceptable. Sensitive individuals should consider limiting heavy exertion."
    },
    {
        "min": 101,
        "max": 150,
        "label": "Unhealthy for Sensitive Groups",
        "color": "#ff7e00",
        "textColor": "#ffffff",
        "desc": "Members of sensitive groups may experience health effects. General public is less likely to be affected."
    },
    {
        "min": 151,
        "max": 200,
        "label": "Unhealthy",
        "color": "#ff0000",
        "textColor": "#ffffff",
        "desc": "Some members of the general public may experience health effects; sensitive groups may experience more serious effects."
    },
    {
        "min": 201,
        "max": 300,
        "label": "Very Unhealthy",
        "color": "#8f3f97",
        "textColor": "#ffffff",
        "desc": "Health alert: The risk of health effects is increased for everyone."
    },
    {
        "min": 301,
        "max": 1000,
        "label": "Hazardous",
        "color": "#7e0023",
        "textColor": "#ffffff",
        "desc": "Health warning of emergency conditions: everyone is more likely to be affected."
    }
]

def get_aqi_category(aqi_val: float) -> dict:
    """Returns category metadata for a given AQI scalar."""
    if aqi_val is None or aqi_val < 0:
        aqi_val = 0
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_val <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]

# Forecast Horizons in Hours
FORECAST_HOURS = [24, 48, 72]

# Feature Store Details
FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_forecast_view"
FEATURE_VIEW_VERSION = 1
