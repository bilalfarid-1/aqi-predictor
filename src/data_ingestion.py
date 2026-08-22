"""
Data Ingestion Module for Open-Meteo Air Quality & Weather API, with AQICN fallback.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any

from src.config import CITIES, DEFAULT_CITY, AQICN_API_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_air_quality_forecast(city: str = DEFAULT_CITY, past_days: int = 7, forecast_days: int = 4) -> pd.DataFrame:
    """
    Fetches real-time, recent past, and forward forecast air quality parameters from Open-Meteo API.
    """
    city_info = CITIES.get(city, CITIES[DEFAULT_CITY])
    lat, lon = city_info["lat"], city_info["lon"]
    tz = city_info.get("tz", "auto")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index,us_aqi,european_aqi",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": tz
    }

    try:
        resp = requests.get(OPEN_METEO_AQI_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.error(f"Empty hourly air quality response for {city}")
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df["city"] = city
        df["latitude"] = lat
        df["longitude"] = lon
        logger.info(f"Successfully fetched {len(df)} air quality rows for {city} (past {past_days}d + next {forecast_days}d)")
        return df

    except Exception as e:
        logger.error(f"Error fetching air quality forecast for {city}: {e}")
        return pd.DataFrame()


def fetch_weather_forecast(city: str = DEFAULT_CITY, past_days: int = 7, forecast_days: int = 4) -> pd.DataFrame:
    """
    Fetches hourly weather data (temperature, humidity, wind, pressure, precipitation) from Open-Meteo API.
    """
    city_info = CITIES.get(city, CITIES[DEFAULT_CITY])
    lat, lon = city_info["lat"], city_info["lon"]
    tz = city_info.get("tz", "auto")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": tz
    }

    try:
        resp = requests.get(OPEN_METEO_WEATHER_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.error(f"Empty hourly weather response for {city}")
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        logger.info(f"Successfully fetched {len(df)} weather rows for {city}")
        return df

    except Exception as e:
        logger.error(f"Error fetching weather forecast for {city}: {e}")
        return pd.DataFrame()


def fetch_combined_data(city: str = DEFAULT_CITY, past_days: int = 7, forecast_days: int = 4) -> pd.DataFrame:
    """
    Merges air quality and weather telemetry into a synchronized hourly DataFrame.
    """
    aq_df = fetch_air_quality_forecast(city, past_days, forecast_days)
    weather_df = fetch_weather_forecast(city, past_days, forecast_days)

    if aq_df.empty:
        return weather_df
    if weather_df.empty:
        return aq_df

    merged = pd.merge(aq_df, weather_df, on="time", how="inner")
    
    # Calculate EPA-standard AQI if us_aqi is missing/nan
    if "us_aqi" not in merged.columns or merged["us_aqi"].isnull().all():
        merged["us_aqi"] = calculate_epa_aqi_from_pm25(merged.get("pm2_5", pd.Series(dtype=float)))

    return merged


def fetch_historical_archive(city: str = DEFAULT_CITY, start_date: str = "2025-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches long-range historical air quality and weather data for backfilling and model training.
    """
    city_info = CITIES.get(city, CITIES[DEFAULT_CITY])
    lat, lon = city_info["lat"], city_info["lon"]
    tz = city_info.get("tz", "auto")

    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Air quality historical parameters
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index,us_aqi",
        "timezone": tz
    }

    # Weather historical parameters
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
        "timezone": tz
    }

    logger.info(f"Fetching historical archive for {city} from {start_date} to {end_date}...")
    try:
        aq_resp = requests.get(OPEN_METEO_ARCHIVE_AQI_URL, params=aq_params, timeout=30)
        aq_resp.raise_for_status()
        aq_df = pd.DataFrame(aq_resp.json().get("hourly", {}))
        if not aq_df.empty and "time" in aq_df.columns:
            aq_df["time"] = pd.to_datetime(aq_df["time"])
            aq_df["city"] = city
            aq_df["latitude"] = lat
            aq_df["longitude"] = lon

        # Historical weather
        w_resp = requests.get(OPEN_METEO_ARCHIVE_WEATHER_URL, params=weather_params, timeout=30)
        w_resp.raise_for_status()
        w_df = pd.DataFrame(w_resp.json().get("hourly", {}))
        if not w_df.empty and "time" in w_df.columns:
            w_df["time"] = pd.to_datetime(w_df["time"])

        if not aq_df.empty and not w_df.empty:
            merged = pd.merge(aq_df, w_df, on="time", how="inner")
        elif not aq_df.empty:
            merged = aq_df
        else:
            merged = pd.DataFrame()

        logger.info(f"Successfully retrieved {len(merged)} historical rows for {city}")
        return merged

    except Exception as e:
        logger.error(f"Error fetching historical archive: {e}")
        # Fallback to recent window if archive endpoint encounters issue
        logger.info("Falling back to recent 90-day window via forecast endpoint...")
        return fetch_combined_data(city, past_days=90, forecast_days=1)


def calculate_epa_aqi_from_pm25(pm25_series: pd.Series) -> pd.Series:
    """
    Computes standard US-EPA AQI values from PM2.5 concentrations (ug/m3) using official breakpoints.
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    def aqi_calc(c):
        if pd.isna(c) or c < 0:
            return np.nan
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= c <= c_high:
                return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low)
        if c > 500.4:
            return 500
        return 0

    return pm25_series.apply(aqi_calc)
