# Ventilation index
"""
Feature Engineering Engine for AQI Prediction:
- Cyclical temporal transforms (hour_sin, hour_cos, day_of_week, month)
- Multi-scale lag features (1h, 3h, 6h, 12h, 24h, 48h)
- Rolling aggregations (mean, std, min, max, EMA)
- Meteorological interaction indices (Ventilation Index, AQI Delta Rates)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Transforms raw air quality & meteorological readings into machine learning features.
    """
    if df.empty:
        return df

    data = df.copy()
    data = data.sort_values(by="time").reset_index(drop=True)

    # 1. Temporal & Cyclical Features
    data["hour"] = data["time"].dt.hour
    data["dayofweek"] = data["time"].dt.dayofweek
    data["day"] = data["time"].dt.day
    data["month"] = data["time"].dt.month
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Cyclical Sine/Cosine Encodings
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["day_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # 2. Fill Missing Values smoothly (forward fill + spline/linear interpolation)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].interpolate(method="linear", limit_direction="both")
    data[numeric_cols] = data[numeric_cols].ffill().bfill()

    # 3. Base Target Column
    if "us_aqi" not in data.columns and "pm2_5" in data.columns:
        from src.data_ingestion import calculate_epa_aqi_from_pm25
        data["us_aqi"] = calculate_epa_aqi_from_pm25(data["pm2_5"])

    # 4. Lag Features for Target & Key Pollutants
    lag_steps = [1, 3, 6, 12, 24, 48]
    for lag in lag_steps:
        data[f"aqi_lag_{lag}h"] = data["us_aqi"].shift(lag)
        if "pm2_5" in data.columns:
            data[f"pm25_lag_{lag}h"] = data["pm2_5"].shift(lag)
        if "temperature_2m" in data.columns:
            data[f"temp_lag_{lag}h"] = data["temperature_2m"].shift(lag)
        if "wind_speed_10m" in data.columns:
            data[f"wind_lag_{lag}h"] = data["wind_speed_10m"].shift(lag)

    # 5. Rolling Window Aggregations
    windows = [6, 12, 24, 48]
    for w in windows:
        data[f"aqi_roll_mean_{w}h"] = data["us_aqi"].rolling(window=w, min_periods=1).mean()
        data[f"aqi_roll_std_{w}h"] = data["us_aqi"].rolling(window=w, min_periods=1).std().fillna(0)
        data[f"aqi_roll_max_{w}h"] = data["us_aqi"].rolling(window=w, min_periods=1).max()
        data[f"aqi_roll_min_{w}h"] = data["us_aqi"].rolling(window=w, min_periods=1).min()

    # Exponential Moving Average (EMA)
    data["aqi_ema_12h"] = data["us_aqi"].ewm(span=12, adjust=False).mean()
    data["aqi_ema_24h"] = data["us_aqi"].ewm(span=24, adjust=False).mean()

    # 6. Physical & Meteorological Interaction Features
    if "wind_speed_10m" in data.columns and "temperature_2m" in data.columns:
        # Ventilation Proxy (stagnant air vs dispersing winds)
        data["ventilation_index"] = data["wind_speed_10m"] / (data["temperature_2m"].abs() + 1.0)
    else:
        data["ventilation_index"] = 0.0

    # Rate of Change (Delta AQI / Delta t)
    data["aqi_change_rate_1h"] = data["us_aqi"] - data["aqi_lag_1h"]
    data["aqi_change_rate_6h"] = (data["us_aqi"] - data["aqi_lag_6h"]) / 6.0
    data["aqi_change_rate_24h"] = (data["us_aqi"] - data["aqi_lag_24h"]) / 24.0

    # Particulate Ratio (PM2.5 / PM10)
    if "pm2_5" in data.columns and "pm10" in data.columns:
        data["pm_ratio"] = (data["pm2_5"] / (data["pm10"] + 1e-5)).clip(0, 1.5)
    else:
        data["pm_ratio"] = 0.5

    # 7. Targets for Multi-Horizon Forecasting (24h, 48h, 72h forward)
    if is_training:
        data["target_aqi_24h"] = data["us_aqi"].shift(-24)
        data["target_aqi_48h"] = data["us_aqi"].shift(-48)
        data["target_aqi_72h"] = data["us_aqi"].shift(-72)
        # Drop rows where future targets cannot be determined or starting lags are NaN
        data = data.dropna(subset=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h", "aqi_lag_48h"]).reset_index(drop=True)
    else:
        # For live inference, forward fill initial lag NaNs
        data = data.bfill().ffill()

    return data


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns list of feature columns excluding timestamps, strings, and targets.
    """
    exclude = {"time", "city", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"}
    return [col for col in df.columns if col not in exclude and np.issubdtype(df[col].dtype, np.number)]
