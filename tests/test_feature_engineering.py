"""
Unit tests for feature engineering transformations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import create_features, get_feature_columns


def test_create_features():
    # Synthetic hourly sample
    times = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(100)]
    df = pd.DataFrame({
        "time": times,
        "city": "Lahore",
        "us_aqi": np.random.uniform(50, 200, size=100),
        "pm2_5": np.random.uniform(20, 100, size=100),
        "temperature_2m": np.random.uniform(15, 35, size=100),
        "wind_speed_10m": np.random.uniform(2, 15, size=100)
    })

    feat_df = create_features(df, is_training=False)
    cols = get_feature_columns(feat_df)

    assert "hour_sin" in feat_df.columns
    assert "hour_cos" in feat_df.columns
    assert "aqi_lag_1h" in feat_df.columns
    assert "aqi_roll_mean_24h" in feat_df.columns
    assert len(cols) > 10
