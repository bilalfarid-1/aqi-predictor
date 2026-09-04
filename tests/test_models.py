"""
Unit tests for model training and forecasting.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import create_features
from src.models.train import MultiHorizonForecaster


def test_forecaster_train_and_predict():
    times = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(250)]
    df = pd.DataFrame({
        "time": times,
        "city": "Lahore",
        "us_aqi": np.random.uniform(50, 200, size=250),
        "pm2_5": np.random.uniform(20, 100, size=250),
        "temperature_2m": np.random.uniform(15, 35, size=250),
        "wind_speed_10m": np.random.uniform(2, 15, size=250)
    })

    train_data = create_features(df, is_training=True)
    forecaster = MultiHorizonForecaster(model_type="ridge")
    metrics = forecaster.train(train_data)

    assert 24 in metrics
    assert "rmse" in metrics[24]

    # Test inference
    test_row = create_features(df.iloc[-100:], is_training=False).iloc[[-1]]
    preds = forecaster.predict(test_row)
    assert 24 in preds
    assert 0 <= preds[24][0] <= 500
