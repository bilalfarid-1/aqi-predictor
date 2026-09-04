"""
Unit tests for data ingestion clients.
"""
import pytest
import pandas as pd
from src.data_ingestion import fetch_combined_data, calculate_epa_aqi_from_pm25


def test_epa_aqi_calculation():
    pm25_series = pd.Series([10.0, 25.0, 45.0, 100.0, 200.0])
    aqi_series = calculate_epa_aqi_from_pm25(pm25_series)
    
    assert len(aqi_series) == 5
    assert aqi_series.iloc[0] <= 50     # Good
    assert 51 <= aqi_series.iloc[1] <= 100  # Moderate
    assert aqi_series.iloc[4] > 200     # Very Unhealthy


def test_fetch_combined_data_structure():
    df = fetch_combined_data(city="Lahore", past_days=2, forecast_days=1)
    if not df.empty:
        assert "time" in df.columns
        assert "city" in df.columns
        assert "pm2_5" in df.columns or "us_aqi" in df.columns
