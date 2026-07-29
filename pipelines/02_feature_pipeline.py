"""
Pipeline 02: Hourly Feature Pipeline (Run via GitHub Actions Cron):
Fetches recent air quality and weather telemetry, updates feature groups, and ensures real-time freshness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from src.config import CITIES, DEFAULT_CITY
from src.data_ingestion import fetch_combined_data
from src.feature_engineering import create_features
from src.feature_store import FeatureStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_feature_pipeline():
    logger.info("Executing scheduled hourly feature ingestion pipeline...")
    fs_manager = FeatureStoreManager()

    for city in CITIES.keys():
        logger.info(f"Fetching recent data for {city}...")
        raw_df = fetch_combined_data(city=city, past_days=3, forecast_days=4)
        if raw_df.empty:
            continue

        feature_df = create_features(raw_df, is_training=False)
        fs_manager.save_features(feature_df)

    logger.info("✅ Hourly feature pipeline completed successfully.")


if __name__ == "__main__":
    run_feature_pipeline()
