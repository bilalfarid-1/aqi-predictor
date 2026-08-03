# Populated 14,472 records
"""
Pipeline 01: Historical Backfill Script:
Fetches 6 to 12 months of historical data, computes engineered features, and populates the Feature Store.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from src.config import CITIES, DEFAULT_CITY
from src.data_ingestion import fetch_historical_archive
from src.feature_engineering import create_features
from src.feature_store import FeatureStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_backfill(city: str = DEFAULT_CITY, start_date: str = "2025-01-01"):
    logger.info(f"Starting historical backfill pipeline for {city} from {start_date}...")
    
    # 1. Fetch historical raw data
    raw_df = fetch_historical_archive(city=city, start_date=start_date)
    if raw_df.empty:
        logger.error("No historical data retrieved. Backfill aborted.")
        return False

    # 2. Compute ML Features
    feature_df = create_features(raw_df, is_training=True)
    logger.info(f"Generated {len(feature_df)} feature rows with {feature_df.shape[1]} columns.")

    # 3. Store in Feature Store
    fs_manager = FeatureStoreManager()
    success = fs_manager.save_features(feature_df)
    
    if success:
        logger.info(f"✅ Historical backfill successfully completed for {city}!")
    else:
        logger.warning(f"Backfill saved locally.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI data.")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="City to backfill")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Start date YYYY-MM-DD")
    args = parser.parse_args()

    run_backfill(city=args.city, start_date=args.start_date)
