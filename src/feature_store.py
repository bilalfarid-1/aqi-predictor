"""
Hopsworks Feature Store & Offline Cache Interface:
Enables dual-mode operation (Live Hopsworks Cloud & Local Offline Mock Cache)
to guarantee zero downtime and 100% testability.
"""
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    DATA_DIR
)

logger = logging.getLogger(__name__)


class FeatureStoreManager:
    def __init__(self, api_key: Optional[str] = None, project_name: Optional[str] = None):
        self.api_key = api_key or HOPSWORKS_API_KEY
        self.project_name = project_name or HOPSWORKS_PROJECT_NAME
        self.project = None
        self.fs = None
        self._is_connected = False
        self._local_cache_path = DATA_DIR / f"{FEATURE_GROUP_NAME}_cache.parquet"

        self._connect()

    def _connect(self):
        """Attempts connection to Hopsworks Feature Store if API key is present."""
        if not self.api_key or self.api_key.strip() == "":
            logger.warning("No HOPSWORKS_API_KEY found. Operating in Local Cache / Standalone Mode.")
            return

        try:
            import hopsworks
            logger.info(f"Connecting to Hopsworks project: {self.project_name}...")
            self.project = hopsworks.login(api_key_value=self.api_key, project=self.project_name)
            self.fs = self.project.get_feature_store()
            self._is_connected = True
            logger.info("Successfully connected to Hopsworks Feature Store!")
        except Exception as e:
            logger.warning(f"Could not connect to Hopsworks ({e}). Using Local Offline Cache.")
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def save_features(self, df: pd.DataFrame) -> bool:
        """
        Saves engineered features into Hopsworks Feature Group and local cache.
        """
        if df.empty:
            logger.warning("Attempted to save empty DataFrame to Feature Store.")
            return False

        # Always save local cache first
        try:
            df.to_parquet(self._local_cache_path, index=False)
            logger.info(f"Saved {len(df)} feature records to local cache: {self._local_cache_path}")
        except Exception as e:
            logger.error(f"Error saving local parquet cache: {e}")

        # If connected to Hopsworks, push to Feature Group
        if self._is_connected and self.fs:
            try:
                # Ensure timestamp formatted
                upload_df = df.copy()
                if "time" in upload_df.columns:
                    upload_df["time"] = pd.to_datetime(upload_df["time"])

                feature_group = self.fs.get_or_create_feature_group(
                    name=FEATURE_GROUP_NAME,
                    version=FEATURE_GROUP_VERSION,
                    primary_key=["city", "time"],
                    description="Hourly AQI and Weather Features with Lags and Rolling Metrics",
                    event_time="time",
                    online_enabled=True
                )
                feature_group.insert(upload_df, wait=False)
                logger.info(f"Successfully inserted {len(upload_df)} rows to Hopsworks Feature Group '{FEATURE_GROUP_NAME}'")
                return True
            except Exception as e:
                logger.error(f"Error pushing to Hopsworks Feature Group: {e}")
                return False
        return True

    def get_features(self, city: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves feature records from Hopsworks or local cache.
        """
        if self._is_connected and self.fs:
            try:
                fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
                df = fg.read()
                if city and "city" in df.columns:
                    df = df[df["city"] == city]
                logger.info(f"Retrieved {len(df)} rows from Hopsworks Feature Group.")
                return df
            except Exception as e:
                logger.warning(f"Failed to read from Hopsworks Feature Group ({e}). Reading from local cache.")

        # Fallback to local cache
        if self._local_cache_path.exists():
            try:
                df = pd.read_parquet(self._local_cache_path)
                if city and "city" in df.columns:
                    df = df[df["city"] == city]
                logger.info(f"Retrieved {len(df)} rows from local cache.")
                return df
            except Exception as e:
                logger.error(f"Error reading local parquet cache: {e}")

        logger.warning("No feature records found in Hopsworks or local cache.")
        return pd.DataFrame()
