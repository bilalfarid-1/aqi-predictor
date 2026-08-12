"""
Pipeline 03: Automated Model Retraining Pipeline (Run via GitHub Actions Cron):
Retrieves training features from Feature Store, benchmarks models, evaluates metrics, and registers champions.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from src.config import DEFAULT_CITY, MODELS_DIR
from src.feature_store import FeatureStoreManager
from src.models.train import benchmark_all_models, MultiHorizonForecaster
from src.models.registry import register_model_to_hopsworks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(city: str = DEFAULT_CITY):
    logger.info(f"Starting automated model training & benchmarking pipeline for {city}...")
    
    fs_manager = FeatureStoreManager()
    df = fs_manager.get_features(city=city)

    # If no features in store, run a quick backfill
    if df.empty or len(df) < 50:
        logger.info("Feature store empty. Ingesting training data directly...")
        from src.data_ingestion import fetch_combined_data
        from src.feature_engineering import create_features
        raw_df = fetch_combined_data(city=city, past_days=90, forecast_days=1)
        df = create_features(raw_df, is_training=True)
        fs_manager.save_features(df)

    # Benchmark Ridge, Random Forest, XGBoost, LightGBM
    best_model_name, scorecard = benchmark_all_models(df)
    
    # Save champion to Model Registry
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_multi_horizon.joblib")
    metrics_summary = {
        "avg_rmse": scorecard[best_model_name]["avg_rmse"],
        "avg_mae": scorecard[best_model_name]["avg_mae"],
        "avg_r2": scorecard[best_model_name]["avg_r2"]
    }
    
    register_model_to_hopsworks(
        model_name=f"aqi_forecaster_{best_model_name}",
        model_path=model_path,
        metrics=metrics_summary,
        description=f"Champion {best_model_name.upper()} multi-horizon AQI forecaster."
    )

    logger.info("✅ Model training & evaluation pipeline completed successfully!")
    return scorecard


if __name__ == "__main__":
    run_training_pipeline()
