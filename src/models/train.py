# LightGBM & Ridge
"""
Model Training & Experimentation Engine:
- Multi-Model Benchmarking (Ridge Regression, Random Forest, XGBoost, LightGBM)
- Multi-Horizon Forecasting (+24h, +48h, +72h)
- Cross-Validation & Metric Tracking
"""
import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import MODELS_DIR, FORECAST_HOURS
from src.feature_engineering import get_feature_columns
from src.models.evaluate import compute_regression_metrics

logger = logging.getLogger(__name__)


class MultiHorizonForecaster:
    """
    Trains separate specialized models for multi-step horizons: +24h, +48h, and +72h.
    """
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.models: Dict[int, Any] = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.metrics: Dict[int, Dict[str, float]] = {}

    def _get_base_estimator(self):
        if self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        elif self.model_type == "lightgbm":
            return LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        else: # Default: XGBoost
            return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

    def train(self, df: pd.DataFrame, train_split_ratio: float = 0.8) -> Dict[int, Dict[str, float]]:
        """
        Trains and validates multi-horizon models on chronological train/test split.
        """
        self.feature_names = get_feature_columns(df)
        logger.info(f"Training multi-horizon {self.model_type.upper()} on {len(self.feature_names)} features...")

        # Chronological Time-Series Split (no shuffle)
        split_idx = int(len(df) * train_split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train_raw = train_df[self.feature_names]
        X_test_raw = test_df[self.feature_names]

        # Fit Scaler on training data
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        for horizon in FORECAST_HOURS:
            target_col = f"target_aqi_{horizon}h"
            if target_col not in df.columns:
                logger.warning(f"Target column '{target_col}' not found. Skipping horizon +{horizon}h.")
                continue

            y_train = train_df[target_col].values
            y_test = test_df[target_col].values

            model = self._get_base_estimator()
            model.fit(X_train, y_train)

            # Predict & Evaluate
            y_pred = model.predict(X_test)
            metrics = compute_regression_metrics(y_test, y_pred)
            self.metrics[horizon] = metrics
            self.models[horizon] = model

            logger.info(f"Horizon +{horizon}h: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")

        return self.metrics

    def predict(self, feature_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Generates predictions for each horizon using the latest feature row.
        """
        X = feature_df[self.feature_names]
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for horizon, model in self.models.items():
            preds = model.predict(X_scaled)
            # Clip predictions to realistic AQI bounds (0 to 500)
            predictions[horizon] = np.clip(preds, 0, 500)
        return predictions

    def save(self, file_path: Optional[str] = None):
        """Persists trained forecaster to local disk."""
        if file_path is None:
            file_path = os.path.join(MODELS_DIR, f"{self.model_type}_multi_horizon.joblib")
        joblib.dump(self, file_path)
        logger.info(f"Saved {self.model_type} forecaster model to: {file_path}")

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "MultiHorizonForecaster":
        """Loads trained forecaster from local disk."""
        if file_path is None:
            file_path = os.path.join(MODELS_DIR, "xgboost_multi_horizon.joblib")
        return joblib.load(file_path)


def benchmark_all_models(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Trains Ridge, Random Forest, XGBoost, and LightGBM models, reporting comparative metrics.
    Returns the champion model name and metric scorecard.
    """
    models_to_test = ["ridge", "random_forest", "xgboost", "lightgbm"]
    scorecard = {}
    best_model_name = "xgboost"
    best_avg_rmse = float("inf")

    for m_type in models_to_test:
        forecaster = MultiHorizonForecaster(model_type=m_type)
        metrics = forecaster.train(df)
        avg_rmse = np.mean([v["rmse"] for v in metrics.values()])
        avg_mae = np.mean([v["mae"] for v in metrics.values()])
        avg_r2 = np.mean([v["r2"] for v in metrics.values()])

        scorecard[m_type] = {
            "metrics_per_horizon": metrics,
            "avg_rmse": round(avg_rmse, 2),
            "avg_mae": round(avg_mae, 2),
            "avg_r2": round(avg_r2, 4)
        }

        # Persist each model artifact
        forecaster.save()

        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_model_name = m_type

    logger.info(f"🏆 Champion Model: {best_model_name.upper()} (Avg RMSE: {best_avg_rmse:.2f})")
    return best_model_name, scorecard
