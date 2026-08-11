"""
Evaluation metrics for AQI regression forecasting:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 Score (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes all standard regression metrics for model assessment.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Safe MAPE calculation to prevent zero division
    denom = np.where(y_true == 0, 1e-5, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }
