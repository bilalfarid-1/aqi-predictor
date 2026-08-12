"""
Hopsworks Model Registry Connector:
Registers trained champion models with metrics, parameters, and versioning.
"""
import os
import logging
from typing import Optional, Dict, Any

from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, MODELS_DIR

logger = logging.getLogger(__name__)


def register_model_to_hopsworks(model_name: str, model_path: str, metrics: Dict[str, Any], description: str = "AQI Predictor Model"):
    """
    Uploads trained model artifact to Hopsworks Model Registry.
    """
    if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY.strip() == "":
        logger.info("No HOPSWORKS_API_KEY. Model safely persisted in local model registry cache.")
        return False

    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        mr = project.get_model_registry()

        model_meta = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=description,
            input_example=None
        )
        model_meta.save(model_path)
        logger.info(f"Successfully registered model '{model_name}' to Hopsworks Model Registry!")
        return True

    except Exception as e:
        logger.error(f"Error registering model to Hopsworks: {e}")
        return False
