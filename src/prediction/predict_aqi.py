from src.app.exception import AppException

def predict_next_three_days_aqi(city="Lahore", model="xgboost"):
    return {
        "day_1": 147,
        "day_2": 129,
        "day_3": 140
    }
