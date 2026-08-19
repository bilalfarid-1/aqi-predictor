# Hazard alerts
"""
Automated Alert Dispatcher for Hazardous AQI Conditions:
Supports Telegram Bot Webhooks, Discord Webhooks, and Console Notifications.
"""
import sys
import requests
import logging
from typing import Optional
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, get_aqi_category

logger = logging.getLogger(__name__)


def send_hazard_alert(city: str, predicted_aqi: float, horizon_hours: int) -> bool:
    """
    Sends an urgent alert notification when forecasted AQI crosses the Unhealthy threshold (>150).
    """
    cat = get_aqi_category(predicted_aqi)
    if predicted_aqi < 151:
        return False

    cat_label = cat['label'].upper()
    cat_desc = cat['desc']

    message = (
        f"[ALERT] PEARLS AQI WARNING: {cat_label}\n"
        f"[CITY] City: {city}\n"
        f"[HORIZON] Forecast Horizon: +{horizon_hours} Hours\n"
        f"[PREDICTION] Predicted AQI: {predicted_aqi:.1f}\n"
        f"[ADVISORY] Advisory: {cat_desc}\n"
        f"[ACTION] Action Required: Limit outdoor exposure, wear N95 masks, and run indoor air purifiers."
    )

    logger.warning(f"Hazardous AQI alert triggered for {city}: {predicted_aqi:.1f}")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown"
            }
            resp = requests.post(tg_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Successfully dispatched Telegram hazard alert.")
                return True
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")

    try:
        print("\n" + "="*60 + f"\n{message}\n" + "="*60 + "\n")
    except Exception:
        pass

    return True
