import requests
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict

logger = logging.getLogger(__name__)


def get_earthquake_hazard(lat: float, lon: float, days: int = 30, radius_km: int = 300) -> float:
    """Estimate short-term earthquake probability using recent seismicity from USGS.
    Uses a simple Poisson process approximation: p(T) = 1 - exp(-lambda*T), where
    lambda is the daily event rate within the radius (M4+), and T is 3 days.
    Returns a probability in [0,1]. If no data or out of coverage, returns ~0.
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%d"),
            "endtime": end.strftime("%Y-%m-%d"),
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": radius_km,
            "minmagnitude": 4.0,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return 0.0
        data = r.json()
        n = int(data.get("metadata", {}).get("count", 0))
        if n <= 0:
            return 0.0
        daily_rate = n / float(days)
        T_days = 3.0
        prob = 1.0 - math.exp(-daily_rate * T_days)
        # cap extreme values
        return max(0.0, min(1.0, prob))
    except Exception as e:
        logger.warning(f"USGS hazard fetch failed: {e}")
        return 0.0


