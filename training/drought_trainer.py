import os
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

logger = logging.getLogger(__name__)


def fetch_era5_features(lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, float]:
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "start_date": start.strftime('%Y-%m-%d'),
        "end_date": end.strftime('%Y-%m-%d'),
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        daily = r.json().get('daily', {})
        def smean(k):
            arr = daily.get(k) or []
            return float(np.mean(arr)) if arr else 0.0
        def ssum(k):
            arr = daily.get(k) or []
            return float(np.sum(arr)) if arr else 0.0
        return {
            't2m_mean': smean('temperature_2m_mean'),
            'precip_7d': ssum('precipitation_sum'),
            'rh_mean': smean('relative_humidity_2m_mean'),
        }
    except Exception:
        return {'t2m_mean': 25.0, 'precip_7d': 0.0, 'rh_mean': 40.0}


def build_training_set() -> Tuple[np.ndarray, np.ndarray]:
    # Unsupervised proxy: sample global grid; label drought when precip_7d is in the bottom 10% and t2m high
    lats = list(range(-60, 61, 5))
    lons = list(range(-180, 181, 5))
    pts = [(float(a), float(b)) for a in lats for b in lons]
    now = datetime.utcnow()
    start = now - timedelta(days=7)
    X = []
    y = []
    vals = []
    for lat, lon in pts:
        f = fetch_era5_features(lat, lon, start, now)
        X.append([f['t2m_mean'], f['precip_7d'], f['rh_mean']])
        vals.append(f['precip_7d'])
    import numpy as np
    X = np.array(X, dtype=float)
    thresh = float(np.percentile([v for v in vals], 10))
    for i, (lat, lon) in enumerate(pts):
        precip = X[i, 1]
        temp = X[i, 0]
        y.append(1 if (precip <= thresh and temp > 28.0) else 0)
    return X, np.array(y, dtype=int)


def train_and_save(model_dir: str) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    X, y = build_training_set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, y)
    import joblib
    m = os.path.join(model_dir, 'drought_model.joblib')
    s = os.path.join(model_dir, 'drought_scaler.joblib')
    joblib.dump(clf, m)
    joblib.dump(scaler, s)
    logger.info(f"Saved drought model to {m}")
    return {"model": m, "scaler": s}


