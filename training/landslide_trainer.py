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


def fetch_gdacs_landslides(days: int = 14) -> List[Tuple[float, float]]:
    """Fetch recent landslide alerts from GDACS (global)."""
    try:
        url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS?fromdate={fromd}"
        fromd = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        r = requests.get(url.format(fromd=fromd), timeout=30)
        if r.status_code != 200:
            return []
        feats = r.json().get('features') or []
        out: List[Tuple[float, float]] = []
        for f in feats:
            try:
                props = f.get('properties') or {}
                # GDACS event type for landslide
                if str(props.get('eventtype', '')).lower() not in ('ls', 'landslide'):
                    continue
                geom = f.get('geometry') or {}
                coords = geom.get('coordinates')
                if isinstance(coords, list) and len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        out.append((lat, lon))
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"GDACS landslide fetch failed: {e}")
        return []


def fetch_era5_features(lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, float]:
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,relative_humidity_2m_mean,temperature_2m_mean",
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
            'precip_3d': ssum('precipitation_sum'),
            'rh_mean': smean('relative_humidity_2m_mean'),
            't2m_mean': smean('temperature_2m_mean'),
        }
    except Exception:
        return {'precip_3d': 0.0, 'rh_mean': 60.0, 't2m_mean': 20.0}


def build_training_set(days: int = 14, max_pos: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    now = datetime.utcnow()
    start = now - timedelta(days=3)
    pos = fetch_gdacs_landslides(days=days)[:max_pos]
    if not pos:
        raise RuntimeError("No GDACS landslide alerts; cannot train landslide model")
    X: List[List[float]] = []
    y: List[int] = []
    for lat, lon in pos:
        f = fetch_era5_features(lat, lon, start, now)
        X.append([f['precip_3d'], f['rh_mean'], f['t2m_mean']])
        y.append(1)
    # Negatives: shift coords
    for lat, lon in pos:
        nlat = max(-60, min(60, lat + 3.0))
        nlon = ((lon + 3.0 + 180) % 360) - 180
        f = fetch_era5_features(nlat, nlon, start, now)
        X.append([f['precip_3d'], f['rh_mean'], f['t2m_mean']])
        y.append(0)
    import numpy as np
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_and_save(model_dir: str, days: int = 14) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    X, y = build_training_set(days=days)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, y)
    m = os.path.join(model_dir, 'landslide_model.joblib')
    s = os.path.join(model_dir, 'landslide_scaler.joblib')
    joblib.dump(clf, m)
    joblib.dump(scaler, s)
    logger.info(f"Saved landslide model to {m}")
    return {"model": m, "scaler": s}


