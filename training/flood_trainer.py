import os
import io
import math
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

logger = logging.getLogger(__name__)


def country_grid() -> List[Tuple[float, float]]:
    """Create a coarse global grid (-60..60 lat to avoid poles) for negatives/background."""
    lats = list(range(-60, 61, 5))
    lons = list(range(-180, 181, 5))
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def fetch_glofas_alerts(days: int = 10) -> List[Tuple[float, float]]:
    """Fetch recent flood reports from GDACS (global) as a proxy for positive labels.
    GDACS provides global alerts including flood events with approximate coordinates.
    """
    try:
        url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS?fromdate={fromd}"
        fromd = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        r = requests.get(url.format(fromd=fromd), timeout=30)
        if r.status_code != 200:
            return []
        items = r.json().get('features') or []
        out: List[Tuple[float, float]] = []
        for f in items:
            try:
                props = f.get('properties') or {}
                if str(props.get('eventtype', '')).lower() != 'fl':
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
        logger.warning(f"GDACS fetch failed: {e}")
        return []


def fetch_era5_features(lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, float]:
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,pressure_msl_mean,temperature_2m_mean,wind_speed_10m_max",
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
            'mslp': smean('pressure_msl_mean'),
            't2m': smean('temperature_2m_mean'),
            'wind_max': smean('wind_speed_10m_max'),
        }
    except Exception:
        return {'precip_3d': 0.0, 'mslp': 1013.0, 't2m': 20.0, 'wind_max': 4.0}


def build_training_set(days: int = 10, max_pos: int = 300, neg_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    now = datetime.utcnow()
    start = now - timedelta(days=3)
    pos = fetch_glofas_alerts(days=days)[:max_pos]
    if not pos:
        raise RuntimeError("No global flood alerts available from GDACS; cannot train flood model")
    X: List[List[float]] = []
    y: List[int] = []
    for lat, lon in pos:
        feats = fetch_era5_features(lat, lon, start, now)
        X.append([feats['precip_3d'], feats['mslp'], feats['t2m'], feats['wind_max']])
        y.append(1)
    # negatives: sample grid points not near positives
    grid = country_grid()
    import random
    random.shuffle(grid)
    for lat, lon in grid:
        if len(X) >= len(pos) * (1 + neg_factor):
            break
        # skip if close to any positive (coarse)
        if any(abs(lat - plat) < 1.0 and abs(lon - plon) < 1.0 for plat, plon in pos[:200]):
            continue
        feats = fetch_era5_features(lat, lon, start, now)
        X.append([feats['precip_3d'], feats['mslp'], feats['t2m'], feats['wind_max']])
        y.append(0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_and_save(model_dir: str, days: int = 10) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    X, y = build_training_set(days=days)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, y)
    m = os.path.join(model_dir, 'flood_model.joblib')
    s = os.path.join(model_dir, 'flood_scaler.joblib')
    joblib.dump(clf, m)
    joblib.dump(scaler, s)
    logger.info(f"Saved flood model to {m}")
    return {"model": m, "scaler": s}


