import os
import io
import csv
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

logger = logging.getLogger(__name__)


def fetch_storm_reports(days: int = 7) -> List[Tuple[float, float]]:
    """Fetch global storm proxies via Open-Meteo archive using high wind thresholds grid sampling.
    For simplicity and to avoid paid feeds, we sample a set of grid points and mark positives where mean wind>10m/s.
    """
    # Sample coarse grid
    lats = list(range(-60, 61, 10))
    lons = list(range(-180, 181, 15))
    points = [(lat, lon) for lat in lats for lon in lons]
    pos: List[Tuple[float, float]] = []
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    for lat, lon in points:
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/era5",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "wind_speed_10m,pressure_msl,cloud_cover,precipitation",
                    "start_date": start.strftime('%Y-%m-%d'),
                    "end_date": now.strftime('%Y-%m-%d'),
                    "timezone": "UTC",
                },
                timeout=25,
            )
            if r.status_code != 200:
                continue
            data = r.json().get('hourly', {})
            ws = data.get('wind_speed_10m') or []
            if len(ws) > 0 and float(np.mean(ws)) > 10.0:
                pos.append((lat, lon))
        except Exception:
            continue
    return pos


def build_training_set(days: int = 7, max_pos: int = 200, neg_per_pos: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    positives = fetch_storm_reports(days=days)[:max_pos]
    X: List[List[float]] = []
    y: List[int] = []
    for lat, lon in positives:
        feats = fetch_features(lat, lon, start, now)
        X.append([feats['wind_mean'], feats['pressure_mean'], feats['cloud_mean'], feats['precip_total']])
        y.append(1)
    # negatives from nearby points with calmer winds
    for lat, lon in positives:
        nlat = max(-60, min(60, lat + 5))
        nlon = ((lon + 20 + 180) % 360) - 180
        feats = fetch_features(nlat, nlon, start, now)
        X.append([feats['wind_mean'], feats['pressure_mean'], feats['cloud_mean'], feats['precip_total']])
        y.append(0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def fetch_features(lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, float]:
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,pressure_msl,cloud_cover,precipitation",
        "start_date": start.strftime('%Y-%m-%d'),
        "end_date": end.strftime('%Y-%m-%d'),
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        hourly = r.json().get('hourly', {})
        def smean(k):
            arr = hourly.get(k) or []
            return float(np.mean(arr)) if arr else 0.0
        def ssum(k):
            arr = hourly.get(k) or []
            return float(np.sum(arr)) if arr else 0.0
        return {
            'wind_mean': smean('wind_speed_10m'),
            'pressure_mean': smean('pressure_msl'),
            'cloud_mean': smean('cloud_cover'),
            'precip_total': ssum('precipitation'),
        }
    except Exception:
        return {'wind_mean': 0.0, 'pressure_mean': 1013.0, 'cloud_mean': 40.0, 'precip_total': 0.0}


def train_and_save(model_dir: str, days: int = 7) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    X, y = build_training_set(days=days)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, y)
    import joblib
    mpath = os.path.join(model_dir, 'storm_model.joblib')
    spath = os.path.join(model_dir, 'storm_scaler.joblib')
    joblib.dump(clf, mpath)
    joblib.dump(scaler, spath)
    logger.info(f"Saved storm model to {mpath}")
    return {"model": mpath, "scaler": spath}


