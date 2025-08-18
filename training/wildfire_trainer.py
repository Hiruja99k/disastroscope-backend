import os
import io
import csv
import math
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

logger = logging.getLogger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_firms_hotspots(token: str, days: int = 7) -> List[Tuple[float, float]]:
    """Fetch recent wildfire hotspots from NASA FIRMS.
    Tries tokenized API first; falls back to public 24h/48h CSV if token fails.
    Returns list of (lat, lon).
    """
    hotspots: List[Tuple[float, float]] = []
    urls = []
    # Token API (area world, VIIRS_SNPP_NRT)
    if token:
        urls.append(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{token}/VIIRS_SNPP_NRT/world/{days}")
    # Public fallbacks
    urls.extend([
        "https://firms.modaps.eosdis.nasa.gov/active_fire/c6/csv/V1/global/VIIRS_SNPP_NRT_Global_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/active_fire/c6/csv/V1/global/VIIRS_SNPP_NRT_Global_48h.csv",
        "https://firms.modaps.eosdis.nasa.gov/active_fire/c6/csv/V1/global/MODIS_C6_1_Global_24h.csv",
    ])
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            content = resp.content.decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    lat = float(row.get('latitude') or row.get('Latitude') or row.get('lat'))
                    lon = float(row.get('longitude') or row.get('Longitude') or row.get('lon'))
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        hotspots.append((lat, lon))
                except Exception:
                    continue
            if hotspots:
                break
        except Exception as e:
            logger.warning(f"FIRMS fetch failed for {url}: {e}")
    return hotspots


def fetch_era5_features(lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, float]:
    """Fetch ERA5 reanalysis via Open-Meteo archive and aggregate simple features."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover",
        "start_date": start.strftime('%Y-%m-%d'),
        "end_date": end.strftime('%Y-%m-%d'),
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hourly = data.get('hourly', {})
        # Extract arrays
        temp = hourly.get('temperature_2m') or []
        rh = hourly.get('relative_humidity_2m') or []
        ws = hourly.get('wind_speed_10m') or []
        pcp = hourly.get('precipitation') or []
        cc = hourly.get('cloud_cover') or []
        def safe_mean(x):
            return float(np.mean(x)) if isinstance(x, list) and len(x) > 0 else 0.0
        def safe_sum(x):
            return float(np.sum(x)) if isinstance(x, list) and len(x) > 0 else 0.0
        feats = {
            'temperature_mean': safe_mean(temp),
            'humidity_mean': safe_mean(rh),
            'wind_mean': safe_mean(ws),
            'precip_total': safe_sum(pcp),
            'cloud_mean': safe_mean(cc),
        }
        return feats
    except Exception as e:
        logger.warning(f"ERA5 fetch failed for ({lat},{lon}): {e}")
        return {'temperature_mean': 20.0, 'humidity_mean': 50.0, 'wind_mean': 3.0, 'precip_total': 0.0, 'cloud_mean': 40.0}


def build_training_set(token: str, days: int = 7, max_pos: int = 300, neg_per_pos: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    hotspots = fetch_firms_hotspots(token, days=days)
    random.shuffle(hotspots)
    if not hotspots:
        raise RuntimeError("No FIRMS hotspots fetched; cannot train wildfire model")
    hotspots = hotspots[:max_pos]

    X: List[List[float]] = []
    y: List[int] = []

    # Positive samples
    for lat, lon in hotspots:
        feats = fetch_era5_features(lat, lon, start, now)
        X.append([feats['temperature_mean'], feats['humidity_mean'], feats['wind_mean'], feats['precip_total'], feats['cloud_mean']])
        y.append(1)

    # Negative samples: offset away from hotspots and ensure no hotspot within 50 km
    for lat, lon in hotspots:
        for _ in range(neg_per_pos):
            for _try in range(5):
                dlat = random.uniform(1.0, 3.0) * random.choice([-1, 1])
                dlon = random.uniform(1.0, 3.0) * random.choice([-1, 1])
                nlat = max(-85.0, min(85.0, lat + dlat))
                nlon = max(-179.0, min(179.0, lon + dlon))
                far = True
                for hlat, hlon in hotspots[:200]:
                    if haversine_km(nlat, nlon, hlat, hlon) < 50.0:
                        far = False
                        break
                if far:
                    feats = fetch_era5_features(nlat, nlon, start, now)
                    X.append([feats['temperature_mean'], feats['humidity_mean'], feats['wind_mean'], feats['precip_total'], feats['cloud_mean']])
                    y.append(0)
                    break

    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_and_save(model_dir: str, token: str, days: int = 7) -> Dict[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Building wildfire training set from FIRMS + ERA5...")
    X, y = build_training_set(token, days=days)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xs, y)
    model_path = os.path.join(model_dir, 'wildfire_model.joblib')
    scaler_path = os.path.join(model_dir, 'wildfire_scaler.joblib')
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved wildfire model to {model_path}")
    return {"model": model_path, "scaler": scaler_path}


