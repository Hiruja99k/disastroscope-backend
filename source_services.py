import io
import csv
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import requests

logger = logging.getLogger(__name__)


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0
    import math
    p1 = math.radians(a_lat)
    p2 = math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    x = math.sin(dp/2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def fetch_gdacs_events_near(lat: float, lon: float, days: int = 10, radius_km: float = 300.0) -> List[Dict[str, Any]]:
    """Fetch GDACS global alerts (flood, cyclone, landslide) near coordinates within timeframe."""
    try:
        url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS?fromdate={fromd}"
        fromd = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        r = requests.get(url.format(fromd=fromd), timeout=20)
        r.raise_for_status()
        feats = r.json().get('features') or []
        out: List[Dict[str, Any]] = []
        for f in feats:
            props = f.get('properties') or {}
            etype = str(props.get('eventtype', '')).lower()  # 'tc' tropical cyclone, 'fl' flood, 'ls' landslide
            if etype not in ('tc', 'fl', 'ls'):
                continue
            geom = f.get('geometry') or {}
            coords = geom.get('coordinates')
            if isinstance(coords, list) and len(coords) >= 2:
                elon, elat = float(coords[0]), float(coords[1])
                if haversine_km(lat, lon, elat, elon) <= radius_km:
                    out.append({
                        'type': etype,
                        'title': props.get('name') or props.get('eventname') or 'GDACS Alert',
                        'severity': props.get('alertlevel', ''),
                        'coordinates': {'lat': elat, 'lng': elon},
                        'source': 'GDACS'
                    })
        return out
    except Exception as e:
        logger.warning(f"GDACS near fetch failed: {e}")
        return []


def fetch_firms_count_near(lat: float, lon: float, days: int = 7, radius_km: float = 50.0, token: str | None = None) -> int:
    """Count FIRMS hotspots near a point. Falls back to public 24h CSV."""
    urls = []
    if token:
        urls.append(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{token}/VIIRS_SNPP_NRT/world/{days}")
    urls.extend([
        "https://firms.modaps.eosdis.nasa.gov/active_fire/c6/csv/V1/global/VIIRS_SNPP_NRT_Global_24h.csv",
        "https://firms.modaps.eosdis.nasa.gov/active_fire/c6/csv/V1/global/VIIRS_SNPP_NRT_Global_48h.csv",
    ])
    total = 0
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            content = r.content.decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                try:
                    hlat = float(row.get('latitude') or row.get('Latitude') or row.get('lat'))
                    hlon = float(row.get('longitude') or row.get('Longitude') or row.get('lon'))
                    if haversine_km(lat, lon, hlat, hlon) <= radius_km:
                        total += 1
                except Exception:
                    continue
            break
        except Exception:
            continue
    return total


def fetch_openaq_near(lat: float, lon: float, radius_m: int = 10000) -> Dict[str, Any]:
    try:
        url = "https://api.openaq.org/v2/latest"
        params = {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 5}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json().get('results') or []
        pm25 = None
        pm10 = None
        for s in data:
            for m in s.get('measurements', []):
                if m.get('parameter') == 'pm25' and pm25 is None:
                    pm25 = m.get('value')
                if m.get('parameter') == 'pm10' and pm10 is None:
                    pm10 = m.get('value')
        return {"pm25": pm25, "pm10": pm10}
    except Exception as e:
        logger.warning(f"OpenAQ fetch failed: {e}")
        return {}


