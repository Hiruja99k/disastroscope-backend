"""
Live feed ingester for real disaster data.
Pulls events from public sources (USGS, NASA EONET, OpenFEMA) and streams
normalized events into Tinybird in near real-time.
"""

from __future__ import annotations

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)


class LiveFeedIngester:
    """Background ingester that periodically fetches disaster events and
    streams them to Tinybird. Designed to be lightweight and resilient.
    """

    def __init__(self, tinybird_service: Any):
        self.tinybird = tinybird_service
        self.poll_seconds = int(os.getenv("FEED_POLL_SECONDS", "180"))
        self.enabled = bool(self.tinybird and getattr(self.tinybird, "is_initialized", lambda: False)())
        self._stop = threading.Event()

    # ----------------------------- Public API -----------------------------
    def start(self) -> None:
        if not self.enabled:
            logger.warning("LiveFeedIngester not started: Tinybird not initialized")
            return
        thread = threading.Thread(target=self._run_loop, name="LiveFeedIngester", daemon=True)
        thread.start()
        logger.info("LiveFeedIngester started (interval=%ss)", self.poll_seconds)

    def stop(self) -> None:
        self._stop.set()

    # ----------------------------- Main Loop ------------------------------
    def _run_loop(self) -> None:
        # Stagger initial run a little to avoid blocking boot
        time.sleep(2)
        while not self._stop.is_set():
            try:
                self._ingest_once()
            except Exception as e:
                logger.exception(f"Live feed ingest iteration failed: {e}")
            finally:
                self._stop.wait(self.poll_seconds)

    def _ingest_once(self) -> None:
        since_hours = int(os.getenv("FEED_LOOKBACK_HOURS", "24"))
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=since_hours)

        events: List[Dict[str, Any]] = []
        events.extend(self._fetch_usgs_quakes(since))
        events.extend(self._fetch_eonet_events(since))
        events.extend(self._fetch_openfema_disasters(since))

        if not events:
            logger.info("No new events fetched in this cycle")
            return

        sent = 0
        for ev in events:
            try:
                ok = self.tinybird.create_disaster_event(ev)
                if ok:
                    sent += 1
            except Exception as e:
                logger.warning("Failed to stream event to Tinybird: %s", e)
        logger.info("LiveFeedIngester streamed %s/%s events to Tinybird", sent, len(events))

    # ------------------------------ Sources -------------------------------
    def _fetch_usgs_quakes(self, since: datetime) -> List[Dict[str, Any]]:
        # USGS GeoJSON feed - all earthquakes, past day
        # https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, Any]] = []
            for feat in data.get("features", []):
                try:
                    props = feat.get("properties", {})
                    geom = feat.get("geometry", {})
                    coords = (geom.get("coordinates") or [None, None])
                    lon = float(coords[0]) if coords and len(coords) > 0 else None
                    lat = float(coords[1]) if coords and len(coords) > 1 else None
                    ts_ms = props.get("time")
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else None
                    if ts and ts < since:
                        continue
                    if lat is None or lon is None:
                        continue
                    out.append({
                        "id": str(props.get("code") or props.get("ids") or props.get("net") or "usgs"),
                        "type": "earthquake",
                        "severity": self._severity_from_magnitude(props.get("mag")),
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
                        "description": props.get("place", "USGS Earthquake"),
                        "magnitude": float(props.get("mag") or 0),
                        "source": "usgs",
                        "confidence": 0.95,
                    })
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.warning("USGS fetch failed: %s", e)
            return []

    def _fetch_eonet_events(self, since: datetime) -> List[Dict[str, Any]]:
        # NASA EONET v3
        url = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open"
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, Any]] = []
            for ev in data.get("events", []):
                try:
                    categories = ev.get("categories") or []
                    cat = (categories[0].get("id") if categories else "disaster").lower()
                    # Map a few common categories
                    mapped = {
                        "wildfires": "wildfire",
                        "severeStorms": "storm",
                        "volcanoes": "volcano",
                        "floods": "flood",
                        "drought": "drought",
                    }.get(cat, cat)
                    geos = ev.get("geometry") or []
                    if not geos:
                        continue
                    geo = geos[-1]  # latest
                    coords = geo.get("coordinates") or []
                    # EONET coord order can be [lon, lat]
                    if isinstance(coords, list) and len(coords) >= 2:
                        lon, lat = float(coords[0]), float(coords[1])
                    else:
                        continue
                    ts = geo.get("date")
                    ts_iso = ts if isinstance(ts, str) else datetime.now(timezone.utc).isoformat()
                    ts_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                    if ts_dt < since:
                        continue
                    out.append({
                        "id": str(ev.get("id")),
                        "type": mapped,
                        "severity": "medium",
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts_iso,
                        "description": ev.get("title", "EONET Event"),
                        "magnitude": 0.0,
                        "source": "eonet",
                        "confidence": 0.8,
                    })
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.warning("EONET fetch failed: %s", e)
            return []

    def _fetch_openfema_disasters(self, since: datetime) -> List[Dict[str, Any]]:
        # FEMA OpenFEMA declared disasters
        url = (
            "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
            "?$orderby=declarationDate desc&$top=200"
        )
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, Any]] = []
            for row in data.get("DisasterDeclarationsSummaries", []):
                try:
                    dec = row.get("declarationDate")
                    if not dec:
                        continue
                    ts_dt = datetime.fromisoformat(dec.replace("Z", "+00:00"))
                    if ts_dt < since:
                        continue
                    lat = float(row.get("latitude") or 0)
                    lon = float(row.get("longitude") or 0)
                    if lat == 0 and lon == 0:
                        continue
                    dtype = str(row.get("incidentType", "disaster")).lower()
                    mapped = {
                        "fire": "wildfire",
                        "hurricane": "storm",
                        "tornado": "tornado",
                        "flood": "flood",
                        "drought": "drought",
                        "earthquake": "earthquake",
                    }.get(dtype, dtype)
                    out.append({
                        "id": str(row.get("disasterNumber")),
                        "type": mapped,
                        "severity": str(row.get("incidentType", "")).lower(),
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts_dt.isoformat(),
                        "description": row.get("declarationTitle", "FEMA Incident"),
                        "magnitude": 0.0,
                        "source": "openfema",
                        "confidence": 0.9,
                    })
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.warning("OpenFEMA fetch failed: %s", e)
            return []

    # ----------------------------- Utilities ------------------------------
    @staticmethod
    def _severity_from_magnitude(mag: Optional[float]) -> str:
        try:
            m = float(mag or 0)
        except Exception:
            m = 0.0
        if m >= 6.5:
            return "extreme"
        if m >= 5.5:
            return "high"
        if m >= 4.5:
            return "medium"
        return "low"


def create_ingester(tinybird_service_obj: Any) -> LiveFeedIngester:
    return LiveFeedIngester(tinybird_service_obj)


