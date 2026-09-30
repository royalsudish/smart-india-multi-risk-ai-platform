"""
Smart Multi-Risk Platform - NASA FIRMS Active Fire Telemetry Connector
======================================================================
Ingests Near-Real-Time (NRT) active fire and thermal anomaly hotspots from:
- VIIRS NOAA-20 NRT (J1_VIIRS_NRT / VIIRS_NOAA20_NRT)
- VIIRS NOAA-21 NRT (J2_VIIRS_NRT / VIIRS_NOAA21_NRT)

Features:
- Geographically bounded bounding box queries (lat/lon radius) to avoid world data downloads.
- Extraction of Brightness Temperature, Fire Radiative Power (FRP), Day/Night, and Confidence.
- Calculation of epicentral distance from target location and detection recency in hours.
- Categorization into distance bands: 0-10 km, 10-25 km, 25-50 km, 50-100 km.
- Short-TTL in-memory caching (300s) to prevent redundant remote requests.
- Strict timeout and graceful degradation if FIRMS_MAP_KEY is missing or remote service is unreachable.
"""

import csv
import io
import math
import os
import socket
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import urllib3.util.connection as urllib3_conn

from backend.config.settings import FIRMS_MAP_KEY, FIRMS_BASE_URL
from backend.data_sources.http_client import session
from backend.utils.cache import provider_cache, make_cache_key

# Force IPv4 resolution to prevent dual-stack IPv6 TLS handshake timeouts on systems with unrouted IPv6
def _allowed_gai_family():
    return socket.AF_INET

urllib3_conn.allowed_gai_family = _allowed_gai_family

EARTH_RADIUS_KM = 6371.0
FIRMS_CACHE_TTL = 300.0  # 5 minutes cache for dynamic fire data
DEFAULT_RADIUS_KM = 100.0
DEFAULT_TIMEOUT = float(os.getenv("FIRMS_TIMEOUT_SECONDS", "10.0"))

# Supported VIIRS satellite instruments (preferring NOAA-20 and NOAA-21 over Suomi-NPP)
PREFERRED_FIRMS_SOURCES = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in kilometers."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_detection_age_hours(acq_date_str: str, acq_time_str: str) -> float:
    """
    Computes elapsed age in hours from satellite acquisition timestamp.
    acq_date_str: 'YYYY-MM-DD'
    acq_time_str: 'HHMM' (UTC)
    """
    try:
        t_clean = str(acq_time_str).strip().zfill(4)
        dt_str = f"{acq_date_str.strip()} {t_clean}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = (now - dt).total_seconds() / 3600.0
        return max(0.0, round(diff, 1))
    except Exception:
        return 0.0


def fetch_firms_hotspots(
    lat: float,
    lon: float,
    radius_km: float = DEFAULT_RADIUS_KM,
    day_range: int = 1,
    map_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetches active fire detections within a geographic bounding box around (lat, lon).
    Returns parsed hotspots, distance bands, nearest detection distance, and peak FRP.
    """
    # 0. Support explicit fixture overrides (for unit tests and deterministic simulation)
    if overrides and "firms_fire" in overrides:
        return overrides["firms_fire"]

    active_key = map_key or FIRMS_MAP_KEY
    if not active_key or active_key.strip() == "":
        return {
            "source": "NASA_FIRMS",
            "status": "unavailable",
            "missing_provider": True,
            "reason": "FIRMS_MAP_KEY not configured in environment.",
            "hotspots": [],
            "hotspots_count": 0,
            "nearest_fire_distance_km": None,
            "latest_detection_age_hours": None,
            "max_frp": 0.0,
            "detections_10km": 0,
            "detections_25km": 0,
            "detections_50km": 0,
            "detections_100km": 0,
            "satellite_sources": [],
        }

    # 1. Check cache
    cache_key = make_cache_key("firms_hotspots_v1", round(lat, 2), round(lon, 2), {"radius_km": radius_km, "day_range": day_range})
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    # 2. Compute bounding box: min_lon,min_lat,max_lon,max_lat
    deg_lat = radius_km / 111.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    deg_lon = radius_km / (111.0 * cos_lat)

    min_lat = max(-90.0, lat - deg_lat)
    max_lat = min(90.0, lat + deg_lat)
    min_lon = max(-180.0, lon - deg_lon)
    max_lon = min(180.0, lon + deg_lon)

    bbox_str = f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"

    all_hotspots: List[Dict[str, Any]] = []
    used_sources: List[str] = []
    provider_available = False

    s = session()
    headers = {"User-Agent": "SmartMultiRiskPlatform/1.0"}

    # Query NOAA-21 first, then NOAA-20
    for src in PREFERRED_FIRMS_SOURCES:
        url = f"{FIRMS_BASE_URL.rstrip('/')}/{active_key}/{src}/{bbox_str}/{day_range}"
        try:
            resp = s.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                provider_available = True
                csv_text = resp.text
                if csv_text and not csv_text.startswith("Invalid") and "latitude" in csv_text:
                    reader = csv.DictReader(io.StringIO(csv_text))
                    for row in reader:
                        try:
                            h_lat = float(row.get("latitude", 0.0))
                            h_lon = float(row.get("longitude", 0.0))
                            dist = haversine_km(lat, lon, h_lat, h_lon)
                            if dist > radius_km:
                                continue

                            acq_date = row.get("acq_date", "")
                            acq_time = row.get("acq_time", "")
                            age_hrs = calculate_detection_age_hours(acq_date, acq_time)

                            frp_val = float(row.get("frp", 0.0)) if row.get("frp") else 0.0
                            b4 = float(row.get("bright_ti4", 0.0)) if row.get("bright_ti4") else 0.0
                            b5 = float(row.get("bright_ti5", 0.0)) if row.get("bright_ti5") else 0.0
                            conf = row.get("confidence", "nominal")
                            sat = row.get("satellite", src)
                            inst = row.get("instrument", "VIIRS")
                            dn = row.get("daynight", "D")

                            all_hotspots.append({
                                "latitude": round(h_lat, 4),
                                "longitude": round(h_lon, 4),
                                "coordinates": [round(h_lon, 4), round(h_lat, 4)],
                                "distance_km": round(dist, 1),
                                "acq_date": acq_date,
                                "acq_time": acq_time,
                                "detection_age_hours": age_hrs,
                                "satellite": sat,
                                "instrument": inst,
                                "confidence": conf,
                                "bright_ti4": round(b4, 1),
                                "bright_ti5": round(b5, 1),
                                "frp": round(frp_val, 1),
                                "daynight": dn,
                                "source": src,
                            })
                            used_sources.append(src)
                        except Exception:
                            continue
        except Exception:
            # Continue to next source if one times out
            continue

    if not provider_available:
        result = {
            "source": "NASA_FIRMS",
            "status": "unavailable",
            "missing_provider": True,
            "reason": "NASA FIRMS endpoint timed out or was unreachable.",
            "hotspots": [],
            "hotspots_count": 0,
            "nearest_fire_distance_km": None,
            "latest_detection_age_hours": None,
            "max_frp": 0.0,
            "detections_10km": 0,
            "detections_25km": 0,
            "detections_50km": 0,
            "detections_100km": 0,
            "satellite_sources": [],
        }
        provider_cache.set(cache_key, result, ttl=60.0)
        return result

    # Sort hotspots by distance ascending
    all_hotspots.sort(key=lambda h: h["distance_km"])

    det_10 = sum(1 for h in all_hotspots if h["distance_km"] <= 10.0)
    det_25 = sum(1 for h in all_hotspots if h["distance_km"] <= 25.0)
    det_50 = sum(1 for h in all_hotspots if h["distance_km"] <= 50.0)
    det_100 = sum(1 for h in all_hotspots if h["distance_km"] <= 100.0)

    nearest_dist = all_hotspots[0]["distance_km"] if all_hotspots else None
    min_age = min((h["detection_age_hours"] for h in all_hotspots), default=None)
    max_frp = max((h["frp"] for h in all_hotspots), default=0.0)

    result = {
        "source": "NASA_FIRMS",
        "status": "available",
        "missing_provider": False,
        "hotspots": all_hotspots,
        "hotspots_count": len(all_hotspots),
        "nearest_fire_distance_km": nearest_dist,
        "latest_detection_age_hours": min_age,
        "max_frp": max_frp,
        "detections_10km": det_10,
        "detections_25km": det_25,
        "detections_50km": det_50,
        "detections_100km": det_100,
        "satellite_sources": list(set(used_sources)),
        "bbox": bbox_str,
        "reason": f"NASA FIRMS returned {len(all_hotspots)} hotspot(s) within {radius_km} km.",
    }

    provider_cache.set(cache_key, result, ttl=FIRMS_CACHE_TTL)
    return result
