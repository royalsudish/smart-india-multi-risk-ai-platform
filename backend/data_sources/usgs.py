from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2
from backend.config.settings import USGS_EARTHQUAKE_URL
from backend.data_sources.http_client import get_json
from backend.utils.cache import provider_cache, make_cache_key, TTL_USGS

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def get_recent_earthquake_features(lat, lon, max_radius_km=500, days=30, timeout=10):
    """
    Queries USGS Earthquake Hazards Program GeoJSON feed for seismic events
    within the configured regional radius (default 500 km) and time window (default 30 days).

    Returns:
        dict with:
            - max_quake_mag (float): maximum recorded magnitude in the window
            - quake_depth (float): focal depth in km for the maximum event
            - tsunami_flag (int): 1 if tsunami warning/watch was generated, 0 otherwise
            - event_count (int): number of events in the window
            - event_distance_km (float): distance from queried location to the maximum event
            - event_age_days (float): age in days since the maximum event occurred
            - event_time_utc (str): ISO-8601 UTC timestamp of the maximum event
            - event_id (str): USGS event ID
            - event_title (str): USGS event description/place
    """
    cache_key = make_cache_key("usgs_recent", lat, lon, {"r": max_radius_km, "d": days})
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    params = {
        "format": "geojson",
        "latitude": float(lat),
        "longitude": float(lon),
        "maxradiuskm": int(max_radius_km),
        "starttime": start_time.strftime("%Y-%m-%d"),
        "minmagnitude": 1.0,
    }

    data = get_json(USGS_EARTHQUAKE_URL, params, timeout=timeout)
    features = data.get("features") or []

    if not features:
        return {
            "max_quake_mag": 0.0,
            "quake_depth": 0.0,
            "tsunami_flag": 0,
            "event_count": 0,
            "event_distance_km": None,
            "event_age_days": None,
            "event_time_utc": None,
            "event_id": None,
            "event_title": None,
            "event_lat": None,
            "event_lon": None,
        }

    # Identify the maximum magnitude event
    best_event = None
    best_mag = -999.0

    for feat in features:
        props = feat.get("properties") or {}
        mag = props.get("mag")
        if mag is not None and float(mag) > best_mag:
            best_mag = float(mag)
            best_event = feat

    if best_event is None:
        return {
            "max_quake_mag": 0.0,
            "quake_depth": 0.0,
            "tsunami_flag": 0,
            "event_count": len(features),
            "event_distance_km": None,
            "event_age_days": None,
            "event_time_utc": None,
            "event_id": None,
            "event_title": None,
            "event_lat": None,
            "event_lon": None,
        }

    props = best_event.get("properties") or {}
    geom = best_event.get("geometry") or {}
    coords = geom.get("coordinates") or [0.0, 0.0, 0.0]
    depth = float(coords[2]) if len(coords) >= 3 else 0.0
    tsunami = int(props.get("tsunami") or 0)

    ev_lon = float(coords[0]) if len(coords) >= 1 else float(lon)
    ev_lat = float(coords[1]) if len(coords) >= 2 else float(lat)
    dist_km = round(haversine_km(float(lat), float(lon), ev_lat, ev_lon), 2)

    time_ms = props.get("time")
    if time_ms:
        ev_time = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
        age_days = round((end_time - ev_time).total_seconds() / 86400.0, 2)
        time_iso = ev_time.isoformat()
    else:
        age_days = 999.0
        time_iso = None

    res = {
        "max_quake_mag": round(float(best_mag), 2),
        "quake_depth": round(depth, 2),
        "tsunami_flag": tsunami,
        "event_count": len(features),
        "event_distance_km": dist_km,
        "event_age_days": age_days,
        "event_time_utc": time_iso,
        "event_id": best_event.get("id"),
        "event_title": props.get("title") or props.get("place"),
        "event_lat": round(ev_lat, 6),
        "event_lon": round(ev_lon, 6),
    }
    provider_cache.set(cache_key, res, ttl=TTL_USGS)
    return res

