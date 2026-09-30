import time
import threading
from math import radians, sin, cos, sqrt, atan2
from backend.config.settings import GDACS_EVENTS_URL
from backend.data_sources.http_client import session

EARTH_RADIUS_KM = 6371.0

# In-memory cache for GDACS global feed with 5-minute TTL
_CACHE_LOCK = threading.Lock()
_CACHE_DATA = None
_CACHE_TIMESTAMP = 0.0
_CACHE_TTL_SECONDS = 300.0  # 5 minutes


def clear_gdacs_cache():
    global _CACHE_DATA, _CACHE_TIMESTAMP
    with _CACHE_LOCK:
        _CACHE_DATA = None
        _CACHE_TIMESTAMP = 0.0


def is_gdacs_cached():
    with _CACHE_LOCK:
        now = time.time()
        return _CACHE_DATA is not None and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def _fetch_gdacs_events(timeout=12):
    global _CACHE_DATA, _CACHE_TIMESTAMP

    with _CACHE_LOCK:
        now = time.time()
        if _CACHE_DATA is not None and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
            return _CACHE_DATA

    s = session()
    headers = {
        "User-Agent": "SmartMultiRiskPlatform/1.0",
        "Accept": "application/json",
    }
    r = s.get(GDACS_EVENTS_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    features = data.get("features") or []

    with _CACHE_LOCK:
        _CACHE_DATA = features
        _CACHE_TIMESTAMP = time.time()

    return features


from datetime import datetime, timezone

def _parse_gdacs_datetime(dt_str):
    if not dt_str:
        return None
    try:
        s = str(dt_str).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def is_gdacs_event_active(props, now_utc=None):
    """
    Determines whether a GDACS event is currently active.
    1. Checks explicit 'iscurrent' status (false => inactive, true => active).
    2. Validates against 'fromdate' and 'todate' timezone-aware UTC interval.
    3. An event that ended in the past is never considered active.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    is_curr_raw = props.get("iscurrent")
    is_current = None
    if str(is_curr_raw).lower() in ("true", "1"):
        is_current = True
    elif str(is_curr_raw).lower() in ("false", "0"):
        is_current = False

    start_dt = _parse_gdacs_datetime(props.get("fromdate"))
    end_dt = _parse_gdacs_datetime(props.get("todate"))

    # If end date is in the past, event has expired
    if end_dt and now_utc > end_dt:
        return False

    # If start date is in the future, event has not started
    if start_dt and now_utc < start_dt:
        return False

    # Within valid time window
    if start_dt and end_dt and (start_dt <= now_utc <= end_dt):
        return True

    # If todate is missing or open-ended, rely on iscurrent
    if is_current is False:
        return False
    if is_current is True:
        return True

    return False


def get_cyclone_and_volcano_alerts(lat, lon, cyclone_radius_km=1000, volcano_radius_km=500, timeout=12):
    """
    Checks GDACS events for Tropical Cyclones (TC) and Volcanoes (VO)
    that satisfy BOTH temporal validity (currently active) AND geographic relevance
    within the configured impact radii from the requested coordinates.

    Returns:
        dict with:
            - gdacs_cyclone (int): 1 if active cyclone within impact radius, 0 otherwise
            - gdacs_volcano (int): 1 if active volcano within impact radius, 0 otherwise
            - active_cyclones_nearby (int): count of nearby active cyclones
            - active_volcanoes_nearby (int): count of nearby active volcanoes
            - audit_events (list): audit metadata for inspected events
    """
    events = _fetch_gdacs_events(timeout=timeout)
    now_utc = datetime.now(timezone.utc)

    cyclone_count = 0
    volcano_count = 0
    audit_events = []

    for ev in events:
        geom = ev.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        ev_lon = float(coords[0])
        ev_lat = float(coords[1])
        dist = haversine_km(lat, lon, ev_lat, ev_lon)

        props = ev.get("properties") or {}
        etype = props.get("eventtype")
        if etype not in ("TC", "VO"):
            continue

        active = is_gdacs_event_active(props, now_utc)
        radius = cyclone_radius_km if etype == "TC" else volcano_radius_km
        geo_relevant = (dist <= radius)

        audit_events.append({
            "event_id": props.get("eventid") or ev.get("id"),
            "event_name": props.get("eventname") or props.get("name"),
            "event_type": etype,
            "event_start": props.get("fromdate"),
            "event_end": props.get("todate"),
            "status": props.get("iscurrent"),
            "distance_km": round(dist, 2),
            "lat": round(ev_lat, 6),
            "lon": round(ev_lon, 6),
            "active": active,
            "geographically_relevant": geo_relevant,
        })

        if active and geo_relevant:
            if etype == "TC":
                cyclone_count += 1
            elif etype == "VO":
                volcano_count += 1

    return {
        "gdacs_cyclone": 1 if cyclone_count > 0 else 0,
        "gdacs_volcano": 1 if volcano_count > 0 else 0,
        "active_cyclones_nearby": cyclone_count,
        "active_volcanoes_nearby": volcano_count,
        "audit_events": audit_events,
    }
