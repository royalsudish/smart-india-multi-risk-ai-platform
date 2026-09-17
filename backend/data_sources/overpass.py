from math import radians, sin, cos, sqrt, atan2, pi
from backend.config.settings import OVERPASS_ENDPOINTS
from backend.data_sources.http_client import post_form_json

EARTH_RADIUS_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
    return 2 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(1-a))

def _run(query):
    errors = []
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            return post_form_json(endpoint, {"data": query}, timeout=35), endpoint
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))

def drainage_features(lat, lon, radius_m=2000):
    radius_m = max(200, min(int(radius_m), 10000))
    q = f"""
    [out:json][timeout:25];
    (
      way["waterway"~"river|stream|canal|drain|ditch"](around:{radius_m},{lat},{lon});
    );
    out tags geom;
    """
    data, endpoint = _run(q)
    elements = data.get("elements", [])

    total_length_m = 0.0
    min_distance_m = None
    by_type = {}

    for el in elements:
        kind = (el.get("tags") or {}).get("waterway", "unknown")
        by_type[kind] = by_type.get(kind, 0) + 1
        geom = el.get("geometry") or []

        for p in geom:
            d = haversine_m(lat, lon, p["lat"], p["lon"])
            if min_distance_m is None or d < min_distance_m:
                min_distance_m = d

        for a, b in zip(geom, geom[1:]):
            total_length_m += haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])

    area_km2 = pi * (radius_m / 1000.0) ** 2
    drainage_density = (total_length_m / 1000.0) / area_km2 if area_km2 else 0.0

    return {
        "source_endpoint": endpoint,
        "radius_m": radius_m,
        "waterway_count": len(elements),
        "waterway_types": by_type,
        "total_waterway_length_km": round(total_length_m / 1000.0, 4),
        "distance_to_drain_m_approx": round(min_distance_m, 2) if min_distance_m is not None else None,
        "drainage_density_km_per_km2": round(drainage_density, 6),
        "note": "Distance is approximated from returned OSM geometry vertices."
    }
