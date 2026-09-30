import math
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from backend.config.settings import OVERPASS_ENDPOINTS
from backend.data_sources.http_client import post_form_json
from backend.utils.cache import provider_cache

EARTH_RADIUS_M = 6371000.0

# ==============================================================================
# POLICY-DEFINED DRAINAGE PROXIMITY TIERS (INFORMATIONAL UI ONLY)
# ==============================================================================
# NOTE: These tiers are purely descriptive UI classifications based on distance
# to the nearest mapped OSM waterway. They are NOT ML thresholds or model outputs.
# ==============================================================================
DRAINAGE_PROXIMITY_POLICY = [
    {
        "max_dist_m": 250.0,
        "badge_text": "IMMEDIATE (< 250m)",
        "badge_class": "badge-info",
        "description": "Adjacent or in immediate proximity to a mapped drainage channel.",
        "policy_type": "POLICY-DEFINED PROXIMITY TIERS (INFORMATIONAL ONLY)",
    },
    {
        "max_dist_m": 750.0,
        "badge_text": "MODERATE (250m–750m)",
        "badge_class": "badge-neutral",
        "description": "Within reasonable overland flow reach of a mapped drainage channel.",
        "policy_type": "POLICY-DEFINED PROXIMITY TIERS (INFORMATIONAL ONLY)",
    },
    {
        "max_dist_m": float("inf"),
        "badge_text": "DISTANT (> 750m)",
        "badge_class": "badge-warning",
        "description": "Remote from any mapped drainage channel within the analysis radius.",
        "policy_type": "POLICY-DEFINED PROXIMITY TIERS (INFORMATIONAL ONLY)",
    },
]


def evaluate_drainage_proximity(nearest_dist_m: Optional[float]) -> Dict[str, str]:
    if nearest_dist_m is None:
        return {
            "badge_text": "NO MAPPED OSM WATERWAYS IN RADIUS",
            "badge_class": "badge-neutral",
            "description": "No mapped OSM waterway features were returned within the analysis radius. This does not prove physical drainage is absent.",
            "policy_type": "POLICY-DEFINED PROXIMITY TIERS (INFORMATIONAL ONLY)",
        }
    val = float(nearest_dist_m)
    for tier in DRAINAGE_PROXIMITY_POLICY:
        if val < tier["max_dist_m"]:
            return {
                "badge_text": tier["badge_text"],
                "badge_class": tier["badge_class"],
                "description": tier["description"],
                "policy_type": tier["policy_type"],
            }
    last = DRAINAGE_PROXIMITY_POLICY[-1]
    return {
        "badge_text": last["badge_text"],
        "badge_class": last["badge_class"],
        "description": last["description"],
        "policy_type": last["policy_type"],
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on the sphere in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _to_local_meters(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Projects (lat, lon) to local metric coordinates (x, y) relative to (lat0, lon0)
    using an Equirectangular projection. Accurate to < 0.1% for distances under 10 km.
    """
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def _to_lat_lon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Converts local metric coordinates (x, y) back to (lat, lon)."""
    lat = lat0 + math.degrees(y / EARTH_RADIUS_M)
    cos_lat0 = math.cos(math.radians(lat0))
    lon = lon0 + math.degrees(x / (EARTH_RADIUS_M * cos_lat0)) if abs(cos_lat0) > 1e-7 else lon0
    return lat, lon


def point_to_segment_metric(
    x1: float, y1: float, x2: float, y2: float, lat0: float, lon0: float
) -> Tuple[float, float, float]:
    """
    Calculates the true perpendicular point-to-segment distance in local metric space
    from origin (0, 0) to segment (x1, y1) -> (x2, y2).
    Returns: (distance_m, closest_lat, closest_lon)
    """
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        dist = math.hypot(x1, y1)
        clat, clon = _to_lat_lon(x1, y1, lat0, lon0)
        return dist, clat, clon

    # Project origin (0, 0) onto line segment
    t = -(x1 * dx + y1 * dy) / l2
    t_clamped = max(0.0, min(1.0, t))
    xc = x1 + t_clamped * dx
    yc = y1 + t_clamped * dy
    dist = math.hypot(xc, yc)
    clat, clon = _to_lat_lon(xc, yc, lat0, lon0)
    return dist, clat, clon


def clip_segment_to_circle(x1: float, y1: float, x2: float, y2: float, radius_m: float) -> float:
    """
    Calculates the exact length of segment (x1, y1) -> (x2, y2) that lies inside
    the circle x^2 + y^2 <= radius_m^2.
    """
    dx = x2 - x1
    dy = y2 - y1
    alpha = dx * dx + dy * dy
    if alpha <= 1e-9:
        return 0.0 if math.hypot(x1, y1) > radius_m else 0.0

    beta = 2.0 * (x1 * dx + y1 * dy)
    gamma = x1 * x1 + y1 * y1 - radius_m * radius_m

    delta = beta * beta - 4.0 * alpha * gamma
    if delta <= 0.0:
        return 0.0

    sqrt_delta = math.sqrt(delta)
    t1 = (-beta - sqrt_delta) / (2.0 * alpha)
    t2 = (-beta + sqrt_delta) / (2.0 * alpha)

    t_start = max(0.0, min(t1, t2))
    t_end = min(1.0, max(t1, t2))

    if t_start < t_end:
        return (t_end - t_start) * math.sqrt(alpha)
    return 0.0


def _build_deterministic_cache_key(lat: float, lon: float, radius_m: int) -> str:
    """
    Builds a deterministic cache key preserving exact floating-point coordinate precision.
    Uses repr(float(coord)) to avoid decimal rounding/truncation.
    Serializes parameters canonically with sorted keys and SHA-256 digest.
    """
    lat_key = repr(float(lat))
    lon_key = repr(float(lon))
    param_dict = {
        "radius_m": int(radius_m),
        "waterways": ["canal", "ditch", "drain", "river", "stream"],
    }
    serialized_params = json.dumps(param_dict, sort_keys=True)
    param_hash = hashlib.sha256(serialized_params.encode("utf-8")).hexdigest()[:16]
    return f"overpass:drainage:{lat_key}:{lon_key}:{int(radius_m)}:{param_hash}"


def _run_overpass_query(query: str, per_attempt_timeout: float = 8.0) -> Tuple[Dict[str, Any], str]:
    """
    Executes Overpass QL query with bounded per-attempt timeout and multi-endpoint failover.
    Total wait time is bounded: (per_attempt_timeout * len(OVERPASS_ENDPOINTS)).
    """
    errors = []
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            res = post_form_json(endpoint, {"data": query}, timeout=per_attempt_timeout)
            if isinstance(res, dict) and "elements" in res:
                return res, endpoint
            errors.append(f"{endpoint}: Invalid response structure")
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))


def drainage_features(lat: float, lon: float, radius_m: int = 2000, use_cache: bool = True) -> Dict[str, Any]:
    """
    Retrieves and analyzes OpenStreetMap waterways within radius_m of (lat, lon).
    - True point-to-polyline metric distance.
    - Geometry clipping to circular analysis buffer.
    - Mapped drainage network density.
    - GeoJSON FeatureCollection with [lon, lat] coordinate order.
    - No-mapped-data semantics.
    - Preserves exact coordinates without rounding.
    """
    lat = float(lat)
    lon = float(lon)
    radius_m = max(200, min(int(radius_m), 10000))
    area_km2 = math.pi * (radius_m / 1000.0) ** 2

    cache_key = _build_deterministic_cache_key(lat, lon, radius_m)

    if use_cache:
        cached = provider_cache.get(cache_key)
        if cached is not None:
            cached_copy = dict(cached)
            cached_copy["cache_hit"] = True
            return cached_copy

    t0 = time.time()
    q = f"""
    [out:json][timeout:15];
    (
      way["waterway"~"river|stream|canal|drain|ditch"](around:{radius_m},{lat},{lon});
    );
    out tags geom;
    """

    try:
        data, endpoint = _run_overpass_query(q, per_attempt_timeout=8.0)
    except Exception as exc:
        unavailable_result = {
            "status": "unavailable",
            "message": f"Overpass drainage service temporarily unavailable: {exc}",
            "cache_hit": False,
            "cache_key": cache_key,
            "analysis_radius_m": radius_m,
            "analysis_area_km2": round(area_km2, 4),
            "waterway_count": None,
            "total_waterway_length_km": None,
            "total_unclipped_length_km": None,
            "nearest_drain_distance_m": None,
            "nearest_drain_name": None,
            "nearest_drain_type": None,
            "nearest_drain_coords": None,
            "drainage_density_km_per_km2": None,
            "waterway_types": {},
            "proximity_advisory": {
                "badge_text": "DRAINAGE DATA UNAVAILABLE",
                "badge_class": "badge-warning",
                "description": "OpenStreetMap drainage data could not be retrieved at this time.",
                "policy_type": "PROVIDER_STATUS",
            },
            "source": "OpenStreetMap via Overpass API",
            "source_endpoint": None,
            "error": str(exc),
            "note": "OpenStreetMap drainage data could not be retrieved at this time. All core ML prediction models continue operating normally.",
            "query_latency_ms": round((time.time() - t0) * 1000, 2),
            "geojson": {
                "type": "FeatureCollection",
                "features": []
            }
        }
        return unavailable_result

    elements = data.get("elements", [])

    if not elements:
        no_data_result = {
            "status": "no_mapped_data_in_radius",
            "message": f"No mapped OSM waterways found within {radius_m}m radius.",
            "cache_hit": False,
            "cache_key": cache_key,
            "analysis_radius_m": radius_m,
            "analysis_area_km2": round(area_km2, 4),
            "waterway_count": 0,
            "total_waterway_length_km": 0.0,
            "total_unclipped_length_km": 0.0,
            "nearest_drain_distance_m": None,
            "nearest_drain_name": None,
            "nearest_drain_type": None,
            "nearest_drain_coords": None,
            "drainage_density_km_per_km2": 0.0,
            "waterway_types": {},
            "proximity_advisory": evaluate_drainage_proximity(None),
            "source": "OpenStreetMap via Overpass API",
            "source_endpoint": endpoint,
            "note": "No mapped OSM waterway features were returned within the analysis radius. This does not prove physical drainage is absent.",
            "query_latency_ms": round((time.time() - t0) * 1000, 2),
            "geojson": {
                "type": "FeatureCollection",
                "features": []
            }
        }
        if use_cache:
            provider_cache.set(cache_key, no_data_result, ttl=86400.0)
        return no_data_result

    total_unclipped_length_m = 0.0
    total_clipped_length_m = 0.0
    min_distance_m: Optional[float] = None
    nearest_drain_type: Optional[str] = None
    nearest_drain_name: Optional[str] = None
    nearest_drain_coords: Optional[List[float]] = None
    by_type: Dict[str, int] = {}
    geojson_features: List[Dict[str, Any]] = []

    for el in elements:
        tags = el.get("tags") or {}
        kind = tags.get("waterway", "unknown")
        name = tags.get("name") or tags.get("name:en") or f"Unnamed {kind.capitalize()}"
        by_type[kind] = by_type.get(kind, 0) + 1

        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue

        way_unclipped_m = 0.0
        way_clipped_m = 0.0
        geojson_coords: List[List[float]] = []

        # Convert geometry to local metric coordinates
        local_pts = []
        for pt in geom:
            p_lat, p_lon = pt["lat"], pt["lon"]
            # GeoJSON standard coordinate order is [longitude, latitude]
            geojson_coords.append([p_lon, p_lat])
            x, y = _to_local_meters(p_lat, p_lon, lat, lon)
            local_pts.append((x, y))

        for i in range(len(local_pts) - 1):
            x1, y1 = local_pts[i]
            x2, y2 = local_pts[i + 1]

            # 1. Unclipped length
            seg_len = math.hypot(x2 - x1, y2 - y1)
            way_unclipped_m += seg_len

            # 2. Clipped length to circular buffer
            clipped_len = clip_segment_to_circle(x1, y1, x2, y2, radius_m)
            way_clipped_m += clipped_len

            # 3. True point-to-polyline metric distance
            d_seg, c_lat, c_lon = point_to_segment_metric(x1, y1, x2, y2, lat, lon)
            if min_distance_m is None or d_seg < min_distance_m:
                min_distance_m = d_seg
                nearest_drain_type = kind
                nearest_drain_name = name
                nearest_drain_coords = [round(c_lat, 6), round(c_lon, 6)]

        total_unclipped_length_m += way_unclipped_m
        total_clipped_length_m += way_clipped_m

        # Build GeoJSON LineString Feature (strictly [lon, lat])
        geojson_features.append({
            "type": "Feature",
            "id": el.get("id"),
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_coords,
            },
            "properties": {
                "osm_id": el.get("id"),
                "waterway": kind,
                "name": name,
                "unclipped_length_m": round(way_unclipped_m, 2),
                "in_buffer_length_m": round(way_clipped_m, 2),
            }
        })

    # Drainage density uses strictly the in-buffer clipped length
    drainage_density = (total_clipped_length_m / 1000.0) / area_km2 if area_km2 > 0 else 0.0

    result = {
        "status": "ok",
        "cache_hit": False,
        "cache_key": cache_key,
        "analysis_radius_m": radius_m,
        "analysis_area_km2": round(area_km2, 4),
        "waterway_count": len(elements),
        "waterway_types": by_type,
        "total_waterway_length_km": round(total_clipped_length_m / 1000.0, 4),
        "total_unclipped_length_km": round(total_unclipped_length_m / 1000.0, 4),
        "nearest_drain_distance_m": round(min_distance_m, 2) if min_distance_m is not None else None,
        "nearest_drain_name": nearest_drain_name,
        "nearest_drain_type": nearest_drain_type,
        "nearest_drain_coords": nearest_drain_coords,
        "drainage_density_km_per_km2": round(drainage_density, 6),
        "proximity_advisory": evaluate_drainage_proximity(min_distance_m),
        "source": "OpenStreetMap via Overpass API",
        "source_endpoint": endpoint,
        "note": "Mapped Drainage Network Density reflects mapped geometry in OpenStreetMap within the buffer. It does not represent hydraulic capacity, maintenance state, or underground pipe dimensions.",
        "query_latency_ms": round((time.time() - t0) * 1000, 2),
        "geojson": {
            "type": "FeatureCollection",
            "features": geojson_features,
        }
    }

    if use_cache:
        provider_cache.set(cache_key, result, ttl=86400.0)

    return result
