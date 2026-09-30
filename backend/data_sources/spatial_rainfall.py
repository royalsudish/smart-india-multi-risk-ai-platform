"""
Open-Meteo Spatial Rainfall Intelligence (Phase E Replacement)
Provides contextual spatial precipitation intelligence derived from
Open-Meteo's recent hourly numerical weather model data.

Terminology: SPATIAL RAINFALL
Provider: Open-Meteo
Source Type: recent_hourly_model_data
Provider/Source Label: Open-Meteo Recent Hourly Model Data
Data Semantics: recently archived numerical weather forecast/model values exposed by the Open-Meteo Forecast API

Preserves 100% model invariance for all 4 ML models.
Dedicated asynchronous non-blocking endpoint: GET /api/waterlogging/rainfall/spatial
"""

import math
import time
from datetime import datetime, timezone
import zoneinfo
from typing import Dict, Any, List, Optional, Tuple

import requests

from backend.config.settings import (
    APP_USER_AGENT,
    REQUEST_TIMEOUT_SECONDS,
    OPEN_METEO_SPATIAL_URL,
)
from backend.utils.cache import (
    provider_cache,
    make_cache_key,
    TTL_SPATIAL_RAINFALL,
)

PROVIDER_NAME = "Open-Meteo"
SOURCE_TYPE = "recent_hourly_model_data"
SOURCE_LABEL = "Open-Meteo Recent Hourly Model Data"
DATA_SEMANTICS = "recently archived numerical weather forecast/model values exposed by the Open-Meteo Forecast API"
COVERAGE_THRESHOLD = 0.75  # 75% completeness requirement
REQUESTED_SAMPLING_SPACING_DEG = 0.1
GRID_DELTA_DEG = 0.5
TIMEZONE_INDIA = "Asia/Kolkata"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(6371.0 * c, 3)


def get_spatial_grid_coordinates(
    center_lat: float,
    center_lon: float,
    delta_deg: float = GRID_DELTA_DEG,
    step_deg: float = REQUESTED_SAMPLING_SPACING_DEG
) -> Dict[str, Any]:
    """
    Generates an 11x11 sampling grid (121 points) centered on (center_lat, center_lon)
    with requested sampling spacing of step_deg (default 0.1 degree, ~11 km spacing).
    This sampling spacing is chosen by this application and may differ from the
    underlying weather model's native resolution.
    """
    clamped_lat = max(-89.5, min(89.5, float(center_lat)))
    clamped_lon = max(-179.5, min(179.5, float(center_lon)))

    # Compute coordinate arrays: 11 points in lat, 11 points in lon
    steps = int(round(delta_deg / step_deg))
    lat_coords = [round(clamped_lat + i * step_deg, 4) for i in range(-steps, steps + 1)]
    lon_coords = [round(clamped_lon + j * step_deg, 4) for j in range(-steps, steps + 1)]

    # Actual bounding box: [min_lon, min_lat, max_lon, max_lat]
    actual_bbox = [
        round(min(lon_coords) - step_deg / 2.0, 4),
        round(min(lat_coords) - step_deg / 2.0, 4),
        round(max(lon_coords) + step_deg / 2.0, 4),
        round(max(lat_coords) + step_deg / 2.0, 4),
    ]

    # Flatten coordinates in row-major order (lat outer, lon inner)
    flat_lats: List[float] = []
    flat_lons: List[float] = []
    for r_lat in lat_coords:
        for c_lon in lon_coords:
            flat_lats.append(r_lat)
            flat_lons.append(c_lon)

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "delta_deg": delta_deg,
        "step_deg": step_deg,
        "lat_coords": lat_coords,
        "lon_coords": lon_coords,
        "grid_rows": len(lat_coords),
        "grid_columns": len(lon_coords),
        "cell_count": len(flat_lats),
        "actual_bbox": actual_bbox,
        "flat_lats": flat_lats,
        "flat_lons": flat_lons,
    }


def generate_spatial_geojson(
    grid_info: Dict[str, Any],
    grid_24h_values: List[List[Optional[float]]],
    coordinate_mappings: List[Dict[str, Any]],
    start_time: str,
    end_time: str
) -> Dict[str, Any]:
    """
    Builds an RFC 7946 GeoJSON FeatureCollection of visualization sampling cells.
    The polygons are visualization/sample cells constructed around the requested
    sampling points; they are NOT native Open-Meteo model grid-cell boundaries.
    Coordinates are strictly [longitude, latitude] with closed polygon rings.
    """
    lat_coords = grid_info["lat_coords"]
    lon_coords = grid_info["lon_coords"]
    step = grid_info["step_deg"] / 2.0
    grid_cols = grid_info["grid_columns"]

    features = []
    for r_idx, lat_c in enumerate(lat_coords):
        for c_idx, lon_c in enumerate(lon_coords):
            val = None
            if r_idx < len(grid_24h_values) and c_idx < len(grid_24h_values[r_idx]):
                val = grid_24h_values[r_idx][c_idx]

            idx = r_idx * grid_cols + c_idx
            mapping = coordinate_mappings[idx] if idx < len(coordinate_mappings) else {}

            w = round(lon_c - step, 4)
            e = round(lon_c + step, 4)
            s = round(lat_c - step, 4)
            n = round(lat_c + step, 4)

            # Closed polygon ring: [lon, lat]
            polygon_coords = [
                [[w, s], [e, s], [e, n], [w, n], [w, s]]
            ]

            cell_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords,
                },
                "properties": {
                    "sampling_center_lat": lat_c,
                    "sampling_center_lon": lon_c,
                    "provider_returned_lat": mapping.get("returned_lat", lat_c),
                    "provider_returned_lon": mapping.get("returned_lon", lon_c),
                    "precipitation_mm": val,
                    "rainfall_24h_mm": val,
                    "status": "valid" if val is not None else "missing",
                    "source": PROVIDER_NAME,
                    "source_type": SOURCE_TYPE,
                    "time_window": "rolling_24h",
                    "geometry_type": "visualization_sampling_cell",
                    "sampling_spacing_deg": REQUESTED_SAMPLING_SPACING_DEG,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            }
            features.append(cell_feature)

    return {
        "type": "FeatureCollection",
        "bbox": grid_info["actual_bbox"],
        "properties": {
            "provider": PROVIDER_NAME,
            "source_type": SOURCE_TYPE,
            "data_semantics": DATA_SEMANTICS,
            "time_window": "rolling_24h",
            "geometry_type": "visualization_sampling_cell",
            "sampling_spacing_deg": REQUESTED_SAMPLING_SPACING_DEG,
            "start_time": start_time,
            "end_time": end_time,
            "grid_rows": grid_info["grid_rows"],
            "grid_columns": grid_info["grid_columns"],
            "cell_count": len(features),
        },
        "features": features,
    }


def aggregate_spatial_rainfall(
    lat: float,
    lon: float,
    delta_deg: float = GRID_DELTA_DEG
) -> Dict[str, Any]:
    """
    Main entry point for Phase E Open-Meteo Spatial Rainfall Intelligence.
    Asynchronously callable, isolated from core ML prediction pipelines.

    Computes:
      - Temporal accumulation windows for selected point (1h, 3h, 6h, 12h, 24h, rolling_72h, rolling_168h)
        directly summing hourly precipitation in mm.
      - 11x11 sampling grid (121 points) rolling 24h rainfall using standardized
        comma-separated multi-location request.
      - RFC 7946 GeoJSON FeatureCollection of visualization sampling cells.
      - Mapping of requested vs returned coordinates.
      - Coverage diagnostics, deterministic SHA-256 caching, and telemetry.
    """
    t_start = time.perf_counter()

    # 1. Grid Definition
    grid_info = get_spatial_grid_coordinates(lat, lon, delta_deg=delta_deg)
    grid_rows = grid_info["grid_rows"]
    grid_cols = grid_info["grid_columns"]
    cell_count = grid_info["cell_count"]

    # 2. Deterministic Cache Key
    cache_params = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "delta": delta_deg,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "sampling_spacing": REQUESTED_SAMPLING_SPACING_DEG,
    }
    cache_key = make_cache_key("spatial_rainfall", lat, lon, cache_params)
    cached = provider_cache.get(cache_key)
    if cached is not None:
        cached["cache_status"] = "hit"
        cached["execution_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return cached

    headers = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}

    # 3. Standardized Multi-Location Request (comma-separated lists)
    # Open-Meteo officially documents multiple locations as comma-separated latitude/longitude lists:
    # latitude=lat1,lat2,lat3,...&longitude=lon1,lon2,lon3,...
    lat_str = ",".join(f"{coord:.4f}" for coord in grid_info["flat_lats"])
    lon_str = ",".join(f"{coord:.4f}" for coord in grid_info["flat_lons"])

    batch_params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "hourly": "precipitation,rain,showers",
        "past_days": 7,
        "forecast_days": 1,
        "precipitation_unit": "mm",
        "timezone": TIMEZONE_INDIA,
    }

    batch_resp = None
    provider_failed = False

    # Attempt fetch with retry and 429 backoff
    for attempt in range(2):
        try:
            r = requests.get(
                OPEN_METEO_SPATIAL_URL,
                params=batch_params,
                headers=headers,
                timeout=min(15.0, REQUEST_TIMEOUT_SECONDS)
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) == cell_count:
                    batch_resp = data
                    break
            elif r.status_code == 429:
                time.sleep(2.5)
        except Exception:
            if attempt == 1:
                provider_failed = True
            time.sleep(0.5)

    if batch_resp is None:
        provider_failed = True

    # 4. Parse time indices & build requested vs returned coordinate mappings
    tz_ist = zoneinfo.ZoneInfo(TIMEZONE_INDIA)
    now_ist = datetime.now(tz_ist)

    # Initialize grid values to None (null)
    grid_24h_values: List[List[Optional[float]]] = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]

    center_hourly_precip: List[Optional[float]] = []
    center_hourly_times: List[str] = []
    past_indices: List[int] = []

    # Build explicit mapping between requested coordinates and returned locations
    coordinate_mappings: List[Dict[str, Any]] = []
    
    # The center point is index 60 in an 11x11 grid (middle row and column)
    center_idx = (grid_rows // 2) * grid_cols + (grid_cols // 2)

    if not provider_failed and batch_resp:
        for idx, loc_obj in enumerate(batch_resp):
            r_idx = idx // grid_cols
            c_idx = idx % grid_cols
            req_lat = grid_info["flat_lats"][idx]
            req_lon = grid_info["flat_lons"][idx]
            ret_lat = round(loc_obj.get("latitude", req_lat), 4)
            ret_lon = round(loc_obj.get("longitude", req_lon), 4)

            coordinate_mappings.append({
                "grid_index": idx,
                "row": r_idx,
                "col": c_idx,
                "requested_lat": req_lat,
                "requested_lon": req_lon,
                "returned_lat": ret_lat,
                "returned_lon": ret_lon,
                "coordinates_exact_match": (ret_lat == req_lat and ret_lon == req_lon),
            })

        # Extract times from center point
        center_loc = batch_resp[center_idx]
        h_dict = center_loc.get("hourly", {})
        center_hourly_times = h_dict.get("time", [])

        # Filter indices corresponding to past hours (<= current time)
        for i, t_str in enumerate(center_hourly_times):
            try:
                dt = datetime.fromisoformat(t_str).replace(tzinfo=tz_ist)
                if dt <= now_ist:
                    past_indices.append(i)
            except Exception:
                pass

        # If parsing by time produced too few past hours, default to past 7 days (168 hours)
        if len(past_indices) < 24 and len(center_hourly_times) >= 24:
            past_indices = list(range(min(len(center_hourly_times), 168)))

        # Extract center precipitation for past hours
        all_precip = h_dict.get("precipitation", [])
        center_hourly_precip = [
            all_precip[i] if i < len(all_precip) else None
            for i in past_indices
        ]

        # Compute rolling 24h rainfall for each grid point
        past_24_indices = past_indices[-24:] if len(past_indices) >= 24 else past_indices
        for idx, loc_obj in enumerate(batch_resp):
            r_idx = idx // grid_cols
            c_idx = idx % grid_cols
            loc_precips = loc_obj.get("hourly", {}).get("precipitation", [])
            
            slice_vals = [
                loc_precips[i] if (i < len(loc_precips) and loc_precips[i] is not None) else None
                for i in past_24_indices
            ]
            valid_vals = [v for v in slice_vals if v is not None]
            coverage = len(valid_vals) / 24.0

            if coverage >= COVERAGE_THRESHOLD:
                grid_24h_values[r_idx][c_idx] = max(0.0, round(sum(valid_vals), 2))
            else:
                grid_24h_values[r_idx][c_idx] = None
    else:
        # Generate dummy mappings for failure case
        for idx in range(cell_count):
            r_idx = idx // grid_cols
            c_idx = idx % grid_cols
            coordinate_mappings.append({
                "grid_index": idx,
                "row": r_idx,
                "col": c_idx,
                "requested_lat": grid_info["flat_lats"][idx],
                "requested_lon": grid_info["flat_lons"][idx],
                "returned_lat": None,
                "returned_lon": None,
                "coordinates_exact_match": False,
            })

    # 5. Calculate Temporal Windows for Selected (Center) Point
    # Explicit rolling window semantics:
    # 1h: rolling_1h
    # 3h: rolling_3h
    # 6h: rolling_6h
    # 12h: rolling_12h
    # 24h: rolling_24h
    # 3d: rolling_72h ("Sum of the most recent 72 completed hourly precipitation values")
    # 7d: rolling_168h ("Sum of the most recent 168 completed hourly precipitation values")
    def compute_window(hours_count: int, semantics: str, description: str) -> Dict[str, Any]:
        if provider_failed or not center_hourly_precip:
            return {
                "value_mm": None,
                "unit": "mm",
                "status": "provider_failure" if provider_failed else "unavailable",
                "window_semantics": semantics,
                "description": description,
                "expected_count": hours_count,
                "valid_count": 0,
                "missing_count": hours_count,
                "coverage_fraction": 0.0,
                "expected_hours": hours_count,
                "valid_hours": 0,
                "missing_hours": hours_count,
            }

        slice_vals = center_hourly_precip[-hours_count:]
        valid_vals = [v for v in slice_vals if v is not None]
        valid_count = len(valid_vals)
        missing_count = max(0, hours_count - valid_count)
        coverage_fraction = round(valid_count / float(hours_count), 3)

        if coverage_fraction < COVERAGE_THRESHOLD:
            return {
                "value_mm": None,
                "unit": "mm",
                "status": "insufficient_coverage",
                "window_semantics": semantics,
                "description": description,
                "expected_count": hours_count,
                "valid_count": valid_count,
                "missing_count": missing_count,
                "coverage_fraction": coverage_fraction,
                "expected_hours": hours_count,
                "valid_hours": valid_count,
                "missing_hours": missing_count,
            }

        accum_mm = round(sum(valid_vals), 2)
        return {
            "value_mm": max(0.0, accum_mm),
            "unit": "mm",
            "status": "available",
            "window_semantics": semantics,
            "description": description,
            "expected_count": hours_count,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "coverage_fraction": coverage_fraction,
            "expected_hours": hours_count,
            "valid_hours": valid_count,
            "missing_hours": missing_count,
        }

    windows = {
        "1h": compute_window(1, "rolling_1h", "Sum of the most recent 1 completed hourly precipitation value"),
        "3h": compute_window(3, "rolling_3h", "Sum of the most recent 3 completed hourly precipitation values"),
        "6h": compute_window(6, "rolling_6h", "Sum of the most recent 6 completed hourly precipitation values"),
        "12h": compute_window(12, "rolling_12h", "Sum of the most recent 12 completed hourly precipitation values"),
        "24h": compute_window(24, "rolling_24h", "Sum of the most recent 24 completed hourly precipitation values"),
        "3d": compute_window(72, "rolling_72h", "Sum of the most recent 72 completed hourly precipitation values"),
        "7d": compute_window(168, "rolling_168h", "Sum of the most recent 168 completed hourly precipitation values"),
    }

    # 6. GeoJSON Assembly (visualization sampling cells)
    start_time_iso = center_hourly_times[past_indices[-24]] if len(past_indices) >= 24 else ""
    end_time_iso = center_hourly_times[past_indices[-1]] if len(past_indices) >= 1 else ""

    geojson = generate_spatial_geojson(
        grid_info,
        grid_24h_values,
        coordinate_mappings=coordinate_mappings,
        start_time=start_time_iso,
        end_time=end_time_iso
    )

    valid_cells = sum(
        1 for r in grid_24h_values for v in r if v is not None
    )

    duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
    overall_status = "available" if (not provider_failed and valid_cells > 0) else "unavailable"

    coverage_fraction = round(valid_cells / float(cell_count), 3)

    # Sample coordinate mappings (center, 4 corners)
    sample_indices = [
        0,                             # NW corner
        grid_cols - 1,                 # NE corner
        center_idx,                    # Center
        (grid_rows - 1) * grid_cols,   # SW corner
        cell_count - 1                 # SE corner
    ]
    sample_mappings = [coordinate_mappings[i] for i in sample_indices if i < len(coordinate_mappings)]

    res = {
        "status": overall_status,
        "provider": PROVIDER_NAME,
        "source_type": SOURCE_TYPE,
        "source_label": SOURCE_LABEL,
        "data_semantics": DATA_SEMANTICS,
        "analysis_time": datetime.now(tz_ist).isoformat(),
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "selected_location": {
            "selected_lat": lat,
            "selected_lon": lon,
            "requested_sampling_spacing_deg": REQUESTED_SAMPLING_SPACING_DEG,
            "requested_grid_dimensions": f"{grid_rows} x {grid_cols}",
            "requested_span_deg": delta_deg,
            "grid_rows": grid_rows,
            "grid_columns": grid_cols,
        },
        "windows": windows,
        "grid": {
            "requested_sampling_spacing_deg": REQUESTED_SAMPLING_SPACING_DEG,
            "requested_grid_dimensions": f"{grid_rows} x {grid_cols}",
            "geometry_type": "visualization_sampling_cell",
            "actual_bbox": grid_info["actual_bbox"],
            "grid_rows": grid_rows,
            "grid_columns": grid_cols,
            "cell_count": cell_count,
            "valid_cells_count": valid_cells,
            "missing_cells_count": cell_count - valid_cells,
            "sample_coordinate_mappings": sample_mappings,
            "grid_values_24h": grid_24h_values,
            "geojson": geojson,
        },
        "geojson": geojson,
        "coverage": {
            "expected_count": cell_count,
            "valid_count": valid_cells,
            "missing_count": cell_count - valid_cells,
            "coverage_fraction": coverage_fraction,
            "threshold_required": COVERAGE_THRESHOLD,
            "status": "complete" if valid_cells == cell_count else ("partial" if valid_cells > 0 else "insufficient_coverage"),
        },
        "cache": {
            "status": "miss",
            "ttl_seconds": TTL_SPATIAL_RAINFALL,
            "execution_ms": duration_ms,
        },
        "limitations": [
            "The map samples Open-Meteo at approximately 0.1-degree spacing.",
            "This sampling spacing is chosen by this application.",
            "It is not necessarily the native resolution of the weather model.",
            "Open-Meteo Best Match may use different underlying weather models.",
            "Values are numerical weather model data, not local rain-gauge measurements, radar observations, or independent satellite observations.",
            "Polygons are visualization sampling cells constructed around requested sampling points, not native model grid-cell boundaries.",
        ],
        "cache_status": "miss",
        "execution_ms": duration_ms,
    }

    if overall_status == "available":
        provider_cache.set(cache_key, res, ttl=TTL_SPATIAL_RAINFALL)

    return res
