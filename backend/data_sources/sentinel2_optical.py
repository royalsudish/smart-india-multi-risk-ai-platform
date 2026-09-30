"""
Sentinel-2 L2A Optical Satellite Data Source
Provides live surface reflectance indices (NDVI and NDBI) derived from 
Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) Cloud-Optimized GeoTIFFs (COGs).

Data Pipelines:
  - Primary Provider: Element84 Earth Search (AWS Open Data STAC)
  - Secondary Provider: Microsoft Planetary Computer STAC (SAS-signed)
  - Bands utilized:
      * B04 = Red (10m resolution)
      * B08 = Near-Infrared (NIR, 10m resolution)
      * B11 = Short-Wave Infrared (SWIR-1 / 1.6µm, 20m resolution)
      * SCL = Scene Classification Layer (20m resolution QA mask)

Formulas:
  - NDVI = (B08 - B04) / (B08 + B04)  [Vegetation Index: -1.0 to 1.0]
  - NDBI = (B11 - B08) / (B11 + B08)  [Built-up Index: -1.0 to 1.0]

Quality & Cloud Filtering:
  - SCL Scene Classification Layer cloud & shadow masking:
      Rejects: 0 (No data), 1 (Defective), 3 (Cloud shadow), 8 (Cloud med), 9 (Cloud high), 10 (Cirrus)
      Retains: 4 (Vegetation), 5 (Bare soil / Non-vegetated), 6 (Water), 7 (Unclassified surface)
  - Exact pixel with neighborhood valid median fallback when center pixel is cloud-obscured.
"""

import os
import math
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling

from backend.config.settings import APP_USER_AGENT, REQUEST_TIMEOUT_SECONDS
from backend.utils.cache import provider_cache, make_cache_key
from backend.data_sources.sentinel1_sar import sign_planetary_computer_asset

# Configure GDAL environment for fast HTTP range requests
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "10000000")

# STAC Endpoints
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1/search"
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# Cache TTLs (in seconds)
TTL_SCENE_SEARCH = 3600.0       # 1 hour
TTL_OPTICAL_INDICES = 43200.0   # 12 hours (Sentinel-2 revisit is 5 days)

# SCL Mask Definitions
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}  # No data, defective, cloud shadow, med/high cloud, cirrus
SCL_VALID_CLASSES = {2, 4, 5, 6, 7, 11}    # Dark surface, vegetation, bare soil, water, unclassified, snow

# Concurrency lock
_s2_lock = threading.Lock()


def search_sentinel2_scenes(
    lat: float,
    lon: float,
    max_cloud_cover: float = 60.0,
    days_back: int = 90,
    limit: int = 5,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Searches for recent, low-cloud Sentinel-2 L2A scenes intersecting the coordinate.
    Uses Earth Search (AWS) first, falling back to Planetary Computer if unreachable.
    Returns (scenes_list, provider_name).
    """
    cache_params = {
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "max_cloud": max_cloud_cover,
        "days_back": days_back,
    }
    cache_key = make_cache_key("s2_stac_search", lat, lon, cache_params)
    cached = provider_cache.get(cache_key)
    if cached and isinstance(cached, dict) and "scenes" in cached:
        return cached["scenes"], cached.get("provider", "cached")

    headers = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_back)
    time_range = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    point_geom = {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]}

    # 1. Primary Provider: Element84 Earth Search (AWS Open Data)
    payload_es = {
        "collections": ["sentinel-2-l2a"],
        "intersects": point_geom,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    try:
        r = requests.post(EARTH_SEARCH_STAC_URL, json=payload_es, headers=headers, timeout=min(12.0, REQUEST_TIMEOUT_SECONDS))
        if r.status_code == 200:
            features = r.json().get("features", [])
            valid_scenes = []
            for item in features:
                assets = item.get("assets", {})
                # Check for essential spectral bands
                has_red = "red" in assets or "B04" in assets
                has_nir = "nir" in assets or "B08" in assets
                has_swir = "swir16" in assets or "B11" in assets
                if has_red and has_nir and has_swir:
                    valid_scenes.append(item)

            if valid_scenes:
                provider_cache.set(cache_key, {"scenes": valid_scenes, "provider": "Earth Search (AWS)"}, ttl=TTL_SCENE_SEARCH)
                return valid_scenes, "Earth Search (AWS)"
    except Exception:
        pass

    # 2. Secondary Provider: Microsoft Planetary Computer
    payload_pc = {
        "collections": ["sentinel-2-l2a"],
        "intersects": point_geom,
        "datetime": time_range,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    try:
        r = requests.post(PLANETARY_COMPUTER_STAC_URL, json=payload_pc, headers=headers, timeout=min(12.0, REQUEST_TIMEOUT_SECONDS))
        if r.status_code == 200:
            features = r.json().get("features", [])
            valid_scenes = []
            for item in features:
                assets = item.get("assets", {})
                if "B04" in assets and "B08" in assets and "B11" in assets:
                    valid_scenes.append(item)

            if valid_scenes:
                provider_cache.set(cache_key, {"scenes": valid_scenes, "provider": "Microsoft Planetary Computer"}, ttl=TTL_SCENE_SEARCH)
                return valid_scenes, "Microsoft Planetary Computer"
    except Exception:
        pass

    return [], "none"


def _resolve_band_url(item: Dict[str, Any], band_key_candidates: List[str]) -> Optional[str]:
    """Finds and signs asset URL for the specified band."""
    assets = item.get("assets", {})
    for candidate in band_key_candidates:
        if candidate in assets:
            href = assets[candidate].get("href")
            if not href:
                continue
            # If Planetary Computer URL, sign it
            if "planetarycomputer.microsoft.com" in href or "blob.core.windows.net" in href:
                signed_url, _, _ = sign_planetary_computer_asset(href)
                return signed_url
            return href
    return None


def calculate_indices(
    b04_val: float,
    b08_val: float,
    b11_val: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes NDVI and NDBI with strict zero-denominator and nodata protection:
    NDVI = (B08 - B04) / (B08 + B04)
    NDBI = (B11 - B08) / (B11 + B08)
    Clamps valid results into [-1.0, 1.0].
    """
    if not (math.isfinite(b04_val) and math.isfinite(b08_val) and math.isfinite(b11_val)):
        return None, None

    if b04_val <= 0.0 or b08_val <= 0.0 or b11_val <= 0.0:
        return None, None

    # 1. NDVI
    denom_ndvi = b08_val + b04_val
    if abs(denom_ndvi) > 1e-6:
        raw_ndvi = (b08_val - b04_val) / denom_ndvi
        ndvi = max(-1.0, min(1.0, float(raw_ndvi)))
    else:
        ndvi = None

    # 2. NDBI
    denom_ndbi = b11_val + b08_val
    if abs(denom_ndbi) > 1e-6:
        raw_ndbi = (b11_val - b08_val) / denom_ndbi
        ndbi = max(-1.0, min(1.0, float(raw_ndbi)))
    else:
        ndbi = None

    return ndvi, ndbi


def extract_surface_indices_from_scene(
    scene_item: Dict[str, Any],
    lat: float,
    lon: float,
    buffer_deg: float = 0.002,  # ~220m neighborhood (~22x22 pixels at 10m)
) -> Optional[Dict[str, Any]]:
    """
    Opens Red (B04), NIR (B08), SWIR (B11), and SCL GeoTIFFs, samples a window
    around the coordinate, applies SCL cloud filtering, and computes median NDVI/NDBI.
    """
    url_b04 = _resolve_band_url(scene_item, ["red", "B04", "b04"])
    url_b08 = _resolve_band_url(scene_item, ["nir", "B08", "b08"])
    url_b11 = _resolve_band_url(scene_item, ["swir16", "B11", "b11"])
    url_scl = _resolve_band_url(scene_item, ["scl", "SCL", "scl-jp2"])

    if not (url_b04 and url_b08 and url_b11):
        return None

    min_lon = lon - buffer_deg
    min_lat = lat - buffer_deg
    max_lon = lon + buffer_deg
    max_lat = lat + buffer_deg

    try:
        with rasterio.open(url_b04) as src_b04:
            # Transform bounds to native CRS (e.g. UTM)
            proj_bounds = transform_bounds("EPSG:4326", src_b04.crs, min_lon, min_lat, max_lon, max_lat)
            win_b04 = from_bounds(*proj_bounds, transform=src_b04.transform)
            win_b04 = win_b04.intersection(rasterio.windows.Window(0, 0, src_b04.width, src_b04.height))

            if win_b04.width < 1 or win_b04.height < 1:
                return None

            shape_10m = (int(round(win_b04.height)), int(round(win_b04.width)))
            arr_b04 = src_b04.read(1, window=win_b04, out_shape=shape_10m, resampling=Resampling.bilinear).astype(np.float32)

        with rasterio.open(url_b08) as src_b08:
            win_b08 = from_bounds(*proj_bounds, transform=src_b08.transform)
            win_b08 = win_b08.intersection(rasterio.windows.Window(0, 0, src_b08.width, src_b08.height))
            arr_b08 = src_b08.read(1, window=win_b08, out_shape=shape_10m, resampling=Resampling.bilinear).astype(np.float32)

        with rasterio.open(url_b11) as src_b11:
            win_b11 = from_bounds(*proj_bounds, transform=src_b11.transform)
            win_b11 = win_b11.intersection(rasterio.windows.Window(0, 0, src_b11.width, src_b11.height))
            # Resample 20m SWIR to match 10m grid
            arr_b11 = src_b11.read(1, window=win_b11, out_shape=shape_10m, resampling=Resampling.bilinear).astype(np.float32)

        # SCL Scene Classification Layer (if available)
        arr_scl = None
        if url_scl:
            try:
                with rasterio.open(url_scl) as src_scl:
                    win_scl = from_bounds(*proj_bounds, transform=src_scl.transform)
                    win_scl = win_scl.intersection(rasterio.windows.Window(0, 0, src_scl.width, src_scl.height))
                    arr_scl = src_scl.read(1, window=win_scl, out_shape=shape_10m, resampling=Resampling.nearest)
            except Exception:
                arr_scl = None

        total_pixels = int(arr_b04.size)

        # Build validity and cloud mask
        valid_reflectance = (
            np.isfinite(arr_b04) & np.isfinite(arr_b08) & np.isfinite(arr_b11)
            & (arr_b04 > 0.0) & (arr_b08 > 0.0) & (arr_b11 > 0.0)
        )

        if arr_scl is not None:
            is_cloudy = np.isin(arr_scl, list(SCL_INVALID_CLASSES))
            valid_mask = valid_reflectance & (~is_cloudy)
        else:
            # Simple heuristic cloud reject when SCL is absent:
            # Clouds are very bright in both visible and NIR: B04 > 3000 and B08 > 3000
            is_cloudy = (arr_b04 > 3500.0) & (arr_b08 > 3500.0)
            valid_mask = valid_reflectance & (~is_cloudy)

        valid_count = int(np.sum(valid_mask))
        cloudy_count = int(np.sum(is_cloudy)) if is_cloudy is not None else 0

        # Reject if fewer than 5 valid surface pixels in window (scene is cloud covered at target)
        if valid_count < 5:
            return {
                "valid": False,
                "reason": "target_cloud_covered",
                "valid_pixels": valid_count,
                "total_pixels": total_pixels,
                "cloud_fraction": round(cloudy_count / max(1, total_pixels), 3),
            }

        # Vectorized NDVI & NDBI calculation
        denom_ndvi = arr_b08 + arr_b04
        denom_ndbi = arr_b11 + arr_b08

        ndvi_grid = np.where((denom_ndvi > 1e-6) & valid_mask, (arr_b08 - arr_b04) / denom_ndvi, np.nan)
        ndbi_grid = np.where((denom_ndbi > 1e-6) & valid_mask, (arr_b11 - arr_b08) / denom_ndbi, np.nan)

        clean_ndvi = ndvi_grid[np.isfinite(ndvi_grid)]
        clean_ndbi = ndbi_grid[np.isfinite(ndbi_grid)]

        # Clamping
        clean_ndvi = np.clip(clean_ndvi, -1.0, 1.0)
        clean_ndbi = np.clip(clean_ndbi, -1.0, 1.0)

        # Center pixel evaluation
        mid_r, mid_c = shape_10m[0] // 2, shape_10m[1] // 2
        center_valid = (
            bool(valid_mask[mid_r, mid_c])
            and np.isfinite(arr_b04[mid_r, mid_c])
            and np.isfinite(arr_b08[mid_r, mid_c])
            and np.isfinite(arr_b11[mid_r, mid_c])
            and (arr_b04[mid_r, mid_c] > 0.0)
            and (arr_b08[mid_r, mid_c] > 0.0)
            and (arr_b11[mid_r, mid_c] > 0.0)
        )

        if center_valid:
            b04_val = float(arr_b04[mid_r, mid_c])
            b08_val = float(arr_b08[mid_r, mid_c])
            b11_val = float(arr_b11[mid_r, mid_c])
            sampling_mode = "exact_pixel"
            pixel_or_roi = f"center_pixel ({lat:.4f}, {lon:.4f})"
        else:
            # Fallback to robust median of valid surface reflectance across cloud-free neighborhood
            b04_val = float(np.median(arr_b04[valid_mask]))
            b08_val = float(np.median(arr_b08[valid_mask]))
            b11_val = float(np.median(arr_b11[valid_mask]))
            sampling_mode = "neighborhood_median_cloud_fallback"
            pixel_or_roi = f"neighborhood_valid_median ({valid_count}/{total_pixels} valid pixels within ~220m)"

        bands_used = {
            "B04": round(b04_val, 1),
            "B08": round(b08_val, 1),
            "B11": round(b11_val, 1),
        }

        # Calculate exact formula indices from bands_used
        denom_ndvi = bands_used["B08"] + bands_used["B04"]
        denom_ndbi = bands_used["B11"] + bands_used["B08"]

        raw_ndvi = (bands_used["B08"] - bands_used["B04"]) / denom_ndvi if denom_ndvi > 1e-6 else 0.0
        raw_ndbi = (bands_used["B11"] - bands_used["B08"]) / denom_ndbi if denom_ndbi > 1e-6 else 0.0

        ndvi_val = max(-1.0, min(1.0, float(raw_ndvi)))
        ndbi_val = max(-1.0, min(1.0, float(raw_ndbi)))

        return {
            "valid": True,
            "ndvi": round(float(ndvi_val), 4),
            "ndbi": round(float(ndbi_val), 4),
            "valid_pixels": valid_count,
            "total_pixels": total_pixels,
            "coverage_pct": round((valid_count / max(1, total_pixels)) * 100.0, 1),
            "sampling_mode": sampling_mode,
            "center_pixel_valid": center_valid,
            "pixel_or_roi": pixel_or_roi,
            "bands_used": bands_used,
            "NDVI": {
                "value": float(ndvi_val),
                "formula_check": float(ndvi_val),
            },
            "NDBI": {
                "value": float(ndbi_val),
                "formula_check": float(ndbi_val),
            },
            "raw_bands_mean": {
                "b04_red": round(float(np.mean(arr_b04[valid_mask])), 1),
                "b08_nir": round(float(np.mean(arr_b08[valid_mask])), 1),
                "b11_swir": round(float(np.mean(arr_b11[valid_mask])), 1),
            },
        }
    except Exception as exc:
        return {"valid": False, "reason": f"raster_extraction_error: {exc}"}


def get_live_sentinel2_indices(
    lat: float,
    lon: float,
    max_cloud_cover: float = 60.0,
    days_back: int = 90,
) -> Dict[str, Any]:
    """
    Main entry point for Sentinel-2 L2A optical indices (NDVI and NDBI).
    Searches recent scenes, samples valid surface reflectance, handles cloud filtering,
    and returns standardized feature provenance metadata.
    """
    t_start = time.perf_counter()

    # 1. Check ROI Cache
    cache_params = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "max_cloud": max_cloud_cover,
    }
    cache_key = make_cache_key("s2_optical_roi", lat, lon, cache_params)
    cached = provider_cache.get(cache_key)
    if cached and isinstance(cached, dict):
        cached["cache_hit"] = True
        cached["execution_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        return cached

    with _s2_lock:
        # Check cache again inside lock
        cached = provider_cache.get(cache_key)
        if cached and isinstance(cached, dict):
            cached["cache_hit"] = True
            cached["execution_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return cached

        # 2. Discover Scenes
        scenes, provider_name = search_sentinel2_scenes(
            lat=lat,
            lon=lon,
            max_cloud_cover=max_cloud_cover,
            days_back=days_back,
            limit=5,
        )

        if not scenes:
            return {
                "status": "no_recent_scene",
                "provider": provider_name,
                "message": f"No Sentinel-2 L2A scene found with cloud cover < {max_cloud_cover}% in past {days_back} days.",
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # 3. Iterate through candidate scenes until valid surface pixels found
        for scene in scenes:
            props = scene.get("properties", {})
            scene_id = scene.get("id", "unknown_scene")
            scene_dt = props.get("datetime")
            scene_cloud = props.get("eo:cloud_cover", 0.0)
            mgrs_tile = props.get("grid:code") or props.get("sentinel:grid_square") or "unknown"

            res = extract_surface_indices_from_scene(scene, lat=lat, lon=lon)
            if not res or not res.get("valid"):
                continue

            ndvi_val = res["ndvi"]
            ndbi_val = res["ndbi"]

            provenance = {
                "NDVI": {
                    "value": round(ndvi_val, 4),
                    "unit": "index (-1 to 1)",
                    "source": f"Sentinel-2 L2A ({provider_name})",
                    "quality": "LIVE",
                    "scene_id": scene_id,
                    "scene_date": scene_dt,
                    "cloud_cover": round(float(scene_cloud), 1),
                    "mgrs_tile": mgrs_tile,
                    "sampling_mode": res["sampling_mode"],
                    "pixel_or_roi": res["pixel_or_roi"],
                    "bands_used": res["bands_used"],
                    "formula_check": res["NDVI"]["formula_check"],
                    "valid_pixels": res["valid_pixels"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "NDBI": {
                    "value": round(ndbi_val, 4),
                    "unit": "index (-1 to 1)",
                    "source": f"Sentinel-2 L2A ({provider_name})",
                    "quality": "LIVE",
                    "scene_id": scene_id,
                    "scene_date": scene_dt,
                    "cloud_cover": round(float(scene_cloud), 1),
                    "mgrs_tile": mgrs_tile,
                    "sampling_mode": res["sampling_mode"],
                    "pixel_or_roi": res["pixel_or_roi"],
                    "bands_used": res["bands_used"],
                    "formula_check": res["NDBI"]["formula_check"],
                    "valid_pixels": res["valid_pixels"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

            result_payload = {
                "status": "available",
                "provider": provider_name,
                "scene_id": scene_id,
                "scene_date": scene_dt,
                "cloud_cover": round(float(scene_cloud), 1),
                "mgrs_tile": mgrs_tile,
                "ndvi": round(ndvi_val, 4),
                "ndbi": round(ndbi_val, 4),
                "sampling_mode": res["sampling_mode"],
                "pixel_or_roi": res["pixel_or_roi"],
                "bands_used": res["bands_used"],
                "NDVI": res["NDVI"],
                "NDBI": res["NDBI"],
                "coverage_pct": res["coverage_pct"],
                "raw_bands_mean": res.get("raw_bands_mean", {}),
                "provenance": provenance,
                "cache_hit": False,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

            # Cache the successful extraction
            provider_cache.set(cache_key, result_payload, ttl=TTL_OPTICAL_INDICES)
            return result_payload

        # All candidate scenes were cloudy at the exact coordinate
        return {
            "status": "all_scenes_cloudy_at_coordinate",
            "provider": provider_name,
            "message": "Candidate Sentinel-2 scenes had cloud obscuration over the target coordinate.",
            "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }
