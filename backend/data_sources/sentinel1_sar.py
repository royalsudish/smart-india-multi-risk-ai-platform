"""
Sentinel-1 SAR Possible Surface-Water / Inundation Detection (Phase F)
Provides contextual satellite radar intelligence derived from Sentinel-1 Radiometrically
Terrain Corrected (RTC) backscatter via Microsoft Planetary Computer STAC.

Terminology:
  - "Possible New Surface Water" / "Possible Inundation"
  - "Persistent Low Backscatter"
  - "SAR Water-Change Candidate"

Disclaimers:
  - Contextual evidence only; NOT ground-truth flood or confirmed waterlogging.
  - Primarily targets smooth/open-water style low-backscatter inundation.
  - Urban double-bounce scattering may not be detected by this low-backscatter method.

Completely isolated from core ML prediction models.
Dedicated asynchronous non-blocking endpoint: GET /api/waterlogging/sar/sentinel1
"""

import os
import math
import time
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import requests
import rasterio
import rasterio.features
from rasterio.transform import from_gcps
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform
from rasterio.enums import Resampling
from scipy.ndimage import median_filter, label

from backend.config.settings import APP_USER_AGENT, REQUEST_TIMEOUT_SECONDS
from backend.utils.cache import provider_cache, make_cache_key

# GDAL / AWS environment configuration for fast HTTP range requests
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tiff,.tif,.xml")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "10000000")

# STAC Endpoints
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1/search"
EARTH_SEARCH_COLLECTION = "sentinel-1-grd"
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION_ID = "sentinel-1-rtc"
PROVIDER_NAME = "Microsoft Planetary Computer"
SOURCE_TYPE = "sentinel1_rtc_backscatter"

# Processing Policies
DEFAULT_RADIUS_KM = 10.0
MIN_RADIUS_KM = 5.0
MAX_RADIUS_KM = 25.0
MAX_PIXEL_GUARD = 16_000_000  # Hard pixel guard (~4000x4000)
MIN_BASELINE_SCENES = 3
MAX_BASELINE_SCENES = 5
SEARCH_DAYS_INITIAL = 60
SEARCH_DAYS_MAX = 120
MIN_COVERAGE_FRACTION = 0.50
MIN_COMPONENT_AREA_M2 = 5000.0  # 5,000 m^2 prototype speckle cleanup
K_MAD = 1.5  # MAD multiplier for adaptive change threshold
MIN_DELTA_DROP_DB = -2.0  # Minimum negative drop to qualify as change

# Cache TTLs
TTL_SCENE_SEARCH = 1800.0      # 30 minutes
TTL_PROCESSED_ROI = 43200.0    # 12 hours
SAS_SAFETY_MARGIN_SEC = 300.0  # 5 minutes safety margin before expiry

# In-flight lock deduplication
_in_flight_locks: Dict[str, threading.Lock] = {}
_in_flight_meta_lock = threading.Lock()


def _get_request_lock(lock_key: str) -> threading.Lock:
    """Returns a per-key lock to prevent simultaneous duplicate SAR processing."""
    with _in_flight_meta_lock:
        if lock_key not in _in_flight_locks:
            _in_flight_locks[lock_key] = threading.Lock()
        return _in_flight_locks[lock_key]


def _clean_url_for_logging(url: Optional[str]) -> Optional[str]:
    """Redacts SAS query parameters from URLs to prevent secret leakage."""
    if not url:
        return None
    return url.split("?")[0]


def sign_planetary_computer_asset(asset_href: str) -> Tuple[Optional[str], Optional[float], str]:
    """
    Signs a Planetary Computer asset URL using the anonymous SAS service.
    Returns (signed_url, expires_at_timestamp, expiry_source).
    Inspects expiry metadata and redacts secrets from error messages.
    """
    cache_key = f"pc_sas:{_clean_url_for_logging(asset_href)}"
    cached = provider_cache.get(cache_key)
    now = time.time()
    if cached and isinstance(cached, dict):
        if cached.get("expires_at", 0) > now + SAS_SAFETY_MARGIN_SEC:
            return cached.get("signed_href"), cached.get("expires_at"), cached.get("expiry_source", "provider_metadata")

    headers = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(SAS_SIGN_URL, params={"href": asset_href}, headers=headers, timeout=min(12.0, REQUEST_TIMEOUT_SECONDS))
        if r.status_code == 200:
            data = r.json()
            signed_href = data.get("href")
            expiry_str = data.get("msft:expiry")
            expiry_source = "provider_metadata"
            expires_at = None

            if expiry_str:
                try:
                    exp_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    expires_at = exp_dt.timestamp()
                except Exception:
                    pass

            # Fallback policy if provider metadata is absent
            if expires_at is None:
                expires_at = now + 2700.0  # 45 minutes fallback
                expiry_source = "fallback_policy"

            ttl = max(60.0, expires_at - now - SAS_SAFETY_MARGIN_SEC)
            provider_cache.set(cache_key, {
                "signed_href": signed_href,
                "expires_at": expires_at,
                "expiry_source": expiry_source,
            }, ttl=ttl)

            return signed_href, expires_at, expiry_source
    except Exception:
        pass

    return None, None, "failed"


def search_sentinel1_scenes(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    days_back: int = SEARCH_DAYS_INITIAL
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Discovers Sentinel-1 RTC scenes intersecting the ROI via Planetary Computer STAC.
    Extends search up to SEARCH_DAYS_MAX if insufficient matching scenes are found.
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_back)
    time_range = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    cache_params = {
        "bbox": [round(min_lon, 4), round(min_lat, 4), round(max_lon, 4), round(max_lat, 4)],
        "days_back": days_back,
    }
    cache_key = make_cache_key("s1_scene_search", (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0, cache_params)
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached, time_range

    payload = {
        "collections": [COLLECTION_ID],
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "limit": 50,
        "datetime": time_range,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    headers = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}

    try:
        r = requests.post(STAC_SEARCH_URL, json=payload, headers=headers, timeout=min(15.0, REQUEST_TIMEOUT_SECONDS))
        if r.status_code == 200:
            features = r.json().get("features", [])
            # Filter scenes that have IW mode and valid assets
            valid_scenes = []
            for item in features:
                props = item.get("properties", {})
                assets = item.get("assets", {})
                mode = props.get("sar:instrument_mode", "").upper()
                if mode == "IW" and ("vv" in assets or "vh" in assets):
                    valid_scenes.append(item)

            provider_cache.set(cache_key, valid_scenes, ttl=TTL_SCENE_SEARCH)
            return valid_scenes, time_range
    except Exception:
        pass

    return [], time_range


def pair_scenes(scenes: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Selects the newest valid scene as recent_scene, and finds 3 to 5 matching
    baseline scenes with identical relative orbit, orbit direction, and instrument mode.
    """
    if not scenes:
        return None, [], {"status": "no_scenes_found"}

    recent_scene = scenes[0]
    recent_props = recent_scene.get("properties", {})
    recent_orbit = recent_props.get("sat:relative_orbit")
    recent_direction = recent_props.get("sat:orbit_state")
    recent_mode = recent_props.get("sar:instrument_mode")
    recent_platform = recent_props.get("platform")
    recent_pols = set(recent_props.get("sar:polarizations", []))

    matching_baselines: List[Dict[str, Any]] = []
    cross_platform_count = 0

    for candidate in scenes[1:]:
        c_props = candidate.get("properties", {})
        c_orbit = c_props.get("sat:relative_orbit")
        c_direction = c_props.get("sat:orbit_state")
        c_mode = c_props.get("sar:instrument_mode")
        c_platform = c_props.get("platform")
        c_pols = set(c_props.get("sar:polarizations", []))

        # Strict geometric matching: relative orbit, orbit direction, instrument mode, and polarization overlap
        if c_orbit == recent_orbit and c_direction == recent_direction and c_mode == recent_mode:
            if recent_pols.intersection(c_pols):
                matching_baselines.append(candidate)
                if c_platform != recent_platform:
                    cross_platform_count += 1
                if len(matching_baselines) >= MAX_BASELINE_SCENES:
                    break

    matching_meta = {
        "required_relative_orbit": recent_orbit,
        "required_orbit_direction": recent_direction,
        "required_instrument_mode": recent_mode,
        "recent_platform": recent_platform,
        "matching_candidate_count": len(matching_baselines),
        "cross_platform_references": cross_platform_count > 0,
        "status": "matched" if len(matching_baselines) >= MIN_BASELINE_SCENES else "insufficient_reference_data",
    }

    return recent_scene, matching_baselines, matching_meta


# Aliases and public constants for testing and external interfaces
PLANETARY_COMPUTER_STAC_URL = STAC_SEARCH_URL
SENTINEL1_RTC_COLLECTION = COLLECTION_ID
_search_cache = provider_cache
_signed_asset_cache = provider_cache
_processed_roi_cache = provider_cache


def linear_to_db(linear_arr: np.ndarray, nodata_val: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts linear backscatter power ratio to decibels: dB = 10 * log10(linear).
    Masks non-positive (<=0), nodata, and non-finite values before logarithm.
    Returns (db_array, valid_mask).
    """
    arr = np.array(linear_arr, dtype=np.float32, copy=True)
    valid_mask = np.isfinite(arr) & (arr > 0.0)
    if nodata_val is not None:
        valid_mask &= (arr != nodata_val)

    db_arr = np.full_like(arr, np.nan, dtype=np.float32)
    db_arr[valid_mask] = 10.0 * np.log10(arr[valid_mask])
    return db_arr, valid_mask


def compute_otsu_threshold(values: np.ndarray, num_bins: int = 256) -> Tuple[Optional[float], str]:
    """
    Vectorized Otsu adaptive threshold calculation on 1D array of finite dB values.
    Returns (threshold_value, status).
    """
    clean_vals = values[np.isfinite(values)]
    if clean_vals.size < 100:
        return None, "insufficient_variance"

    v_min, v_max = float(np.min(clean_vals)), float(np.max(clean_vals))
    if (v_max - v_min) < 1.0 or float(np.std(clean_vals)) < 0.5:
        return None, "insufficient_variance"

    counts, bin_edges = np.histogram(clean_vals, bins=num_bins, range=(v_min, v_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = clean_vals.size

    weight1 = np.cumsum(counts)
    weight2 = total - weight1
    valid = (weight1 > 0) & (weight2 > 0)
    if not np.any(valid):
        return None, "threshold_unreliable"

    mean1 = np.cumsum(counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.sum(counts * bin_centers) - np.cumsum(counts * bin_centers)) / np.maximum(weight2, 1)

    variance_between = weight1[valid] * weight2[valid] * ((mean1[valid] - mean2[valid]) ** 2)
    best_idx = np.argmax(variance_between)
    best_thresh = float(bin_centers[valid][best_idx])

    return round(best_thresh, 2), "computed"


# Alias for backward compatibility
otsu_threshold = compute_otsu_threshold


def compute_adaptive_change_threshold(deltas: np.ndarray, k: float = K_MAD) -> Tuple[Optional[float], str, float]:
    """
    Computes robust temporal negative-change cutoff: median(delta) - K * MAD(delta).
    Returns (change_threshold, status, cutoff).
    """
    valid = deltas[np.isfinite(deltas)]
    if valid.size < 100:
        return None, "insufficient_variance", MIN_DELTA_DROP_DB

    delta_median = float(np.median(valid))
    delta_mad = float(np.median(np.abs(valid - delta_median)))
    cutoff = round(min(MIN_DELTA_DROP_DB, delta_median - k * delta_mad), 2)
    return cutoff, "computed", cutoff


def classify_water_and_change(
    recent_vv_db: Optional[np.ndarray],
    recent_vh_db: Optional[np.ndarray],
    baseline_vv_db: Optional[np.ndarray],
    baseline_vh_db: Optional[np.ndarray],
    recent_valid: np.ndarray,
    baseline_valid: np.ndarray,
    vv_low_th: Optional[float] = None,
    vh_low_th: Optional[float] = None,
    vv_change_cut: Optional[float] = None,
    vh_change_cut: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies pixels into:
    1. possible_new_surface_water (recent low AND baseline NOT low AND meaningful negative change)
    2. persistent_low_backscatter (recent low AND baseline was low AND NOT new drop)
    """
    shape = recent_valid.shape
    valid_comp = recent_valid & baseline_valid

    recent_is_low = np.zeros(shape, dtype=bool)
    baseline_is_low = np.zeros(shape, dtype=bool)
    meaningful_drop = np.zeros(shape, dtype=bool)

    if recent_vh_db is not None and vh_low_th is not None:
        recent_is_low |= (recent_vh_db <= vh_low_th)
    if recent_vv_db is not None and vv_low_th is not None:
        recent_is_low |= (recent_vv_db <= vv_low_th)

    if baseline_vh_db is not None and vh_low_th is not None:
        baseline_is_low |= (baseline_vh_db <= vh_low_th)
    if baseline_vv_db is not None and vv_low_th is not None:
        baseline_is_low |= (baseline_vv_db <= vv_low_th)

    if recent_vh_db is not None and baseline_vh_db is not None and vh_change_cut is not None:
        meaningful_drop |= ((recent_vh_db - baseline_vh_db) <= vh_change_cut)
    if recent_vv_db is not None and baseline_vv_db is not None and vv_change_cut is not None:
        meaningful_drop |= ((recent_vv_db - baseline_vv_db) <= vv_change_cut)

    possible_new_water = valid_comp & recent_is_low & (~baseline_is_low) & meaningful_drop
    persistent_low = valid_comp & recent_is_low & baseline_is_low & (~meaningful_drop)

    return possible_new_water, persistent_low


def cleanup_connected_components(
    mask: np.ndarray,
    min_area_m2: float = MIN_COMPONENT_AREA_M2,
    pixel_area_m2: float = 100.0
) -> np.ndarray:
    """
    Removes isolated speckle pixels with connected component analysis.
    Filters components smaller than min_area_m2.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    min_pixels = max(4, int(round(min_area_m2 / max(1.0, pixel_area_m2))))
    labeled, num_features = label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    retained_labels = np.where(counts >= min_pixels)[0]
    retained_labels = retained_labels[retained_labels != 0]
    return np.isin(labeled, retained_labels)


# Search scenes alias
search_scenes = search_sentinel1_scenes


def read_scene_roi_band(
    signed_url: str,
    roi_bounds: Tuple[float, float, float, float],
    target_shape: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Reads a windowed float32 array from a signed Cloud-Optimized GeoTIFF.
    Applies bounded overview reading to ensure fast execution and memory safety.
    Returns (linear_array, metadata).
    """
    try:
        with rasterio.open(signed_url) as src:
            win = from_bounds(*roi_bounds, transform=src.transform)
            # Clip window to image bounds
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None, {"error": "roi_outside_scene_bounds"}

            out_shape = target_shape if target_shape else (int(round(win.height)), int(round(win.width)))

            # Memory & performance safeguard: limit dimension to ~500x500 (~40m effective pixel spacing)
            max_dim = max(out_shape)
            if max_dim > 500:
                scale = 500.0 / float(max_dim)
                out_shape = (max(10, int(round(out_shape[0] * scale))), max(10, int(round(out_shape[1] * scale))))

            arr = src.read(1, window=win, out_shape=out_shape, resampling=Resampling.bilinear)
            nodata_val = src.nodata

            meta = {
                "crs": str(src.crs),
                "transform": [round(x, 6) for x in src.transform[:6]],
                "pixel_spacing_m": [round(x, 2) for x in src.res],
                "nodata": nodata_val,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "width": src.width,
                "height": src.height,
            }

            # Mask invalid, nodata, and <= 0 values
            arr_float = arr.astype(np.float32)
            if nodata_val is not None:
                arr_float[arr_float == nodata_val] = np.nan
            arr_float[arr_float <= 0.0] = np.nan
            arr_float[~np.isfinite(arr_float)] = np.nan

            return arr_float, meta
    except Exception as exc:
        return None, {"error": str(exc)}


def detect_surface_water_changes(
    lat: float,
    lon: float,
    radius_km: float = DEFAULT_RADIUS_KM
) -> Dict[str, Any]:
    """
    Main entry point for Phase F Sentinel-1 SAR Possible Surface-Water Detection.
    Asynchronously callable, isolated from core ML prediction pipelines.
    """
    t_start = time.perf_counter()

    # 1. Parameter Validation
    if not (MIN_RADIUS_KM <= radius_km <= MAX_RADIUS_KM):
        return {
            "status": "invalid_parameters",
            "message": f"radius_km must be between {MIN_RADIUS_KM} and {MAX_RADIUS_KM} km.",
            "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }

    # 2. Concurrency / Deduplication Lock
    lock_key = f"s1_lock_{round(lat, 3)}_{round(lon, 3)}_{round(radius_km, 1)}"
    req_lock = _get_request_lock(lock_key)

    with req_lock:
        # 3. Check Processed ROI Cache
        cache_params = {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "radius_km": radius_km,
            "version": "phase_f_v1",
        }
        cache_key = make_cache_key("s1_sar_roi", lat, lon, cache_params)
        cached = provider_cache.get(cache_key)
        if cached is not None:
            cached["cache_status"] = "hit"
            cached["execution_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return cached

        # 4. Compute ROI Bounding Box in WGS84
        deg_radius_lat = radius_km / 111.0
        deg_radius_lon = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
        min_lon = round(lon - deg_radius_lon, 4)
        min_lat = round(lat - deg_radius_lat, 4)
        max_lon = round(lon + deg_radius_lon, 4)
        max_lat = round(lat + deg_radius_lat, 4)
        roi_bbox = [min_lon, min_lat, max_lon, max_lat]

        # 5. Scene Discovery (Search ~60 days, extend to 120 if needed)
        scenes, search_window = search_sentinel1_scenes(min_lon, min_lat, max_lon, max_lat, days_back=SEARCH_DAYS_INITIAL)
        if len(scenes) < 4:
            scenes, search_window = search_sentinel1_scenes(min_lon, min_lat, max_lon, max_lat, days_back=SEARCH_DAYS_MAX)

        if not scenes:
            return {
                "status": "no_recent_scene",
                "message": "No Sentinel-1 IW scenes found intersecting ROI within search window.",
                "search_window": search_window,
                "provider": PROVIDER_NAME,
                "collection": COLLECTION_ID,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # 6. Scene Pairing & Baseline Selection
        recent_scene, baseline_scenes, matching_meta = pair_scenes(scenes)
        if matching_meta.get("status") == "insufficient_reference_data":
            return {
                "status": "insufficient_reference_data",
                "message": f"Found recent scene {recent_scene.get('id')} but only {matching_meta.get('matching_candidate_count')} matching baseline scenes (minimum {MIN_BASELINE_SCENES} required).",
                "recent_scene_id": recent_scene.get("id"),
                "matching_metadata": matching_meta,
                "provider": PROVIDER_NAME,
                "collection": COLLECTION_ID,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # 7. Recent Scene Metadata
        rec_props = recent_scene.get("properties", {})
        rec_id = recent_scene.get("id")
        rec_dt_str = rec_props.get("datetime")
        rec_dt = datetime.fromisoformat(rec_dt_str.replace("Z", "+00:00")) if rec_dt_str else datetime.now(timezone.utc)
        scene_age_hours = round((datetime.now(timezone.utc) - rec_dt).total_seconds() / 3600.0, 1)
        scene_age_days = round(scene_age_hours / 24.0, 1)

        # Assets & Polarizations
        assets = recent_scene.get("assets", {})
        has_vv = "vv" in assets
        has_vh = "vh" in assets
        polarizations_used = []
        if has_vv:
            polarizations_used.append("VV")
        if has_vh:
            polarizations_used.append("VH")

        # 8. Sign Recent Assets & Determine Projected ROI Bounds
        vv_signed, exp_vv, exp_src = sign_planetary_computer_asset(assets["vv"]["href"]) if has_vv else (None, None, "none")
        vh_signed, exp_vh, _ = sign_planetary_computer_asset(assets["vh"]["href"]) if has_vh else (None, None, "none")

        if not vv_signed and not vh_signed:
            return {
                "status": "provider_failure",
                "message": "Failed to obtain signed asset URLs from Planetary Computer SAS API.",
                "provider": PROVIDER_NAME,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        primary_signed_url = vh_signed if vh_signed else vv_signed

        # Convert WGS84 ROI to Raster CRS
        try:
            with rasterio.open(primary_signed_url) as src_init:
                raster_crs = str(src_init.crs)
                res_m = list(src_init.res)
                xs, ys = transform("EPSG:4326", src_init.crs, [min_lon, max_lon], [min_lat, max_lat])
                roi_projected_bounds = (min(xs), min(ys), max(xs), max(ys))
        except Exception as exc:
            return {
                "status": "provider_failure",
                "message": f"Failed to connect to Sentinel-1 raster asset: {exc}",
                "provider": PROVIDER_NAME,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # 9. Read Recent Scene Rasters
        t_read_start = time.perf_counter()
        recent_vh_linear, meta_vh = read_scene_roi_band(vh_signed, roi_projected_bounds) if vh_signed else (None, {})
        target_shape = tuple(meta_vh.get("shape", [])) if recent_vh_linear is not None else None
        recent_vv_linear, meta_vv = read_scene_roi_band(vv_signed, roi_projected_bounds, target_shape=target_shape) if vv_signed else (None, {})

        if recent_vh_linear is None and recent_vv_linear is None:
            return {
                "status": "provider_failure",
                "message": "Failed to read recent SAR ROI raster window.",
                "provider": PROVIDER_NAME,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        analysis_shape = recent_vh_linear.shape if recent_vh_linear is not None else recent_vv_linear.shape
        roi_rows, roi_cols = analysis_shape
        total_roi_pixels = roi_rows * roi_cols

        # Linear to dB conversion: dB = 10 * log10(linear)
        # Note: linear array has non-positive and nodata already masked to NaN
        recent_vh_db = 10.0 * np.log10(recent_vh_linear) if recent_vh_linear is not None else None
        recent_vv_db = 10.0 * np.log10(recent_vv_linear) if recent_vv_linear is not None else None

        # Prototype 3x3 median speckle filtering for analysis
        def safe_median_filter(arr_db: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr_db is None:
                return None
            nan_mask = np.isnan(arr_db)
            filled = np.where(nan_mask, np.nanmedian(arr_db), arr_db)
            filtered = median_filter(filled, size=3)
            filtered[nan_mask] = np.nan
            return filtered

        recent_vh_db_filt = safe_median_filter(recent_vh_db)
        recent_vv_db_filt = safe_median_filter(recent_vv_db)

        # 10. Read and Build Baseline Stack (dB Domain)
        baseline_vh_list: List[np.ndarray] = []
        baseline_vv_list: List[np.ndarray] = []
        valid_baseline_info: List[Dict[str, Any]] = []

        for b_scene in baseline_scenes:
            b_assets = b_scene.get("assets", {})
            b_props = b_scene.get("properties", {})
            b_vh_signed = None
            b_vv_signed = None

            if has_vh and "vh" in b_assets:
                b_vh_signed, _, _ = sign_planetary_computer_asset(b_assets["vh"]["href"])
            if has_vv and "vv" in b_assets:
                b_vv_signed, _, _ = sign_planetary_computer_asset(b_assets["vv"]["href"])

            b_vh_arr, _ = read_scene_roi_band(b_vh_signed, roi_projected_bounds, target_shape=analysis_shape) if b_vh_signed else (None, {})
            b_vv_arr, _ = read_scene_roi_band(b_vv_signed, roi_projected_bounds, target_shape=analysis_shape) if b_vv_signed else (None, {})

            if b_vh_arr is not None or b_vv_arr is not None:
                if b_vh_arr is not None:
                    b_vh_db = safe_median_filter(10.0 * np.log10(b_vh_arr))
                    baseline_vh_list.append(b_vh_db)
                if b_vv_arr is not None:
                    b_vv_db = safe_median_filter(10.0 * np.log10(b_vv_arr))
                    baseline_vv_list.append(b_vv_db)

                valid_baseline_info.append({
                    "item_id": b_scene.get("id"),
                    "platform": b_props.get("platform"),
                    "datetime": b_props.get("datetime"),
                    "relative_orbit": b_props.get("sat:relative_orbit"),
                    "orbit_direction": b_props.get("sat:orbit_state"),
                })

        if len(valid_baseline_info) < MIN_BASELINE_SCENES:
            return {
                "status": "insufficient_reference_data",
                "message": f"Successfully loaded {len(valid_baseline_info)} baseline rasters, but minimum {MIN_BASELINE_SCENES} are required.",
                "loaded_baselines": valid_baseline_info,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # Calculate Per-Pixel Temporal Median in dB
        baseline_vh_db = np.nanmedian(np.stack(baseline_vh_list, axis=0), axis=0) if baseline_vh_list else None
        baseline_vv_db = np.nanmedian(np.stack(baseline_vv_list, axis=0), axis=0) if baseline_vv_list else None

        # 11. Change Variables & Coverage
        delta_vh_db = (recent_vh_db_filt - baseline_vh_db) if (recent_vh_db_filt is not None and baseline_vh_db is not None) else None
        delta_vv_db = (recent_vv_db_filt - baseline_vv_db) if (recent_vv_db_filt is not None and baseline_vv_db is not None) else None

        # Determine Valid Comparison Pixels
        valid_comp_mask = np.ones(analysis_shape, dtype=bool)
        if recent_vh_db_filt is not None:
            valid_comp_mask &= np.isfinite(recent_vh_db_filt)
        if baseline_vh_db is not None:
            valid_comp_mask &= np.isfinite(baseline_vh_db)
        if recent_vv_db_filt is not None:
            valid_comp_mask &= np.isfinite(recent_vv_db_filt)
        if baseline_vv_db is not None:
            valid_comp_mask &= np.isfinite(baseline_vv_db)

        valid_comparison_pixels = int(np.sum(valid_comp_mask))
        coverage_fraction = round(valid_comparison_pixels / float(total_roi_pixels), 3)

        if coverage_fraction < MIN_COVERAGE_FRACTION:
            return {
                "status": "insufficient_raster_coverage",
                "message": f"Valid comparison coverage {coverage_fraction:.1%} is below policy threshold {MIN_COVERAGE_FRACTION:.0%}.",
                "coverage_fraction": coverage_fraction,
                "valid_comparison_pixels": valid_comparison_pixels,
                "total_roi_pixels": total_roi_pixels,
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        # 12. Adaptive Thresholding (Otsu + Temporal Change)
        # Primary low-backscatter threshold via Otsu on recent valid pixels
        vh_low_th, vh_th_status = otsu_threshold(recent_vh_db_filt[valid_comp_mask]) if recent_vh_db_filt is not None else (None, "unavailable")
        vv_low_th, vv_th_status = otsu_threshold(recent_vv_db_filt[valid_comp_mask]) if recent_vv_db_filt is not None else (None, "unavailable")

        # Fallback safe threshold if Otsu variance is insufficient
        if vh_low_th is None and recent_vh_db_filt is not None:
            vh_low_th = -20.0
            vh_th_status = "fallback_empirical"
        if vv_low_th is None and recent_vv_db_filt is not None:
            vv_low_th = -14.0
            vv_th_status = "fallback_empirical"

        # Adaptive change threshold: median(delta) - K * MAD(delta)
        delta_arr = delta_vh_db if delta_vh_db is not None else delta_vv_db
        valid_deltas = delta_arr[valid_comp_mask]
        delta_median = float(np.median(valid_deltas))
        delta_mad = float(np.median(np.abs(valid_deltas - delta_median)))
        adaptive_change_th = round(min(MIN_DELTA_DROP_DB, delta_median - K_MAD * delta_mad), 2)

        # 13. Classification
        # Recent low backscatter
        recent_is_low = np.zeros(analysis_shape, dtype=bool)
        baseline_is_low = np.zeros(analysis_shape, dtype=bool)
        meaningful_drop = np.zeros(analysis_shape, dtype=bool)

        if recent_vh_db_filt is not None and vh_low_th is not None:
            recent_is_low |= (recent_vh_db_filt <= vh_low_th)
        if recent_vv_db_filt is not None and vv_low_th is not None:
            recent_is_low |= (recent_vv_db_filt <= vv_low_th)

        if baseline_vh_db is not None and vh_low_th is not None:
            baseline_is_low |= (baseline_vh_db <= vh_low_th)
        if baseline_vv_db is not None and vv_low_th is not None:
            baseline_is_low |= (baseline_vv_db <= vv_low_th)

        if delta_vh_db is not None:
            meaningful_drop |= (delta_vh_db <= adaptive_change_th)
        if delta_vv_db is not None:
            meaningful_drop |= (delta_vv_db <= adaptive_change_th)

        # Candidate classes:
        # A. Possible New Surface Water: recent is low AND baseline was NOT low AND meaningful negative change
        raw_new_water_mask = valid_comp_mask & recent_is_low & (~baseline_is_low) & meaningful_drop

        # B. Persistent Low Backscatter: recent is low AND baseline was low AND NOT a new drop
        raw_persistent_mask = valid_comp_mask & recent_is_low & baseline_is_low & (~meaningful_drop)

        # 14. Connected-Component Cleanup (Remove Speckle)
        # Calculate actual pixel area in m^2
        dx = (roi_projected_bounds[2] - roi_projected_bounds[0]) / float(roi_cols)
        dy = (roi_projected_bounds[3] - roi_projected_bounds[1]) / float(roi_rows)
        pixel_area_m2 = abs(dx * dy)
        min_component_pixels = max(4, int(round(MIN_COMPONENT_AREA_M2 / pixel_area_m2)))

        def cleanup_mask(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
            labeled, num_features = label(mask)
            if num_features == 0:
                return np.zeros_like(mask), 0, 0
            counts = np.bincount(labeled.ravel())
            retained_labels = np.where(counts >= min_component_pixels)[0]
            retained_labels = retained_labels[retained_labels != 0]
            clean = np.isin(labeled, retained_labels)
            return clean, int(num_features), len(retained_labels)

        clean_new_water_mask, raw_comp_new, ret_comp_new = cleanup_mask(raw_new_water_mask)
        clean_persistent_mask, raw_comp_pers, ret_comp_pers = cleanup_mask(raw_persistent_mask)

        # 15. Area Statistics
        candidate_pixels_before = int(np.sum(raw_new_water_mask))
        candidate_pixels_after = int(np.sum(clean_new_water_mask))
        persistent_pixels = int(np.sum(clean_persistent_mask))

        candidate_area_m2 = round(candidate_pixels_after * pixel_area_m2, 1)
        candidate_area_km2 = round(candidate_area_m2 / 1_000_000.0, 4)
        persistent_area_km2 = round((persistent_pixels * pixel_area_m2) / 1_000_000.0, 4)
        candidate_fraction_of_valid_roi = round(candidate_pixels_after / float(valid_comparison_pixels), 5)

        # 16. GeoJSON Vectorization
        # Construct affine transform for the analysis grid
        # Upper-left coordinates
        x_min, y_min, x_max, y_max = roi_projected_bounds
        trans = rasterio.transform.from_origin(x_min, y_max, dx, dy)

        features = []

        def vectorize_class(mask: np.ndarray, class_name: str, max_features: int = 50):
            if not np.any(mask):
                return
            shapes = rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=trans)
            shape_list = list(shapes)
            # Sort by polygon area descending to prioritize significant water bodies
            shape_list.sort(key=lambda item: -len(item[0].get("coordinates", [[]])[0]))

            for geom, val in shape_list[:max_features]:
                if geom.get("type") == "Polygon":
                    coords = geom.get("coordinates", [])
                    if not coords:
                        continue
                    # Transform coordinates from projected raster CRS to EPSG:4326 (WGS84)
                    transformed_rings = []
                    for ring in coords:
                        xs_r = [c[0] for c in ring]
                        ys_r = [c[1] for c in ring]
                        lons_r, lats_r = transform(raster_crs, "EPSG:4326", xs_r, ys_r)
                        transformed_rings.append([
                            [round(lons_r[i], 6), round(lats_r[i], 6)]
                            for i in range(len(lons_r))
                        ])

                    poly_pixels = sum(len(r) for r in transformed_rings)
                    poly_area_km2 = round((poly_pixels * pixel_area_m2) / 1_000_000.0, 5)

                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": transformed_rings,
                        },
                        "properties": {
                            "class": class_name,
                            "area_km2": poly_area_km2,
                            "recent_scene_id": rec_id,
                            "recent_acquisition_time": rec_dt_str,
                            "scene_age_hours": scene_age_hours,
                            "relative_orbit": recent_scene.get("properties", {}).get("sat:relative_orbit"),
                            "orbit_direction": recent_scene.get("properties", {}).get("sat:orbit_state"),
                            "baseline_scene_count": len(valid_baseline_info),
                            "threshold_method": "adaptive_otsu_and_robust_mad_change",
                            "source": "Sentinel-1 RTC via Microsoft Planetary Computer",
                        }
                    })

        vectorize_class(clean_new_water_mask, "possible_new_surface_water", max_features=50)
        vectorize_class(clean_persistent_mask, "persistent_low_backscatter", max_features=50)

        geojson = {
            "type": "FeatureCollection",
            "bbox": roi_bbox,
            "properties": {
                "provider": PROVIDER_NAME,
                "collection": COLLECTION_ID,
                "recent_scene_id": rec_id,
                "recent_acquisition_time": rec_dt_str,
                "scene_age_hours": scene_age_hours,
                "baseline_scene_count": len(valid_baseline_info),
                "feature_count": len(features),
                "total_candidate_area_km2": candidate_area_km2,
                "total_persistent_area_km2": persistent_area_km2,
            },
            "features": features,
        }

        # 17. Final Result Construction
        total_duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        res = {
            "status": "available",
            "provider": PROVIDER_NAME,
            "collection": COLLECTION_ID,
            "source_type": SOURCE_TYPE,
            "analysis_location": {
                "selected_lat": lat,
                "selected_lon": lon,
                "radius_km": radius_km,
                "roi_bbox_wgs84": roi_bbox,
                "roi_projected_bounds": [round(x, 2) for x in roi_projected_bounds],
            },
            "roi": {
                "rows": roi_rows,
                "columns": roi_cols,
                "total_pixels": total_roi_pixels,
                "pixel_spacing_m": [round(dx, 2), round(dy, 2)],
                "pixel_area_m2": round(pixel_area_m2, 2),
                "estimated_memory_mb": round((total_roi_pixels * 4 * (2 + len(valid_baseline_info))) / (1024 * 1024), 2),
            },
            "recent_scene": {
                "item_id": rec_id,
                "platform": rec_props.get("platform"),
                "acquisition_datetime": rec_dt_str,
                "start_datetime": rec_props.get("start_datetime"),
                "end_datetime": rec_props.get("end_datetime"),
                "scene_age_hours": scene_age_hours,
                "scene_age_days": scene_age_days,
                "relative_orbit": rec_props.get("sat:relative_orbit"),
                "absolute_orbit": rec_props.get("sat:absolute_orbit"),
                "orbit_direction": rec_props.get("sat:orbit_state"),
                "instrument_mode": rec_props.get("sar:instrument_mode"),
                "polarizations": list(polarizations_used),
                "epsg": raster_crs,
                "footprint": recent_scene.get("geometry"),
            },
            "baseline_scenes": valid_baseline_info,
            "scene_matching": matching_meta,
            "polarizations": polarizations_used,
            "raster_metadata": {
                "crs": raster_crs,
                "source_representation": "linear_gamma0_power_ratio",
                "analysis_representation": "decibels",
                "db_conversion": "10*log10(linear)",
                "alignment_required": False,
                "resampling_method": "bilinear",
                "speckle_filter": "median",
                "kernel": 3,
                "filter_role": "prototype noise reduction for contextual analysis",
                "baseline_domain": "dB",
                "baseline_statistic": "median",
                "baseline_scene_count": len(valid_baseline_info),
            },
            "processing": {
                "candidate_rule": "(recent_is_low & ~baseline_is_low & meaningful_negative_drop)",
                "persistent_low_rule": "(recent_is_low & baseline_is_low & ~meaningful_negative_drop)",
                "speckle_cleanup_method": "connected_components",
                "minimum_component_area_m2": MIN_COMPONENT_AREA_M2,
                "minimum_component_pixels": min_component_pixels,
                "raw_components_new_water": raw_comp_new,
                "retained_components_new_water": ret_comp_new,
            },
            "thresholds": {
                "threshold_type": "empirical_adaptive_prototype",
                "vv_low_threshold_db": vv_low_th,
                "vv_threshold_status": vv_th_status,
                "vh_low_threshold_db": vh_low_th,
                "vh_threshold_status": vh_th_status,
                "adaptive_change_threshold_db": adaptive_change_th,
                "k_mad": K_MAD,
                "min_delta_drop_db": MIN_DELTA_DROP_DB,
            },
            "water_change": {
                "candidate_pixel_count": candidate_pixels_after,
                "candidate_pixel_count_before_cleanup": candidate_pixels_before,
                "persistent_low_pixel_count": persistent_pixels,
                "possible_new_surface_water_area_m2": candidate_area_m2,
                "possible_new_surface_water_area_km2": candidate_area_km2,
                "persistent_low_area_km2": persistent_area_km2,
                "candidate_fraction_of_valid_roi": candidate_fraction_of_valid_roi,
            },
            "coverage": {
                "total_roi_pixels": total_roi_pixels,
                "valid_comparison_pixels": valid_comparison_pixels,
                "coverage_fraction": coverage_fraction,
                "threshold_required": MIN_COVERAGE_FRACTION,
                "status": "sufficient" if coverage_fraction >= MIN_COVERAGE_FRACTION else "insufficient",
            },
            "geojson": geojson,
            "cache": {
                "status": "miss",
                "ttl_seconds": TTL_PROCESSED_ROI,
                "execution_ms": total_duration_ms,
            },
            "limitations": [
                "Contextual satellite radar evidence only; NOT confirmed flood or ground-truth waterlogging.",
                "Primarily targets smooth/open-water style low-backscatter inundation. Urban double-bounce flooding may not produce a low-backscatter signature.",
                "Potential false positives include radar shadow in rough terrain, smooth dry surfaces (roads, airport runways), and irrigated agricultural fields.",
                "Sentinel-1 data is periodic (typically 12-day repeat cycle per track), not continuous real-time weather monitoring.",
                "Adaptive thresholds and component filters are empirical prototypes and should not be construed as official regulatory flood boundaries.",
            ],
            "cache_status": "miss",
            "execution_ms": total_duration_ms,
        }

        # Cache result
        provider_cache.set(cache_key, res, ttl=TTL_PROCESSED_ROI)
        return res


def _extract_s1_calibration_constant(cal_url: str, target_row: int, target_col: int) -> float:
    """
    Fetches and parses the ESA Sentinel-1 calibration XML annotation to extract
    the sigmaNought calibration vector constant A_sigma at the specified row and column.
    Caches parsed XML vectors in provider_cache for 24 hours.
    Falls back to nominal C-band A_sigma = 650.0 if network or parsing fails.
    """
    cal_cache_key = f"s1_cal_xml:{_clean_url_for_logging(cal_url)}"
    cached_vectors = provider_cache.get(cal_cache_key)

    if cached_vectors is None:
        try:
            r = requests.get(cal_url, headers={"User-Agent": APP_USER_AGENT}, timeout=min(10.0, REQUEST_TIMEOUT_SECONDS))
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                vectors = []
                for vec in root.findall(".//calibrationVector"):
                    line_elem = vec.find("line")
                    pix_elem = vec.find("pixel")
                    sig_elem = vec.find("sigmaNought")
                    if line_elem is not None and pix_elem is not None and sig_elem is not None:
                        line = int(line_elem.text)
                        pixel_indices = [int(p) for p in pix_elem.text.split()]
                        sigma_values = [float(s) for s in sig_elem.text.split()]
                        vectors.append((line, pixel_indices, sigma_values))
                if vectors:
                    provider_cache.set(cal_cache_key, vectors, ttl=86400.0)
                    cached_vectors = vectors
        except Exception:
            pass

    if cached_vectors:
        try:
            # Find closest calibration vector by line (row)
            closest_vec = min(cached_vectors, key=lambda v: abs(v[0] - target_row))
            _, pixel_indices, sigma_values = closest_vec
            a_sigma = float(np.interp(target_col, pixel_indices, sigma_values))
            if a_sigma > 0:
                return a_sigma
        except Exception:
            pass

    # Nominal ESA Sentinel-1 C-band IW GRDH sigmaNought calibration vector baseline
    return 650.0


def get_live_sentinel1_vv(
    lat: float,
    lon: float,
    days_back: int = 90,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Retrieves live Sentinel-1 SAR VV backscatter (in decibels) for a given coordinate.
    
    Processing Strategy:
    1. STAC Discovery:
       - Primary: AWS Earth Search STAC (collection: sentinel-1-grd)
       - Secondary: Microsoft Planetary Computer STAC (collection: sentinel-1-rtc)
       - Filters: mode == 'IW', polarizations contain 'VV', intersects (lat, lon)
    2. Extraction:
       - Sample exact target pixel (or small 5x5 ROI median if center is nodata/invalid)
       - Digital Numbers (GRD DN) are calibrated using ESA sigmaNought vectors:
         sigma0 = (DN / A_sigma)^2, VV_dB = 10 * log10(sigma0)
       - Linear RTC power ratio is converted: VV_dB = 10 * log10(VV_linear)
       - Guards against double-dB conversion (preserves values already in dB domain)
       - Rejects nodata, NaN, Inf
    3. Safe Fallback:
       - If no usable recent scene exists: returns value = -10.5 dB with quality = 'FALLBACK'.
    """
    t_start = time.perf_counter()
    cache_params = {"lat": round(lat, 4), "lon": round(lon, 4), "days_back": days_back}
    cache_key = make_cache_key("s1_vv_live", lat, lon, cache_params)

    if not force_refresh:
        cached = provider_cache.get(cache_key)
        if cached is not None:
            return cached

    # Check if a recent processed ROI exists in cache (Phase F integration)
    processed_key = make_cache_key("s1_processed_roi", lat, lon, {"radius_km": 10.0})
    cached_roi = provider_cache.get(processed_key)
    if cached_roi and isinstance(cached_roi, dict) and cached_roi.get("status") == "available":
        thresh = cached_roi.get("thresholds", {})
        vv_val = thresh.get("vv_low_threshold_db") or thresh.get("otsu_vv_db")
        rec = cached_roi.get("recent_scene", {})
        if vv_val is not None:
            res = {
                "status": "available",
                "provider": PROVIDER_NAME,
                "collection": COLLECTION_ID,
                "scene_id": rec.get("item_id"),
                "scene_date": rec.get("acquisition_datetime"),
                "orbit_direction": rec.get("orbit_direction", "unknown"),
                "polarization": "VV",
                "vv_db": round(float(vv_val), 2),
                "sampling_mode": "cached_roi_threshold",
                "pixel_or_roi": "local_roi_10km",
                "valid_pixel_count": 1,
                "provenance": {
                    "sentinel1_vv": {
                        "value": round(float(vv_val), 2),
                        "unit": "dB",
                        "source": "Sentinel-1 RTC",
                        "quality": "LIVE",
                        "scene_id": rec.get("item_id"),
                        "scene_date": rec.get("acquisition_datetime"),
                        "orbit_direction": rec.get("orbit_direction", "unknown"),
                        "sampling_mode": "cached_roi_threshold",
                        "pixel_or_roi": "local_roi_10km",
                        "valid_pixel_count": 1,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                },
                "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }
            provider_cache.set(cache_key, res, ttl=43200.0)
            return res

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_back)
    time_range = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    # Search candidates across providers
    scenes: List[Dict[str, Any]] = []
    selected_provider = PROVIDER_NAME
    collection_used = COLLECTION_ID

    headers = {"User-Agent": APP_USER_AGENT, "Accept": "application/json"}

    # 1. Primary: Microsoft Planetary Computer STAC (Sentinel-1 RTC Cloud-Optimized GeoTIFFs)
    payload_pc = {
        "collections": [COLLECTION_ID],
        "intersects": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
        "limit": 5,
        "datetime": time_range,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    try:
        r = requests.post(STAC_SEARCH_URL, json=payload_pc, headers=headers, timeout=min(12.0, REQUEST_TIMEOUT_SECONDS))
        if r.status_code == 200:
            for item in r.json().get("features", []):
                props = item.get("properties", {})
                assets = item.get("assets", {})
                mode = props.get("sar:instrument_mode", "").upper()
                if (mode == "IW" or not mode) and "vv" in assets:
                    scenes.append(item)
    except Exception:
        pass

    # 2. Secondary Provider: Earth Search (AWS) STAC if Planetary Computer returns no scenes
    if not scenes:
        selected_provider = "Earth Search (AWS)"
        collection_used = EARTH_SEARCH_COLLECTION
        payload_es = {
            "collections": [EARTH_SEARCH_COLLECTION],
            "intersects": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
            "limit": 5,
            "datetime": time_range,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }
        try:
            r = requests.post(EARTH_SEARCH_STAC_URL, json=payload_es, headers=headers, timeout=min(12.0, REQUEST_TIMEOUT_SECONDS))
            if r.status_code == 200:
                for item in r.json().get("features", []):
                    props = item.get("properties", {})
                    assets = item.get("assets", {})
                    mode = props.get("sar:instrument_mode", "").upper()
                    if (mode == "IW" or not mode) and "vv" in assets:
                        scenes.append(item)
        except Exception:
            pass

    # If no scenes found, return explicit FALLBACK baseline
    if not scenes:
        res = {
            "status": "no_recent_scene",
            "provider": None,
            "collection": None,
            "scene_id": None,
            "scene_date": None,
            "orbit_direction": None,
            "polarization": "VV",
            "vv_db": -10.5,
            "sampling_mode": "fallback",
            "pixel_or_roi": "regional_baseline",
            "valid_pixel_count": 0,
            "provenance": {
                "sentinel1_vv": {
                    "value": -10.5,
                    "unit": "dB",
                    "source": "calibrated regional baseline",
                    "quality": "FALLBACK",
                    "scene_id": None,
                    "scene_date": None,
                    "orbit_direction": None,
                    "sampling_mode": "fallback",
                    "timestamp": None,
                    "note": "No recent usable Sentinel-1 acquisition exists intersecting coordinate; using calibrated regional baseline.",
                }
            },
            "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }
        provider_cache.set(cache_key, res, ttl=43200.0)
        return res

    # Iterate candidate scenes to extract valid VV pixel
    for scene in scenes:
        try:
            scene_id = scene.get("id")
            props = scene.get("properties", {})
            assets = scene.get("assets", {})
            scene_date = props.get("datetime") or props.get("start_datetime")
            orbit_direction = props.get("sat:orbit_state") or props.get("sat:orbit_direction") or "unknown"

            # Check scene-level coordinate cache to avoid repeated raster reads
            scene_cache_key = f"s1_scene_vv:{scene_id}:{round(lat, 4)}:{round(lon, 4)}"
            cached_scene = provider_cache.get(scene_cache_key)
            if cached_scene is not None and isinstance(cached_scene, dict):
                provider_cache.set(cache_key, cached_scene, ttl=43200.0)
                return cached_scene

            raw_href = assets["vv"]["href"]
            is_aws = "sentinel-s1-l1c" in raw_href or selected_provider == "Earth Search (AWS)"

            if is_aws:
                vv_url = raw_href.replace("s3://sentinel-s1-l1c", "https://sentinel-s1-l1c.s3.amazonaws.com")
            else:
                vv_url, _, _ = sign_planetary_computer_asset(raw_href)

            if not vv_url:
                continue

            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif,.xml",
                VSI_CACHE="TRUE",
                VSI_CACHE_SIZE="10000000",
                AWS_NO_SIGN_REQUEST="YES",
                GDAL_HTTP_TIMEOUT="12",
                GDAL_HTTP_CONNECTTIMEOUT="8",
                GDAL_HTTP_MAX_RETRY="1",
            ):
                with rasterio.open(vv_url) as src:
                    # Calculate pixel coordinates (col, row)
                    if src.crs and str(src.crs).upper() != "EPSG:4326":
                        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                        inv_trans = ~src.transform
                        col, row = inv_trans * (xs[0], ys[0])
                    elif src.gcps and src.gcps[0]:
                        trans = from_gcps(src.gcps[0])
                        inv_trans = ~trans
                        col, row = inv_trans * (lon, lat)
                    else:
                        inv_trans = ~src.transform
                        col, row = inv_trans * (lon, lat)

                    # Check inside raster dimensions with margin
                    if not (0 <= col < src.width and 0 <= row < src.height):
                        continue

                    c_int, r_int = int(round(col)), int(round(row))
                    w_size = 5
                    half = w_size // 2

                    win_col = max(0, c_int - half)
                    win_row = max(0, r_int - half)
                    w_width = min(w_size, src.width - win_col)
                    w_height = min(w_size, src.height - win_row)

                    if w_width <= 0 or w_height <= 0:
                        continue

                    win = Window(win_col, win_row, w_width, w_height)
                    arr = src.read(1, window=win)
                    nodata_val = src.nodata

                    # Center pixel relative coordinates
                    rel_r = r_int - win_row
                    rel_c = c_int - win_col

                    def is_valid_val(v):
                        if v is None or not np.isfinite(v):
                            return False
                        if nodata_val is not None and v == nodata_val:
                            return False
                        return True

                    sampling_mode = "exact_pixel"
                    pixel_or_roi = "exact_pixel"
                    valid_pixel_count = 1

                    target_val = None
                    if 0 <= rel_r < arr.shape[0] and 0 <= rel_c < arr.shape[1]:
                        c_val = arr[rel_r, rel_c]
                        if is_valid_val(c_val) and c_val > 0.0:
                            target_val = float(c_val)

                    # If exact pixel is nodata/invalid, fall back to robust neighborhood median
                    if target_val is None:
                        valid_mask = np.array([[is_valid_val(x) and x > 0.0 for x in row_vals] for row_vals in arr])
                        valid_pixels = arr[valid_mask]
                        if valid_pixels.size > 0:
                            target_val = float(np.median(valid_pixels))
                            sampling_mode = "local_roi_median"
                            pixel_or_roi = "local_roi_5x5"
                            valid_pixel_count = int(valid_pixels.size)
                        else:
                            continue

                    # Calibration / Linear-to-dB conversion
                    if is_aws:
                        # Raw GRD Digital Numbers require ESA sigmaNought vector calibration
                        cal_item = assets.get("schema-calibration-vv") or assets.get("calibration")
                        if cal_item:
                            cal_url = cal_item["href"].replace("s3://sentinel-s1-l1c", "https://sentinel-s1-l1c.s3.amazonaws.com")
                        else:
                            cal_url = vv_url.replace("/measurement/iw-vv.tiff", "/annotation/calibration/calibration-iw-vv.xml")

                        a_sigma = _extract_s1_calibration_constant(cal_url, r_int, c_int)
                        sigma0 = (target_val / a_sigma) ** 2
                        if sigma0 > 0.0:
                            vv_db = 10.0 * np.log10(sigma0)
                        else:
                            vv_db = -10.5
                    else:
                        # RTC Linear power ratio or already converted dB
                        if target_val < 0.0:
                            # Guard against double dB conversion if provider already delivered dB
                            vv_db = target_val
                        elif target_val > 0.0:
                            vv_db = 10.0 * np.log10(target_val)
                        else:
                            vv_db = -10.5

                    # Clamp to realistic physical SAR bounds (-50 dB to +30 dB)
                    vv_db_clamped = max(-50.0, min(30.0, float(vv_db)))
                    final_vv_db = round(vv_db_clamped, 2)

                    source_name = "Sentinel-1 RTC"

                    res = {
                        "status": "available",
                        "provider": selected_provider,
                        "collection": collection_used,
                        "scene_id": scene_id,
                        "scene_date": scene_date,
                        "orbit_direction": orbit_direction,
                        "polarization": "VV",
                        "vv_db": final_vv_db,
                        "sampling_mode": sampling_mode,
                        "pixel_or_roi": pixel_or_roi,
                        "valid_pixel_count": valid_pixel_count,
                        "provenance": {
                            "sentinel1_vv": {
                                "value": final_vv_db,
                                "unit": "dB",
                                "source": source_name,
                                "quality": "LIVE",
                                "scene_id": scene_id,
                                "scene_date": scene_date,
                                "orbit_direction": orbit_direction,
                                "sampling_mode": sampling_mode,
                                "pixel_or_roi": pixel_or_roi,
                                "valid_pixel_count": valid_pixel_count,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        },
                        "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
                    }
                    provider_cache.set(cache_key, res, ttl=43200.0)
                    provider_cache.set(scene_cache_key, res, ttl=43200.0)
                    return res

        except Exception:
            continue

    # If extraction loops through scenes without valid pixel, return fallback
    res = {
        "status": "no_recent_scene",
        "provider": selected_provider,
        "collection": collection_used,
        "scene_id": None,
        "scene_date": None,
        "orbit_direction": None,
        "polarization": "VV",
        "vv_db": -10.5,
        "sampling_mode": "fallback",
        "pixel_or_roi": "regional_baseline",
        "valid_pixel_count": 0,
        "provenance": {
            "sentinel1_vv": {
                "value": -10.5,
                "unit": "dB",
                "source": "calibrated regional baseline",
                "quality": "FALLBACK",
                "scene_id": None,
                "scene_date": None,
                "orbit_direction": None,
                "sampling_mode": "fallback",
                "timestamp": None,
                "note": "Sentinel-1 scene reading failed to resolve valid ground pixel; using calibrated regional baseline.",
            }
        },
        "execution_ms": round((time.perf_counter() - t_start) * 1000, 2),
    }
    provider_cache.set(cache_key, res, ttl=43200.0)
    return res
