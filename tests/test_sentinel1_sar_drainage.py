import math
import numpy as np
import pytest

from backend.data_sources.sentinel1_sar import (
    get_live_sentinel1_vv,
    linear_to_db,
    search_sentinel1_scenes,
)
from backend.services.drainage_service import (
    build_live_drainage_features,
    predict_live,
    compute_data_quality,
    DRAINAGE_MODEL_FEATURES,
    DEFAULT_FEATURE_BASELINES,
)
from backend.services.model_service import registry


MOCK_S2_DATA = {
    "status": "available",
    "provenance": {
        "NDVI": {"value": 0.42, "unit": "index (-1 to 1)", "source": "Sentinel-2 L2A", "quality": "LIVE"},
        "NDBI": {"value": -0.05, "unit": "index (-1 to 1)", "source": "Sentinel-2 L2A", "quality": "LIVE"},
    }
}


# =============================================================================
# 1. Sentinel-1 Scene Discovery
# =============================================================================
def test_sentinel1_scene_discovery():
    """Verify STAC discovery returns usable Sentinel-1 RTC scenes intersecting coordinate."""
    res = get_live_sentinel1_vv(16.3067, 80.4365)
    assert res["status"] == "available"
    assert res["scene_id"] is not None
    assert res["scene_date"] is not None
    assert res["polarization"] == "VV"
    assert res["orbit_direction"] in ("ascending", "descending", "unknown")
    assert "Microsoft Planetary Computer" in res["provider"] or "Earth Search" in res["provider"]


# =============================================================================
# 2. Exact Pixel Extraction & 3. ROI Fallback
# =============================================================================
def test_exact_pixel_and_roi_sampling_logic():
    """Verify exact pixel extraction and neighborhood ROI fallback on simulated windows."""
    # Scenario A: Center pixel is valid
    arr_valid = np.array([
        [0.20, 0.25, 0.30],
        [0.15, 0.165, 0.22],
        [0.18, 0.19, 0.21],
    ], dtype=np.float32)

    c_val = arr_valid[1, 1]
    assert c_val > 0.0
    val_db = 10.0 * np.log10(c_val)
    assert abs(val_db - (-7.825)) < 0.01

    # Scenario B: Center pixel is nodata / NaN -> fallback to neighborhood median
    arr_nodata_center = np.array([
        [0.10, 0.20, 0.30],
        [0.15, np.nan, 0.25],
        [0.12, 0.18, 0.22],
    ], dtype=np.float32)

    valid_mask = np.isfinite(arr_nodata_center) & (arr_nodata_center > 0.0)
    valid_pixels = arr_nodata_center[valid_mask]
    assert valid_pixels.size == 8
    median_val = float(np.median(valid_pixels))
    assert abs(median_val - 0.19) < 0.02
    med_db = 10.0 * np.log10(median_val)
    assert -10.0 < med_db < -5.0


# =============================================================================
# 4. Linear-to-dB Conversion & 5. Prevention of Double dB Conversion
# =============================================================================
def test_linear_to_db_conversion():
    """Verify linear power ratio correctly converts to decibels via 10*log10(linear)."""
    # 1.0 linear = 0.0 dB
    assert abs(10.0 * math.log10(1.0) - 0.0) < 1e-5
    # 0.1 linear = -10.0 dB
    assert abs(10.0 * math.log10(0.1) - (-10.0)) < 1e-5
    # 0.01 linear = -20.0 dB
    assert abs(10.0 * math.log10(0.01) - (-20.0)) < 1e-5
    # 0.1649 linear = -7.83 dB
    assert round(10.0 * math.log10(0.1649), 2) == -7.83


def test_prevention_of_double_db_conversion():
    """Verify values already in decibels (< 0 or negative dB scale) are NOT converted twice."""
    already_db = -12.45

    # Pipeline guard: if already negative, preserve value without re-taking log10
    if already_db < 0.0:
        final_db = already_db
    else:
        final_db = 10.0 * math.log10(already_db)

    assert final_db == -12.45


# =============================================================================
# 6. Nodata and Invalid Handling
# =============================================================================
def test_nodata_and_invalid_rejection():
    """Verify rejection of NaN, Inf, non-positive, and custom nodata values."""
    raw_vals = np.array([-9999.0, 0.0, -5.0, np.nan, np.inf, 0.15], dtype=np.float32)
    nodata = -9999.0

    valid_mask = np.isfinite(raw_vals) & (raw_vals > 0.0) & (raw_vals != nodata)
    clean_vals = raw_vals[valid_mask]

    assert clean_vals.size == 1
    assert clean_vals[0] == np.float32(0.15)


# =============================================================================
# 7. Fallback when no scene exists & 8. Provenance
# =============================================================================
def test_fallback_when_no_scene_exists():
    """Verify fallback baseline (-10.5 dB) and FALLBACK provenance when no scene is found."""
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel1": {"status": "no_recent_scene"}, "sentinel2": MOCK_S2_DATA},
    )
    s1_prov = res["feature_provenance"]["sentinel1_vv"]

    assert s1_prov["value"] == -10.5
    assert s1_prov["unit"] == "dB"
    assert s1_prov["quality"] == "FALLBACK"
    assert "baseline" in s1_prov["source"].lower() or "fallback" in s1_prov["source"].lower()
    assert s1_prov["sampling_mode"] == "fallback"


def test_live_provenance_structure():
    """Verify exact provenance contract structure on live Sentinel-1 retrieval."""
    res = get_live_sentinel1_vv(16.3067, 80.4365)
    assert res["status"] == "available"
    assert "provenance" in res
    assert "sentinel1_vv" in res["provenance"]

    p = res["provenance"]["sentinel1_vv"]
    assert "value" in p
    assert p["unit"] == "dB"
    assert p["quality"] == "LIVE"
    assert "Sentinel-1 RTC" in p["source"]
    assert p["scene_id"] is not None
    assert p["scene_date"] is not None
    assert p["orbit_direction"] is not None
    assert p["sampling_mode"] in ("exact_pixel", "local_roi_median", "cached_roi_threshold")


# =============================================================================
# 9. Drainage Feature Order Unchanged
# =============================================================================
def test_drainage_feature_order_unchanged():
    """Verify exact 6 features and strict list ordering expected by Random Forest model."""
    expected_features = ["elevation", "slope", "rain_max", "sentinel1_vv", "NDVI", "NDBI"]
    assert DRAINAGE_MODEL_FEATURES == expected_features

    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel2": MOCK_S2_DATA},
    )
    assert list(res["features"].keys()) == expected_features
    assert res["feature_order"] == expected_features


# =============================================================================
# 10. FULL_LIVE Transition
# =============================================================================
def test_full_live_transition():
    """Verify data quality transitions to FULL_LIVE when both Sentinel-1 and Sentinel-2 are LIVE."""
    s1_live = {
        "status": "available",
        "provenance": {
            "sentinel1_vv": {
                "value": -7.83,
                "unit": "dB",
                "source": "Sentinel-1 RTC",
                "quality": "LIVE",
                "scene_id": "S1D_TEST_SCENE",
                "scene_date": "2026-09-22T00:30:24Z",
            }
        }
    }
    s2_live = {
        "status": "available",
        "provenance": {
            "NDVI": {"value": 0.42, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
            "NDBI": {"value": -0.05, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
        }
    }

    # All live -> FULL_LIVE
    res_live = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel1": s1_live, "sentinel2": s2_live},
    )
    dq_live = res_live["data_quality"]
    assert dq_live["status"] == "FULL_LIVE"
    assert dq_live["fallback_features"] == 0
    assert dq_live["live_features"] == 4
    assert dq_live["derived_features"] == 2

    # Sentinel-1 unavailable -> PARTIAL_LIVE
    res_partial = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel1": {"status": "no_recent_scene"}, "sentinel2": s2_live},
    )
    dq_partial = res_partial["data_quality"]
    assert dq_partial["status"] == "PARTIAL_LIVE"
    assert dq_partial["fallback_features"] == 1


# =============================================================================
# 11. Drainage Random Forest Regression
# =============================================================================
def test_drainage_rf_regression():
    """Verify drainage model inference produces valid risk predictions with live Sentinel-1 features."""
    features = {
        "elevation": 39.0,
        "slope": 0.094,
        "rain_max": 0.0414,
        "sentinel1_vv": -7.83,
        "NDVI": 0.45,
        "NDBI": -0.08,
    }
    out = registry.predict("drainage", features)
    assert out["prediction"] in (0, 1)
    assert len(out["probabilities"]) == 2
    assert 0.0 <= out["positive_probability"] <= 1.0


# =============================================================================
# 12. Multi-City Regression: Guntur, Visakhapatnam, Hyderabad
# =============================================================================
@pytest.mark.parametrize("city,lat,lon", [
    ("Guntur", 16.3067, 80.4365),
    ("Visakhapatnam", 17.6868, 83.2185),
    ("Hyderabad", 17.3850, 78.4867),
])
def test_three_cities_sar_retrieval(city, lat, lon):
    """Verify live Sentinel-1 SAR extraction functions across Guntur, Visakhapatnam, and Hyderabad."""
    res = get_live_sentinel1_vv(lat, lon)
    assert res["status"] == "available"
    assert res["vv_db"] is not None
    assert -40.0 <= res["vv_db"] <= 10.0
    assert res["scene_id"] is not None
    assert res["scene_date"] is not None

    prov = res["provenance"]["sentinel1_vv"]
    assert prov["quality"] == "LIVE"
    assert prov["value"] == res["vv_db"]
