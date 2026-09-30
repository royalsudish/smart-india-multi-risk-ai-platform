import math
import numpy as np
import pytest

from backend.data_sources.sentinel2_optical import (
    calculate_indices,
    get_live_sentinel2_indices,
    SCL_INVALID_CLASSES,
    SCL_VALID_CLASSES,
)
from backend.services.drainage_service import (
    build_live_drainage_features,
    predict_live,
    DRAINAGE_MODEL_FEATURES,
    DEFAULT_FEATURE_BASELINES,
)
from backend.services.model_service import registry


def test_ndvi_calculation():
    """Verify standard NDVI = (B08 - B04) / (B08 + B04) formula and ranges."""
    # Healthy green vegetation: high NIR (B08), low Red (B04)
    ndvi_veg, _ = calculate_indices(b04_val=1000.0, b08_val=4000.0, b11_val=2000.0)
    assert ndvi_veg is not None
    assert abs(ndvi_veg - (3000.0 / 5000.0)) < 1e-5
    assert abs(ndvi_veg - 0.60) < 1e-5

    # Water surface: low NIR (B08), higher Red (B04)
    ndvi_water, _ = calculate_indices(b04_val=1000.0, b08_val=500.0, b11_val=200.0)
    assert ndvi_water is not None
    assert abs(ndvi_water - (-500.0 / 1500.0)) < 1e-5
    assert abs(ndvi_water - (-0.3333)) < 1e-3


def test_ndbi_calculation():
    """Verify standard NDBI = (B11 - B08) / (B11 + B08) formula and ranges."""
    # Built-up surface: high SWIR (B11), lower NIR (B08)
    _, ndbi_urban = calculate_indices(b04_val=2000.0, b08_val=2500.0, b11_val=3500.0)
    assert ndbi_urban is not None
    assert abs(ndbi_urban - (1000.0 / 6000.0)) < 1e-5
    assert abs(ndbi_urban - 0.1667) < 1e-3

    # Dense vegetation: low SWIR (B11), high NIR (B08)
    _, ndbi_veg = calculate_indices(b04_val=800.0, b08_val=4000.0, b11_val=1500.0)
    assert ndbi_veg is not None
    assert abs(ndbi_veg - (-2500.0 / 5500.0)) < 1e-5
    assert abs(ndbi_veg - (-0.4545)) < 1e-3


def test_divide_by_zero_and_edge_cases():
    """Verify safety against zero denominator, negative reflectance, and non-finite inputs."""
    # Zero denominator
    ndvi, ndbi = calculate_indices(0.0, 0.0, 0.0)
    assert ndvi is None
    assert ndbi is None

    # Negative reflectance
    ndvi_neg, ndbi_neg = calculate_indices(-10.0, 50.0, 50.0)
    assert ndvi_neg is None
    assert ndbi_neg is None

    # Non-finite values
    ndvi_inf, ndbi_inf = calculate_indices(float("inf"), 100.0, 100.0)
    assert ndvi_inf is None
    assert ndbi_inf is None

    ndvi_nan, ndbi_nan = calculate_indices(float("nan"), 100.0, 100.0)
    assert ndvi_nan is None
    assert ndbi_nan is None


def test_cloudy_pixel_fallback_logic():
    """Verify that cloud classes are correctly classified and rejected."""
    # SCL invalid classes (clouds, shadows, cirrus, nodata)
    for c in [0, 1, 3, 8, 9, 10]:
        assert c in SCL_INVALID_CLASSES

    # SCL valid surface classes (vegetation, bare soil, water)
    for c in [2, 4, 5, 6, 7]:
        assert c in SCL_VALID_CLASSES


def test_valid_sentinel2_live_extraction_guntur():
    """Verify real live Sentinel-2 extraction on Guntur, AP."""
    res = get_live_sentinel2_indices(16.3067, 80.4365)
    assert res["status"] == "available"
    assert "scene_id" in res
    assert "scene_date" in res
    assert "cloud_cover" in res
    assert res["cloud_cover"] <= 60.0
    assert -1.0 <= res["ndvi"] <= 1.0
    assert -1.0 <= res["ndbi"] <= 1.0

    prov = res["provenance"]
    assert prov["NDVI"]["quality"] == "LIVE"
    assert prov["NDBI"]["quality"] == "LIVE"
    assert "Sentinel-2" in prov["NDVI"]["source"]
    assert prov["NDVI"]["scene_id"] == res["scene_id"]


def test_fallback_when_no_scene_exists():
    """Verify fallback values (0.42, -0.05) are used with FALLBACK provenance when unavailable."""
    # Pass a simulated precomputed unavailable sentinel2 object
    unavailable_s2 = {
        "status": "no_recent_scene",
        "message": "Simulated cloud obscuration across all scenes",
    }
    features_payload = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel2": unavailable_s2},
    )

    prov = features_payload["feature_provenance"]
    assert prov["NDVI"]["value"] == DEFAULT_FEATURE_BASELINES["NDVI"]
    assert prov["NDVI"]["quality"] == "FALLBACK"
    assert "baseline" in prov["NDVI"]["source"].lower()

    assert prov["NDBI"]["value"] == DEFAULT_FEATURE_BASELINES["NDBI"]
    assert prov["NDBI"]["quality"] == "FALLBACK"
    assert "baseline" in prov["NDBI"]["source"].lower()


def test_drainage_feature_order_and_provenance():
    """Verify that exact 6 features and order are strictly preserved."""
    live_res = predict_live(lat=16.3067, lon=80.4365, include_osm=False)
    assert live_res["success"] is True

    # Exact feature names and order
    assert list(live_res["features"].keys()) == DRAINAGE_MODEL_FEATURES
    assert live_res["features"]["NDVI"] == live_res["feature_provenance"]["NDVI"]["value"]
    assert live_res["features"]["NDBI"] == live_res["feature_provenance"]["NDBI"]["value"]

    # Overall data quality status
    dq = live_res["data_quality"]
    # With NDVI and NDBI live, data quality is PARTIAL_LIVE (or FULL_LIVE if S1 also available)
    assert dq["status"] in ("PARTIAL_LIVE", "FULL_LIVE")
    assert dq["live_features"] >= 3  # elevation, NDVI, NDBI


def test_drainage_model_inference_regression():
    """Verify that Random Forest inference executes accurately with live Sentinel-2 features."""
    live_res = predict_live(lat=16.3067, lon=80.4365, include_osm=False)
    pred = live_res["prediction"]

    assert pred["prediction"] in (0, 1)
    assert 0.0 <= pred["positive_probability"] <= 1.0
    assert pred["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "class_probabilities" in pred
    assert len(pred["used_features"]) == 6


def test_override_priority():
    """Verify that caller overrides take precedence over live Sentinel-2 extraction."""
    custom_ndvi = 0.777
    custom_ndbi = -0.333
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        overrides={"NDVI": custom_ndvi, "NDBI": custom_ndbi},
    )

    assert res["features"]["NDVI"] == custom_ndvi
    assert res["features"]["NDBI"] == custom_ndbi
    assert res["feature_provenance"]["NDVI"]["quality"] == "OVERRIDE"
    assert res["feature_provenance"]["NDBI"]["quality"] == "OVERRIDE"


def test_spectral_consistency_formula_check():
    """Verify strict numerical reproducibility of NDVI and NDBI from exposed bands_used."""
    for lat, lon in [(16.3067, 80.4365), (17.6868, 83.2185), (17.3850, 78.4867)]:
        res = get_live_sentinel2_indices(lat, lon)
        assert res["status"] == "available"
        assert "bands_used" in res
        assert "B04" in res["bands_used"]
        assert "B08" in res["bands_used"]
        assert "B11" in res["bands_used"]
        assert "pixel_or_roi" in res
        assert "sampling_mode" in res
        assert "raw_bands_mean" in res

        bu = res["bands_used"]
        recomputed_ndvi = (bu["B08"] - bu["B04"]) / (bu["B08"] + bu["B04"])
        recomputed_ndbi = (bu["B11"] - bu["B08"]) / (bu["B11"] + bu["B08"])

        # Assert strict numerical equivalence (< 1e-6)
        assert abs(res["NDVI"]["value"] - recomputed_ndvi) < 1e-6
        assert abs(res["NDBI"]["value"] - recomputed_ndbi) < 1e-6
        assert abs(res["NDVI"]["formula_check"] - recomputed_ndvi) < 1e-6
        assert abs(res["NDBI"]["formula_check"] - recomputed_ndbi) < 1e-6

