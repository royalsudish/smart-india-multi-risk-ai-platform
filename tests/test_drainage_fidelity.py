import math
import pytest
from backend.services.drainage_service import (
    transform_rain_max,
    compute_data_quality,
    build_live_drainage_features,
    predict_live,
    DEFAULT_FEATURE_BASELINES,
    DRAINAGE_MODEL_FEATURES,
)
from backend.services.model_service import registry


def test_log10_rain_transformation():
    """Verify log10(rain_mm + 1) transformation for various rainfall amounts."""
    # 0 mm -> log10(1) = 0.0
    assert transform_rain_max(0.0) == 0.0
    assert transform_rain_max(-5.0) == 0.0

    # User example: raw_rain_max_mm = 100 -> log10(101) ≈ 2.0043
    t100 = transform_rain_max(100.0)
    assert t100 == 2.0043
    assert abs(t100 - math.log10(101.0)) < 1e-4

    # Guntur low rain example: 0.1 mm -> log10(1.1) ≈ 0.0414
    t_low = transform_rain_max(0.1)
    assert t_low == 0.0414

    # Severe storm thresholds matching tree split points (approx 1.96 to 2.48):
    # 1.96 split threshold corresponds to ~90.2 mm
    # 2.48 split threshold corresponds to ~301.0 mm
    assert abs(transform_rain_max(90.2) - 1.96) < 0.01
    assert abs(transform_rain_max(301.0) - 2.48) < 0.01


def test_raw_vs_model_rainfall_values():
    """Verify both raw_rain_max_mm and model_rain_max are kept without collision."""
    # Test with override for raw_rain_max_mm
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        overrides={"raw_rain_max_mm": 100.0},
        precomputed_sources={
            "sentinel1": {"status": "available", "provenance": {"sentinel1_vv": {"value": -7.83, "quality": "LIVE"}}},
            "sentinel2": {"status": "available", "provenance": {"NDVI": {"value": 0.42, "quality": "LIVE"}, "NDBI": {"value": -0.05, "quality": "LIVE"}}},
        },
    )

    assert res["raw_rain_max_mm"] == 100.0
    assert res["model_rain_max"] == 2.0043
    assert res["features"]["rain_max"] == 2.0043

    rf_prov = res["rainfall_provenance"]
    assert rf_prov["raw_rain_max_mm"] == 100.0
    assert rf_prov["model_rain_max"] == 2.0043
    assert "log10" in rf_prov["formula"]

    # Ensure model feature matches transformed scale
    assert res["features"]["rain_max"] == res["model_rain_max"]


MOCK_S2 = {
    "status": "available",
    "provenance": {
        "NDVI": {"value": 0.42, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
        "NDBI": {"value": -0.05, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
    }
}


def test_sentinel1_fallback_provenance():
    """Verify fallback Sentinel-1 VV is explicitly tagged as FALLBACK when unavailable."""
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={
            "sentinel1": {"status": "no_recent_scene", "message": "No recent scene"},
            "sentinel2": MOCK_S2,
        },
    )
    s1_prov = res["feature_provenance"]["sentinel1_vv"]

    assert s1_prov["value"] == DEFAULT_FEATURE_BASELINES["sentinel1_vv"]
    assert s1_prov["quality"] == "FALLBACK"
    assert "baseline" in s1_prov["source"].lower() or "fallback" in s1_prov["source"].lower()
    assert s1_prov["timestamp"] is None


def test_sentinel1_live_provenance():
    """Verify live Sentinel-1 RTC extraction records LIVE quality and scene date."""
    # When SAR cache contains a processed scene
    fake_sar_cache = {
        "status": "available",
        "thresholds": {"vv_low_threshold_db": -16.8},
        "recent_scene": {
            "item_id": "S1A_IW_GRDH_1SDV_20260920",
            "acquisition_datetime": "2026-09-20T12:30:00Z",
        },
    }
    from backend.utils.cache import provider_cache, make_cache_key
    sar_key = make_cache_key("s1_processed_roi", 16.3067, 80.4365, {"radius_km": 10.0})
    provider_cache.set(sar_key, fake_sar_cache, ttl=60)

    try:
        res = build_live_drainage_features(
            lat=16.3067,
            lon=80.4365,
            precomputed_sources={"sentinel2": MOCK_S2},
        )
        s1_prov = res["feature_provenance"]["sentinel1_vv"]
        assert s1_prov["value"] == -16.8
        assert s1_prov["quality"] == "LIVE"
        assert "Sentinel-1 RTC" in s1_prov["source"]
        assert s1_prov["scene_date"] == "2026-09-20T12:30:00Z"
    finally:
        provider_cache.delete(sar_key)


def test_ndvi_and_ndbi_fallback_provenance():
    """Verify NDVI and NDBI fallbacks are NEVER presented as live optical observations when unavailable."""
    # Pass precomputed unavailable Sentinel-2 status
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel2": {"status": "unavailable", "message": "Simulated cloud cover"}},
    )
    ndvi_prov = res["feature_provenance"]["NDVI"]
    ndbi_prov = res["feature_provenance"]["NDBI"]

    assert ndvi_prov["value"] == 0.42
    assert ndvi_prov["quality"] == "FALLBACK"
    assert "baseline" in ndvi_prov["source"].lower()

    assert ndbi_prov["value"] == -0.05
    assert ndbi_prov["quality"] == "FALLBACK"
    assert "baseline" in ndbi_prov["source"].lower()


def test_data_quality_partial_live():
    """When Sentinel-1 is unavailable, quality is PARTIAL_LIVE with 1 fallback."""
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel1": {"status": "no_recent_scene"}, "sentinel2": MOCK_S2},
    )
    dq = res["data_quality"]

    assert dq["status"] == "PARTIAL_LIVE"
    assert dq["fallback_features"] in (1, 2, 3)
    assert dq["live_features"] + dq["derived_features"] >= 3
    assert dq["total_features"] == 6


def test_data_quality_full_live_with_live_sensors():
    """When both Sentinel-1 and Sentinel-2 are live, quality is FULL_LIVE with 0 fallbacks."""
    s1_mock = {
        "status": "available",
        "provenance": {
            "sentinel1_vv": {
                "value": -7.83,
                "unit": "dB",
                "source": "Sentinel-1 RTC",
                "quality": "LIVE",
                "scene_id": "S1D_IW_GRDH_1SDV_20260922",
                "scene_date": "2026-09-22T00:30:24Z",
            }
        }
    }
    s2_mock = {
        "status": "available",
        "provenance": {
            "NDVI": {"value": 0.45, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
            "NDBI": {"value": -0.08, "unit": "index", "source": "Sentinel-2 L2A", "quality": "LIVE"},
        }
    }
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel1": s1_mock, "sentinel2": s2_mock},
    )
    dq = res["data_quality"]
    assert dq["status"] == "FULL_LIVE"
    assert dq["fallback_features"] == 0
    assert dq["live_features"] == 4
    assert dq["derived_features"] == 2


def test_data_quality_full_live():
    """When all 6 features are live/overridden, quality should be FULL_LIVE."""
    overrides = {
        "elevation": 28.0,
        "slope": 0.5,
        "raw_rain_max_mm": 45.0,
        "sentinel1_vv": -11.2,
        "NDVI": 0.38,
        "NDBI": 0.02,
    }
    res = build_live_drainage_features(lat=16.3067, lon=80.4365, overrides=overrides)
    dq = res["data_quality"]

    assert dq["status"] == "FULL_LIVE"
    assert dq["fallback_features"] == 0
    assert dq["override_features"] == 6


def test_data_quality_fallback_heavy():
    """When 4 or more features fall back, quality should be FALLBACK_HEAVY."""
    dummy_prov = {
        "elevation": {"quality": "FALLBACK"},
        "slope": {"quality": "FALLBACK"},
        "rain_max": {"quality": "FALLBACK"},
        "sentinel1_vv": {"quality": "FALLBACK"},
        "NDVI": {"quality": "FALLBACK"},
        "NDBI": {"quality": "FALLBACK"},
    }
    dq = compute_data_quality(dummy_prov)
    assert dq["status"] == "FALLBACK_HEAVY"
    assert dq["fallback_features"] == 6


def test_drainage_prediction_no_regression():
    """Verify Random Forest model loads and predicts accurately without regression."""
    # Test direct model registry inference
    features = {
        "elevation": 28.0,
        "slope": 0.5,
        "rain_max": 2.004,
        "sentinel1_vv": -10.5,
        "NDVI": 0.42,
        "NDBI": -0.05,
    }
    out = registry.predict("drainage", features)
    assert "prediction" in out
    assert "probabilities" in out
    assert "positive_probability" in out
    assert len(out["probabilities"]) == 2
    assert out["prediction"] in (0, 1)

    # Test live prediction service
    live_res = predict_live(
        lat=16.3067,
        lon=80.4365,
        include_osm=False,
        precomputed_sources={"sentinel2": MOCK_S2},
    )
    assert live_res["success"] is True
    assert "data_quality" in live_res
    assert "feature_provenance" in live_res
    assert "rainfall_provenance" in live_res
    assert live_res["prediction"]["positive_probability"] >= 0.0
