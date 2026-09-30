import math
import sys
import os
import json
import requests

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services.drainage_service import (
    transform_rain_max,
    compute_data_quality,
    build_live_drainage_features,
    predict_live,
    DEFAULT_FEATURE_BASELINES,
)
from backend.services.model_service import registry

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 80)
    print("STARTING DRAINAGE DATA-FIDELITY AUDIT & VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: log10(rain_mm + 1) Transformation
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing log10(rain_mm + 1) rainfall transformation...")
    t0 = transform_rain_max(0.0)
    assert t0 == 0.0, f"Expected 0.0 for 0mm rain, got {t0}"

    t100 = transform_rain_max(100.0)
    expected_t100 = round(math.log10(101.0), 4)
    assert t100 == 2.0043, f"Expected 2.0043 for 100mm, got {t100}"
    assert abs(t100 - expected_t100) < 1e-4

    t_guntur = transform_rain_max(0.10)
    expected_t_guntur = round(math.log10(1.10), 4)
    assert t_guntur == 0.0414, f"Expected 0.0414 for 0.10mm, got {t_guntur}"

    print(f"  [OK] 0.0 mm   -> model_rain_max: {t0}")
    print(f"  [OK] 0.1 mm   -> model_rain_max: {t_guntur} (log10(1.10) ~= {expected_t_guntur})")
    print(f"  [OK] 100.0 mm -> model_rain_max: {t100} (log10(101) ~= {expected_t100})")
    print(">>> PASS: log10 transformation is exact and mathematically consistent.")

    # -------------------------------------------------------------------------
    # TEST 2: Raw vs Model Rainfall Values
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing raw_rain_max_mm vs model_rain_max exposure...")
    res = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        overrides={"raw_rain_max_mm": 100.0},
    )
    assert res["raw_rain_max_mm"] == 100.0
    assert res["model_rain_max"] == 2.0043
    assert res["features"]["rain_max"] == 2.0043
    assert res["rainfall_provenance"]["raw_rain_max_mm"] == 100.0
    assert res["rainfall_provenance"]["model_rain_max"] == 2.0043
    print(f"  [OK] raw_rain_max_mm = {res['raw_rain_max_mm']} mm")
    print(f"  [OK] model_rain_max  = {res['model_rain_max']}")
    print(f"  [OK] features['rain_max'] = {res['features']['rain_max']}")
    print(f"  [OK] formula = {res['rainfall_provenance']['formula']}")
    print(">>> PASS: Both raw and model rainfall values are preserved without collision.")

    # -------------------------------------------------------------------------
    # TEST 3: Sentinel-1 Fallback Provenance
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing Sentinel-1 fallback provenance...")
    res = build_live_drainage_features(lat=16.3067, lon=80.4365)
    s1_meta = res["feature_provenance"]["sentinel1_vv"]
    assert s1_meta["quality"] == "FALLBACK"
    assert "fallback" in s1_meta["source"].lower()
    print(f"  [OK] sentinel1_vv value: {s1_meta['value']} dB")
    print(f"  [OK] source: {s1_meta['source']}")
    print(f"  [OK] quality: {s1_meta['quality']}")
    print(">>> PASS: Sentinel-1 fallback is explicitly labelled and not masquerading as live.")

    # -------------------------------------------------------------------------
    # TEST 4: NDVI & NDBI Fallback Provenance
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing NDVI and NDBI baseline fallback provenance...")
    res_fb = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel2": {"status": "unavailable", "message": "Simulated cloud cover"}},
    )
    ndvi_meta = res_fb["feature_provenance"]["NDVI"]
    ndbi_meta = res_fb["feature_provenance"]["NDBI"]

    assert ndvi_meta["quality"] == "FALLBACK"
    assert ndvi_meta["value"] == 0.42
    assert "baseline" in ndvi_meta["source"].lower()

    assert ndbi_meta["quality"] == "FALLBACK"
    assert ndbi_meta["value"] == -0.05
    assert "baseline" in ndbi_meta["source"].lower()

    print(f"  [OK] NDVI: {ndvi_meta['value']}, source: '{ndvi_meta['source']}', quality: {ndvi_meta['quality']}")
    print(f"  [OK] NDBI: {ndbi_meta['value']}, source: '{ndbi_meta['source']}', quality: {ndbi_meta['quality']}")
    print(">>> PASS: NDVI and NDBI are clearly identified as regional baseline fallbacks.")

    # -------------------------------------------------------------------------
    # TEST 5: Data Quality Status (PARTIAL_LIVE vs FULL_LIVE vs FALLBACK_HEAVY)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing overall dataset quality statuses...")
    # Nominal retrieval
    nominal = build_live_drainage_features(lat=16.3067, lon=80.4365)
    dq_nom = nominal["data_quality"]
    assert dq_nom["status"] == "PARTIAL_LIVE"
    print(f"  [OK] Nominal live query: status={dq_nom['status']}, live={dq_nom['live_features']}, derived={dq_nom['derived_features']}, fallback={dq_nom['fallback_features']}")

    # Full live retrieval
    full_live = build_live_drainage_features(
        lat=16.3067,
        lon=80.4365,
        overrides={
            "elevation": 28.0,
            "slope": 0.5,
            "raw_rain_max_mm": 45.0,
            "sentinel1_vv": -11.2,
            "NDVI": 0.38,
            "NDBI": 0.02,
        },
    )
    dq_full = full_live["data_quality"]
    assert dq_full["status"] == "FULL_LIVE"
    print(f"  [OK] Full live overrides: status={dq_full['status']}, fallback={dq_full['fallback_features']}")

    # Heavy fallback
    dq_heavy = compute_data_quality({
        "elevation": {"quality": "FALLBACK"},
        "slope": {"quality": "FALLBACK"},
        "rain_max": {"quality": "FALLBACK"},
        "sentinel1_vv": {"quality": "FALLBACK"},
        "NDVI": {"quality": "FALLBACK"},
        "NDBI": {"quality": "FALLBACK"},
    })
    assert dq_heavy["status"] == "FALLBACK_HEAVY"
    print(f"  [OK] Heavy fallback: status={dq_heavy['status']}, fallback={dq_heavy['fallback_features']}")
    print(">>> PASS: Data Quality statuses correctly evaluated.")

    # -------------------------------------------------------------------------
    # TEST 6: Real Sentinel-1 RTC Provenance Simulation
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing real Sentinel-1 RTC extraction provenance...")
    from backend.utils.cache import provider_cache, make_cache_key
    sar_key = make_cache_key("s1_processed_roi", 16.3067, 80.4365, {"radius_km": 10.0})
    provider_cache.set(sar_key, {
        "status": "available",
        "thresholds": {"vv_low_threshold_db": -17.2},
        "recent_scene": {
            "item_id": "S1A_IW_GRDH_1SDV_20260925",
            "acquisition_datetime": "2026-09-25T00:30:00Z",
        },
    }, ttl=60)

    try:
        live_s1_res = build_live_drainage_features(lat=16.3067, lon=80.4365)
        s1_live = live_s1_res["feature_provenance"]["sentinel1_vv"]
        assert s1_live["quality"] == "LIVE"
        assert s1_live["value"] == -17.2
        assert "Sentinel-1 RTC" in s1_live["source"]
        assert s1_live["scene_date"] == "2026-09-25T00:30:00Z"
        print(f"  [OK] Sentinel-1 Live: value={s1_live['value']} dB, source='{s1_live['source']}', date={s1_live['scene_date']}")
    finally:
        provider_cache.delete(sar_key)
    print(">>> PASS: Real Sentinel-1 RTC extractions record LIVE provenance and scene date.")

    # -------------------------------------------------------------------------
    # TEST 7: HTTP API Endpoints & Regression Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing HTTP endpoints against live Flask server...")

    # A. POST /api/predict/drainage
    r_pred = requests.post(f"{BASE_URL}/api/predict/drainage", json={
        "features": {
            "elevation": 27.5,
            "slope": 0.45,
            "rain_max": 2.004,
            "sentinel1_vv": -10.5,
            "NDVI": 0.42,
            "NDBI": -0.05,
        }
    })
    assert r_pred.status_code == 200, f"POST /api/predict/drainage returned {r_pred.status_code}"
    p_json = r_pred.json()
    assert p_json.get("success") is True
    print(f"  [OK] POST /api/predict/drainage -> 200 OK, pred: {p_json.get('data', {}).get('prediction')}")

    # B. POST /api/predict/drainage/live (Guntur, AP)
    r_live = requests.post(f"{BASE_URL}/api/predict/drainage/live", json={"lat": 16.3067, "lon": 80.4365})
    assert r_live.status_code == 200, f"POST /api/predict/drainage/live returned {r_live.status_code}"
    l_json = r_live.json().get("data", {})
    assert "data_quality" in l_json, "data_quality missing from live prediction"
    assert "feature_provenance" in l_json, "feature_provenance missing"
    assert "rainfall_provenance" in l_json, "rainfall_provenance missing"
    assert "model_rain_max" in l_json, "model_rain_max missing"
    assert "raw_rain_max_mm" in l_json, "raw_rain_max_mm missing"
    print(f"  [OK] POST /api/predict/drainage/live -> 200 OK")
    print(f"    - raw_rain_max_mm: {l_json['raw_rain_max_mm']} mm")
    print(f"    - model_rain_max:  {l_json['model_rain_max']}")
    print(f"    - data_quality:    {l_json['data_quality']['status']}")
    print(f"    - prediction:      class={l_json['prediction']['prediction']}, prob={l_json['prediction']['positive_probability']}")

    # C. GET /api/waterlogging/drainage (Guntur, AP)
    r_wl = requests.get(f"{BASE_URL}/api/waterlogging/drainage?lat=16.3067&lon=80.4365&radius_m=2000")
    assert r_wl.status_code == 200, f"GET /api/waterlogging/drainage returned {r_wl.status_code}"
    wl_json = r_wl.json().get("data", {})
    assert "model_prediction" in wl_json
    assert "data_quality" in wl_json
    assert "feature_provenance" in wl_json
    assert "rainfall_provenance" in wl_json
    print(f"  [OK] GET /api/waterlogging/drainage -> 200 OK")
    print(f"    - Data Quality: {wl_json['data_quality']['status']}")
    print(f"    - Model Risk:   {wl_json['model_prediction']['risk_level']} ({wl_json['model_prediction']['positive_probability']})")

    # D. Zero Regression on Core Models: POST /api/predict/all/live
    r_all = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"})
    assert r_all.status_code == 200
    all_json = r_all.json()
    assert all_json.get("weather", {}).get("success") is True
    assert all_json.get("waterlogging", {}).get("success") is True
    assert all_json.get("agriculture", {}).get("success") is True
    assert all_json.get("disaster", {}).get("success") is True
    print("  [OK] POST /api/predict/all/live -> 200 OK (Weather, Waterlogging, Agriculture, Disaster all SUCCESS)")

    print("\n" + "=" * 80)
    print("ALL DRAINAGE FIDELITY AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
