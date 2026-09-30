import os
import sys
import json
import time
import requests

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data_sources.sentinel2_optical import get_live_sentinel2_indices
from backend.services.drainage_service import predict_live, DRAINAGE_MODEL_FEATURES

BASE_URL = "http://127.0.0.1:5000"

TEST_LOCATIONS = [
    {"name": "Guntur, Andhra Pradesh (Primary Test)", "lat": 16.3067, "lon": 80.4365},
    {"name": "Visakhapatnam, Andhra Pradesh (Coastal)", "lat": 17.6868, "lon": 83.2185},
    {"name": "Hyderabad, Telangana (Inland)", "lat": 17.3850, "lon": 78.4867},
]


def run_verification():
    print("=" * 80)
    print("SENTINEL-2 LIVE OPTICAL DATA SOURCE & DRAINAGE INTEGRATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: Multi-Location Live Sentinel-2 Extraction
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Multi-Location Live Sentinel-2 L2A Extractions ---")
    for loc in TEST_LOCATIONS:
        print(f"\nEvaluating: {loc['name']} (lat={loc['lat']}, lon={loc['lon']})...")
        t0 = time.time()
        s2_res = get_live_sentinel2_indices(loc["lat"], loc["lon"])
        elapsed = time.time() - t0

        print(f"  [OK] Status: {s2_res.get('status')}")
        print(f"  [OK] Provider: {s2_res.get('provider')}")
        print(f"  [OK] Scene ID: {s2_res.get('scene_id')}")
        print(f"  [OK] Scene Date: {s2_res.get('scene_date')}")
        print(f"  [OK] Cloud Cover: {s2_res.get('cloud_cover')}%")
        print(f"  [OK] NDVI: {s2_res.get('ndvi')} (exact: {s2_res.get('NDVI', {}).get('value')})")
        print(f"  [OK] NDBI: {s2_res.get('ndbi')} (exact: {s2_res.get('NDBI', {}).get('value')})")
        print(f"  [OK] Sampling Mode: {s2_res.get('sampling_mode')}")
        print(f"  [OK] Pixel/ROI: {s2_res.get('pixel_or_roi')}")
        print(f"  [OK] Bands Used: {s2_res.get('bands_used')}")
        print(f"  [OK] Latency: {elapsed:.2f}s (cached={s2_res.get('cache_hit')})")

        assert s2_res.get("status") == "available", f"Extraction failed for {loc['name']}"
        assert -1.0 <= s2_res.get("ndvi") <= 1.0, f"NDVI out of range: {s2_res.get('ndvi')}"
        assert -1.0 <= s2_res.get("ndbi") <= 1.0, f"NDBI out of range: {s2_res.get('ndbi')}"

        # Strict spectral consistency assertion
        bu = s2_res["bands_used"]
        recomputed_ndvi = (bu["B08"] - bu["B04"]) / (bu["B08"] + bu["B04"])
        recomputed_ndbi = (bu["B11"] - bu["B08"]) / (bu["B11"] + bu["B08"])
        diff_ndvi = abs(s2_res["NDVI"]["value"] - recomputed_ndvi)
        diff_ndbi = abs(s2_res["NDBI"]["value"] - recomputed_ndbi)
        assert diff_ndvi < 1e-6, f"{loc['name']} NDVI diff {diff_ndvi} >= 1e-6"
        assert diff_ndbi < 1e-6, f"{loc['name']} NDBI diff {diff_ndbi} >= 1e-6"
        print(f"  [OK] Formula Consistency: diff_ndvi={diff_ndvi:.2e}, diff_ndbi={diff_ndbi:.2e} (< 1e-6)")

    print("\n>>> PASS: Multi-location Sentinel-2 extractions succeeded with verified numerical consistency.")

    # -------------------------------------------------------------------------
    # PART 2: Live Drainage Feature Pipeline & RF Inference
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Drainage Feature Pipeline & Provenance (Guntur, AP) ---")
    g_res = predict_live(16.3067, 80.4365, include_osm=False)
    assert g_res["success"] is True

    print("Model Input Features (Strict 6F):")
    for k in DRAINAGE_MODEL_FEATURES:
        prov = g_res["feature_provenance"].get(k, {})
        print(f"  - {k:14s}: {g_res['features'][k]} [{prov.get('quality')}] (Source: {prov.get('source')})")

    dq = g_res["data_quality"]
    print(f"\nOverall Data Quality Status: {dq['status']}")
    print(f"  - Live Features:     {dq['live_features']}")
    print(f"  - Derived Features:  {dq['derived_features']}")
    print(f"  - Fallback Features: {dq['fallback_features']}")

    pred = g_res["prediction"]
    print(f"\nRandom Forest Prediction:")
    print(f"  - Class: {pred['prediction']} ({pred['prediction_label']})")
    print(f"  - Risk:  {pred['risk_level']} (positive_probability={pred['positive_probability']})")
    print(f"  - Probabilities: {pred['class_probabilities']}")

    assert dq["status"] in ("PARTIAL_LIVE", "FULL_LIVE"), f"Expected PARTIAL_LIVE or FULL_LIVE, got {dq['status']}"
    assert g_res["feature_provenance"]["NDVI"]["quality"] == "LIVE", "NDVI quality is not LIVE"
    assert g_res["feature_provenance"]["NDBI"]["quality"] == "LIVE", "NDBI quality is not LIVE"
    print(">>> PASS: Drainage Random Forest correctly ingests live Sentinel-2 NDVI and NDBI.")

    # -------------------------------------------------------------------------
    # PART 3: Live Flask HTTP API Verification
    # -------------------------------------------------------------------------
    print("\n--- PART 3: Live HTTP API Endpoints ---")

    # A. Dedicated Sentinel-2 endpoint: GET /api/waterlogging/optical/sentinel2
    print("Testing GET /api/waterlogging/optical/sentinel2...")
    r = requests.get(f"{BASE_URL}/api/waterlogging/optical/sentinel2?lat=16.3067&lon=80.4365")
    assert r.status_code == 200, f"Optical route returned {r.status_code}"
    s2_api = r.json().get("data", {})
    assert s2_api.get("status") == "available"
    print(f"  [OK] /api/waterlogging/optical/sentinel2 -> 200 OK (scene: {s2_api.get('scene_id')}, NDVI: {s2_api.get('ndvi')})")

    # B. Live Drainage Endpoint: GET /api/waterlogging/drainage
    print("Testing GET /api/waterlogging/drainage?lat=16.3067&lon=80.4365...")
    r = requests.get(f"{BASE_URL}/api/waterlogging/drainage?lat=16.3067&lon=80.4365&radius_m=2000")
    assert r.status_code == 200, f"Drainage route returned {r.status_code}"
    wl_data = r.json().get("data", {})
    assert "sentinel2_metadata" in wl_data or "satellite" in wl_data
    assert "model_prediction" in wl_data
    print(f"  [OK] /api/waterlogging/drainage -> 200 OK (Data Quality: {wl_data.get('data_quality', {}).get('status')})")

    # C. POST /api/predict/drainage/live
    print("Testing POST /api/predict/drainage/live...")
    r = requests.post(f"{BASE_URL}/api/predict/drainage/live", json={"lat": 16.3067, "lon": 80.4365})
    assert r.status_code == 200
    dl_data = r.json().get("data", {})
    assert dl_data["feature_provenance"]["NDVI"]["quality"] == "LIVE"
    assert dl_data["feature_provenance"]["NDBI"]["quality"] == "LIVE"
    print(f"  [OK] /api/predict/drainage/live -> 200 OK (NDVI live: {dl_data['features']['NDVI']}, NDBI live: {dl_data['features']['NDBI']})")

    # D. Zero Regression on Unified Endpoint: POST /api/predict/all/live
    print("Testing POST /api/predict/all/live (Core Platform Regression Check)...")
    r = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"})
    assert r.status_code == 200
    all_res = r.json()
    assert all_res.get("weather", {}).get("success") is True
    assert all_res.get("waterlogging", {}).get("success") is True
    assert all_res.get("agriculture", {}).get("success") is True
    assert all_res.get("disaster", {}).get("success") is True
    print("  [OK] POST /api/predict/all/live -> 200 OK (All 4 Core Models Working)")

    print("\n" + "=" * 80)
    print("ALL SENTINEL-2 INTEGRATION VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
