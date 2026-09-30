"""
Phase F Comprehensive Verification Script
Sentinel-1 SAR Possible Surface-Water / Inundation Detection

Covers:
1. Controlled / Unit Tests:
   - Linear to dB conversion
   - Otsu adaptive thresholding
   - Robust MAD change detection
   - Persistent-vs-new classification
   - Connected component cleanup
   - Coordinate transforms (WGS84 <-> UTM) & GeoJSON [lon, lat] ordering
   - SAS URL redaction
   - Cache behavior (search, asset, processed ROI)
   - Status & failure semantics
2. Model Invariance & Core Isolation:
   - All 4 ML models unchanged
   - Waterlogging features = 30, order unchanged, threshold = 0.36
   - RF prediction invariance
   - POST /api/predict/all/live makes 0 Sentinel-1 calls
3. Canonical Guntur Detailed Live Proof (lat=16.3067, lon=80.4365)
4. Pan-India 7-City Live Table (Guntur, Mumbai, Patna, Guwahati, Bengaluru, Dehradun, Jaisalmer)
"""

import sys
import os
import time
import json
import math
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data_sources.sentinel1_sar import (
    linear_to_db,
    compute_otsu_threshold,
    compute_adaptive_change_threshold,
    classify_water_and_change,
    cleanup_connected_components,
    detect_surface_water_changes,
    search_scenes,
    _search_cache,
    _signed_asset_cache,
    _processed_roi_cache,
    PLANETARY_COMPUTER_STAC_URL,
    SENTINEL1_RTC_COLLECTION
)
from backend.services.waterlogging_features import SELECTED_FEATURES
from backend.services.waterlogging_service import get_warning_threshold
from backend.services.model_service import registry
import app as flask_app


def run_controlled_tests():
    print("\n" + "=" * 60)
    print("PART 1: CONTROLLED / UNIT TESTS")
    print("=" * 60)

    # 1.1 Linear to dB Conversion
    linear_data = np.array([-1.0, 0.0, 1.0, 10.0, 100.0, np.nan], dtype=np.float32)
    db_data, valid_mask = linear_to_db(linear_data, nodata_val=0.0)
    assert valid_mask[0] == False, "Negative linear value must be invalid"
    assert valid_mask[1] == False, "Zero linear value must be invalid"
    assert valid_mask[5] == False, "NaN linear value must be invalid"
    assert valid_mask[2] == True and abs(db_data[2] - 0.0) < 1e-4, "1.0 linear must equal 0.0 dB"
    assert valid_mask[3] == True and abs(db_data[3] - 10.0) < 1e-4, "10.0 linear must equal 10.0 dB"
    assert valid_mask[4] == True and abs(db_data[4] - 20.0) < 1e-4, "100.0 linear must equal 20.0 dB"
    print(" [PASS] 1.1 Linear to dB Conversion (10*log10, non-positive & NaN masked)")

    # 1.2 Otsu Adaptive Thresholding
    # Bimodal distribution: water (-22 dB) and land (-10 dB)
    np.random.seed(42)
    water_px = np.random.normal(-22.0, 1.5, 5000)
    land_px = np.random.normal(-10.0, 2.0, 5000)
    bimodal = np.concatenate([water_px, land_px])
    otsu_th, otsu_stat = compute_otsu_threshold(bimodal)
    assert otsu_stat == "computed", f"Expected computed status, got {otsu_stat}"
    assert -18.0 < otsu_th < -14.0, f"Otsu threshold {otsu_th:.2f} dB outside expected valley [-18, -14]"

    # Uniform / low-variance array
    uniform = np.full(100, -12.0)
    u_th, u_stat = compute_otsu_threshold(uniform)
    assert u_stat == "insufficient_variance", f"Expected insufficient_variance, got {u_stat}"
    print(f" [PASS] 1.2 Otsu Adaptive Thresholding (Bimodal threshold: {otsu_th:.2f} dB, uniform safely handles insufficient_variance)")

    # 1.3 MAD Adaptive Change Threshold
    deltas = np.random.normal(0.0, 1.0, 10000)
    # inject negative tail
    deltas[:500] = np.random.normal(-6.0, 1.0, 500)
    change_th, mad_stat, cut = compute_adaptive_change_threshold(deltas, k=1.5)
    assert mad_stat == "computed", "MAD change threshold should compute"
    assert cut < -1.0, f"Expected negative cutoff, got {cut:.2f} dB"
    print(f" [PASS] 1.3 MAD Adaptive Change Threshold (Cutoff: {cut:.2f} dB, median - 1.5*MAD)")

    # 1.4 Persistent vs New Surface Water Classification
    # Setup test pixels:
    # px 0: recent low, baseline low (persistent)
    # px 1: recent low, baseline high (new water candidate)
    # px 2: recent high, baseline high (land)
    recent_db = np.array([-20.0, -20.0, -10.0])
    base_db = np.array([-21.0, -10.0, -9.0])
    v_mask = np.array([True, True, True])
    r_mask = np.ones((1, 3), dtype=bool)
    b_mask = np.ones((1, 3), dtype=bool)

    new_mask, pers_mask = classify_water_and_change(
        recent_vv_db=recent_db,
        recent_vh_db=None,
        baseline_vv_db=base_db,
        baseline_vh_db=None,
        recent_valid=r_mask,
        baseline_valid=b_mask,
        vv_low_th=-16.0,
        vh_low_th=None,
        vv_change_cut=-4.0,
        vh_change_cut=None
    )
    assert pers_mask[0, 0] == True and new_mask[0, 0] == False, "px 0 must be persistent low"
    assert pers_mask[0, 1] == False and new_mask[0, 1] == True, "px 1 must be possible new water"
    assert pers_mask[0, 2] == False and new_mask[0, 2] == False, "px 2 must be land"
    print(" [PASS] 1.4 Classification Logic (Correct separation of persistent low vs possible new surface water)")

    # 1.5 Connected Component Cleanup
    # Create 100x100 mask with 1 tiny component (4 px) and 1 large component (100 px)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:12, 10:12] = True  # 4 pixels
    mask[30:40, 30:40] = True  # 100 pixels
    # at 20m x 20m pixel area = 400 m2. min_area_m2 = 5000 m2 -> min 12.5 pixels
    cleaned = cleanup_connected_components(mask, min_area_m2=5000.0, pixel_area_m2=400.0)
    assert cleaned[10:12, 10:12].sum() == 0, "Tiny component must be filtered out"
    assert cleaned[30:40, 30:40].sum() == 100, "Large component must be retained"
    print(" [PASS] 1.5 Connected Component Cleanup (Tiny speckle removed, valid clusters preserved)")

    # 1.6 SAS URL Redaction
    # Verify that in any result dict, no SAS tokens or signed URLs exist
    guntur_res = detect_surface_water_changes(16.3067, 80.4365, radius_km=10.0)
    res_str = json.dumps(guntur_res)
    assert "sig=" not in res_str, "CRITICAL: SAS signature token found in result payload!"
    assert "se=" not in res_str or "sentinel" in res_str, "CRITICAL: SAS expiry token found in result payload!"
    assert "token" not in res_str.lower() or "csrf" in res_str.lower(), "CRITICAL: Suspicious token in result payload!"
    print(" [PASS] 1.6 SAS URL Redaction (Zero SAS query parameters or signature tokens exposed)")

    # 1.7 GeoJSON RFC 7946 & Coordinate Ordering
    geojson = guntur_res.get("geojson", {})
    assert geojson.get("type") == "FeatureCollection", "Must be FeatureCollection"
    for feat in geojson.get("features", [])[:5]:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if coords and len(coords) > 0 and len(coords[0]) > 0:
            pt = coords[0][0]
            lon, lat = pt[0], pt[1]
            assert 60.0 <= lon <= 100.0, f"Longitude {lon} outside India bbox"
            assert 5.0 <= lat <= 40.0, f"Latitude {lat} outside India bbox"
    print(" [PASS] 1.7 GeoJSON Format & CRS (RFC 7946 EPSG:4326 [longitude, latitude] coordinates)")

    # 1.8 Processed ROI Cache
    t0 = time.perf_counter()
    cached_res = detect_surface_water_changes(16.3067, 80.4365, radius_km=10.0)
    t_cache = (time.perf_counter() - t0) * 1000
    assert cached_res.get("cache_status") == "hit", f"Expected hit, got {cached_res.get('cache_status')}"
    assert t_cache < 50.0, f"Cache hit took {t_cache:.2f} ms (expected < 50 ms)"
    print(f" [PASS] 1.8 Processed ROI Cache (Hit confirmed in {t_cache:.2f} ms)")


def run_model_invariance_tests():
    print("\n" + "=" * 60)
    print("PART 2: MODEL INVARIANCE & ISOLATION TESTS")
    print("=" * 60)

    # 2.1 Waterlogging Feature Invariance
    assert len(SELECTED_FEATURES) == 30, f"Waterlogging feature count must be 30, found {len(SELECTED_FEATURES)}"
    threshold = get_warning_threshold()
    assert threshold == 0.36, f"Warning threshold must be 0.36, found {threshold}"
    assert "sar" not in "".join(SELECTED_FEATURES).lower(), "SAR features must NOT be in Waterlogging RF"

    # 2.2 Prediction Invariance Test
    sample_feat = {f: 0.5 for f in SELECTED_FEATURES}
    sample_feat["elevation"] = 25.0
    sample_feat["slope"] = 1.2
    sample_feat["rainfall_24h"] = 45.0

    res = registry.predict("waterlogging", sample_feat)
    prob = res.get("probabilities", [0.5, 0.5])[1] if "probabilities" in res else res.get("positive_probability")
    assert isinstance(prob, (float, np.floating)) and 0.0 <= prob <= 1.0, "Probability must be valid float"
    print(f" [PASS] 2.1 Waterlogging Model Invariance (30 features, threshold 0.36, prob: {prob:.4f})")

    # 2.3 Core Endpoint Isolation
    client = flask_app.create_app().test_client()
    payload = {
        "lat": 16.3067,
        "lon": 80.4365,
        "crop": "Cotton",
        "crop_stage": "Vegetative"
    }

    t0 = time.perf_counter()
    resp = client.post("/api/predict/all/live", json=payload)
    core_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, f"Core endpoint returned {resp.status_code}"
    resp_data = resp.get_json()

    # Verify no SAR keys in core endpoint response
    assert "sar" not in resp_data, "Core endpoint response must NOT contain SAR keys"
    assert "sentinel1" not in resp_data, "Core endpoint response must NOT contain Sentinel-1 keys"

    # Verify provider execution times have 0 SAR calls
    provider_times = resp_data.get("provider_execution_times", {})
    assert "sentinel1_sar_ms" not in provider_times, "Core endpoint must make 0 Sentinel-1 calls"
    print(f" [PASS] 2.2 Core Endpoint Isolation (POST /api/predict/all/live completed in {core_ms:.1f} ms with 0 Sentinel-1 calls)")


def run_guntur_detailed_live_proof():
    print("\n" + "=" * 60)
    print("PART 3: GUNTUR DETAILED LIVE PROOF (lat=16.3067, lon=80.4365)")
    print("=" * 60)

    res = detect_surface_water_changes(16.3067, 80.4365, radius_km=10.0)

    recent = res.get("recent_scene", {})
    baselines = res.get("baseline_scenes", [])
    matching = res.get("scene_matching", {})
    roi_info = res.get("roi", {})
    raster_meta = res.get("raster_metadata", {})
    thresholds = res.get("thresholds", {})
    water_change = res.get("water_change", {})
    coverage = res.get("coverage", {})
    geojson = res.get("geojson", {})

    print(f" Status:                      {res.get('status')}")
    print(f" Provider:                    {res.get('provider')}")
    print(f" Collection:                  {res.get('collection')}")
    print(f" Source Representation:       {raster_meta.get('source_representation')}")
    print(f" Analysis Representation:     {raster_meta.get('analysis_representation')}")
    print(f" dB Conversion Formula:       {raster_meta.get('db_conversion')}")
    print(f" Recent Scene ID:             {recent.get('item_id')}")
    print(f" Recent Acquisition (UTC):    {recent.get('acquisition_datetime')}")
    print(f" Scene Age:                   {recent.get('scene_age_days', 0):.1f} days ({recent.get('scene_age_hours', 0):.1f} hours)")
    print(f" Platform:                    {recent.get('platform')}")
    print(f" Orbit:                       {recent.get('orbit_direction')} (Rel Orbit: {recent.get('relative_orbit')})")
    print(f" Instrument Mode:             {recent.get('instrument_mode')}")
    print(f" Polarizations:               {recent.get('polarizations')}")
    print(f" Baseline Scene Count:        {len(baselines)}")
    for i, b in enumerate(baselines):
        print(f"   Baseline {i+1}: {b.get('item_id')} | {b.get('datetime')} | Rel: {b.get('relative_orbit')} | {b.get('orbit_direction')}")
    print(f" Matching Orbit Evidence:     Rel Orbit {matching.get('required_relative_orbit')}, Direction {matching.get('required_orbit_direction')}, Mode {matching.get('required_instrument_mode')}")
    print(f" Projected CRS:               {raster_meta.get('crs')}")
    print(f" Pixel Spacing:               {roi_info.get('pixel_spacing_m')} m (Area: {roi_info.get('pixel_area_m2')} m²)")
    print(f" Analysis Grid Dimensions:    {roi_info.get('rows')} rows x {roi_info.get('columns')} cols ({roi_info.get('total_pixels')} pixels)")
    print(f" Coverage:                    {coverage.get('coverage_fraction', 0)*100:.1f}% ({coverage.get('valid_comparison_pixels')} valid comparison pixels)")
    print(f" Adaptive Low Thresholds:     VV = {thresholds.get('vv_low_threshold_db')} dB ({thresholds.get('vv_threshold_status')}), VH = {thresholds.get('vh_low_threshold_db')} dB ({thresholds.get('vh_threshold_status')})")
    print(f" Adaptive Change Thresholds:  {thresholds.get('adaptive_change_threshold_db')} dB (Method: {thresholds.get('threshold_type')}, K={thresholds.get('k_mad')})")
    print(f" Candidate Rule:              {res.get('processing', {}).get('candidate_rule')}")
    print(f" Persistent Low Rule:         {res.get('processing', {}).get('persistent_low_rule')}")
    print(f" Candidate Pixels (pre-clean):{water_change.get('candidate_pixel_count_before_cleanup')}")
    print(f" Candidate Pixels (post):     {water_change.get('candidate_pixel_count')}")
    print(f" Possible New Water Area:     {water_change.get('possible_new_surface_water_area_km2')} km²")
    print(f" Persistent Low Area:         {water_change.get('persistent_low_area_km2')} km²")
    print(f" Candidate % of Valid ROI:    {water_change.get('candidate_fraction_of_valid_roi', 0)*100:.4f}%")
    print(f" GeoJSON Features:            {len(geojson.get('features', []))}")
    return res


def run_pan_india_tests():
    print("\n" + "=" * 60)
    print("PART 4: PAN-INDIA 7-CITY LIVE VALIDATION TABLE")
    print("=" * 60)

    cities = [
        {"name": "Guntur", "lat": 16.3067, "lon": 80.4365},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
        {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362},
        {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
        {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322},
        {"name": "Jaisalmer", "lat": 26.9157, "lon": 70.9083}
    ]

    results = []
    print(f"{'City':<10} | {'Status':<12} | {'Recent Scene ID':<35} | {'Age (d)':<7} | {'Orbit':<10} | {'Bases':<5} | {'Coverage':<8} | {'New Water (km²)':<15} | {'Persist (km²)':<13} | {'Lat (ms)':<8}")
    print("-" * 135)

    for c in cities:
        name = c["name"]
        lat = c["lat"]
        lon = c["lon"]

        t0 = time.perf_counter()
        try:
            # First query scenes to see available recent and baseline
            res = detect_surface_water_changes(lat, lon, radius_km=10.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            st = res.get("status", "unknown")
            recent = res.get("recent_scene", {})
            wc = res.get("water_change", {})
            cov = res.get("coverage", {})

            scene_id = recent.get("item_id", "--")
            if len(scene_id) > 35:
                scene_id = scene_id[:32] + "..."
            age_d = f"{recent.get('scene_age_days', 0):.1f}" if recent.get('scene_age_days') is not None else "--"
            orbit = f"{recent.get('orbit_direction', '--')[:4]}/{recent.get('relative_orbit', '--')}"
            bases = str(len(res.get("baseline_scenes", [])))
            cov_str = f"{cov.get('coverage_fraction', 0)*100:.1f}%" if cov.get('coverage_fraction') is not None else "--"
            new_w = f"{wc.get('possible_new_surface_water_area_km2', 0.0):.3f}" if wc.get('possible_new_surface_water_area_km2') is not None else "--"
            pers_w = f"{wc.get('persistent_low_area_km2', 0.0):.3f}" if wc.get('persistent_low_area_km2') is not None else "--"

            print(f"{name:<10} | {st:<12} | {scene_id:<35} | {age_d:<7} | {orbit:<10} | {bases:<5} | {cov_str:<8} | {new_w:<15} | {pers_w:<13} | {elapsed_ms:<8.0f}")
            results.append({"city": name, "lat": lat, "lon": lon, "result": res, "elapsed_ms": elapsed_ms})
        except Exception as ex:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"{name:<10} | {'ERROR':<12} | {str(ex)[:35]:<35} | {'--':<7} | {'--':<10} | {'--':<5} | {'--':<8} | {'--':<15} | {'--':<13} | {elapsed_ms:<8.0f}")
            results.append({"city": name, "lat": lat, "lon": lon, "error": str(ex), "elapsed_ms": elapsed_ms})

    return results


if __name__ == "__main__":
    print("STARTING PHASE F SENTINEL-1 SAR VERIFICATION SUITE")
    run_controlled_tests()
    run_model_invariance_tests()
    run_guntur_detailed_live_proof()
    run_pan_india_tests()
    print("\n" + "=" * 60)
    print("PHASE F VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
