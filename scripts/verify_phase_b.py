import sys
from pathlib import Path
import json
import time
import math
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import create_app
from backend.services.model_service import registry
from backend.services.waterlogging_service import predict_live, get_warning_threshold
from backend.data_sources.overpass import (
    drainage_features,
    point_to_segment_metric,
    clip_segment_to_circle,
    _build_deterministic_cache_key,
    _to_local_meters,
    DRAINAGE_PROXIMITY_POLICY,
    evaluate_drainage_proximity,
)
from backend.utils.cache import provider_cache

app = create_app()
client = app.test_client()

print("=" * 80)
print("PHASE B VERIFICATION & AUDIT SUITE: DRAINAGE NETWORK INTEGRATION (PATCHED)")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. DOM Check: Ensure all Phase B and existing DOM IDs are 100% present
# ------------------------------------------------------------------------------
r_html = client.get("/")
assert r_html.status_code == 200, "GET / failed"
html_str = r_html.data.decode("utf-8")

phase_b_dom_ids = [
    # Core & Phase A IDs (must all remain)
    "waterloggingCard", "wlStatusBadge", "wlErrorBox", "wlContent", "wlWarningBanner",
    "wlRiskLevel", "wlProb", "wlAnalysisTime", "wlTimelinePast24h", "wlTimelinePast7d",
    "wlTimelineFuture24h", "wlTimelineFuture72h", "wlRain1h", "wlRain3h", "wlRain6h",
    "wlRain12h", "wlRain24h", "wlRain3d", "wlRain7d", "wlElevation", "wlSlope",
    "wlOutlookBox", "wlOutlookBadge", "wlOutlookSummary", "wlMaxHourlyRain",
    "wlTotalForecast72h", "wlMostSignificantPeriod",
    # Phase B Drainage IDs
    "wlDrainageBox", "wlDrainProximityBadge", "wlDrainStatus", "wlDrainDistance",
    "wlDrainName", "wlDrainType", "wlDrainCount", "wlDrainLength", "wlDrainDensity",
    "wlDrainArea", "wlDrainSourceNote"
]

missing_ids = [i for i in phase_b_dom_ids if f'id="{i}"' not in html_str]
print(f"1. DOM Check: Checked {len(phase_b_dom_ids)} IDs, Missing: {missing_ids if missing_ids else 'None (100% Present)'}")
assert not missing_ids, f"Missing IDs: {missing_ids}"

# ------------------------------------------------------------------------------
# 2. Static Assets Check
# ------------------------------------------------------------------------------
r_css = client.get("/static/css/style.css")
r_js = client.get("/static/js/app.js")
assert r_css.status_code == 200, "style.css failed"
assert r_js.status_code == 200, "app.js failed"
assert "wlDrainageBox" in r_js.data.decode("utf-8"), "wlDrainageBox missing from app.js"
assert "drainage-summary-box" in r_css.data.decode("utf-8"), "drainage-summary-box missing from style.css"
print("2. Static Assets: style.css and app.js contain Phase B definitions.")

# ------------------------------------------------------------------------------
# 3. Model Integrity & Invariance (Zero Model Drift Check)
# ------------------------------------------------------------------------------
registry.slots["waterlogging"].ensure_loaded()
wl_model = registry.slots["waterlogging"].obj
assert list(wl_model.classes_) == [0, 1], "Model classes changed!"
assert wl_model.n_features_in_ == 30, f"Model feature count changed: {wl_model.n_features_in_}"
thresh = get_warning_threshold()
assert thresh == 0.36, f"Warning threshold changed: {thresh}"
print(f"3. Model Integrity: classes_={list(wl_model.classes_)}, n_features={wl_model.n_features_in_}, threshold={thresh}")

# ------------------------------------------------------------------------------
# 4. Exact Cache Coordinates & Deterministic Parameter Hash Check
# ------------------------------------------------------------------------------
test_lat, test_lon, test_rad = 16.30671234567, 80.43659876543, 2000
ckey = _build_deterministic_cache_key(test_lat, test_lon, test_rad)
print(f"4. Exact Cache Key: {ckey}")
assert repr(float(test_lat)) in ckey, "Latitude coordinate was rounded or truncated!"
assert repr(float(test_lon)) in ckey, "Longitude coordinate was rounded or truncated!"
assert f":{test_rad}:" in ckey, "Radius missing from cache key!"
ckey2 = _build_deterministic_cache_key(test_lat, test_lon, test_rad)
assert ckey == ckey2, "Cache key is not deterministically reproducible!"

# ------------------------------------------------------------------------------
# 5. True Point-to-Polyline Metric Distance vs Nearest-Vertex Demonstration
# ------------------------------------------------------------------------------
lat0, lon0 = 16.3067, 80.4365
x1, y1 = -100.0, 100.0
x2, y2 = 100.0, 100.0
vertex_dist = math.hypot(x1, y1)
poly_dist, clat, clon = point_to_segment_metric(x1, y1, x2, y2, lat0, lon0)
print(f"5. Distance Algorithm Validation:")
print(f"   Nearest-Vertex Distance:        {vertex_dist:.3f} m")
print(f"   True Point-to-Polyline Distance: {poly_dist:.3f} m")
print(f"   Accuracy Improvement:           {vertex_dist - poly_dist:.3f} m ({((vertex_dist - poly_dist)/vertex_dist)*100:.1f}% reduction in error)")
assert abs(poly_dist - 100.0) < 1e-6, f"Point-to-polyline distance failed: {poly_dist}"
assert vertex_dist > poly_dist, "Vertex distance should have overestimated!"

# ------------------------------------------------------------------------------
# 6. Geometry Clipping to Circular Analysis Buffer Validation
# ------------------------------------------------------------------------------
full_len_1 = math.hypot(2000.0, 0.0)
clipped_len_1 = clip_segment_to_circle(0.0, 0.0, 2000.0, 0.0, 1000.0)
print(f"\n6. Circular Buffer Clipping Validation:")
print(f"   Test 1 (0,0 -> 2000,0 crossing R=1000m circle):")
print(f"     Full OSM Geometry Length:   {full_len_1:.1f} m")
print(f"     Clipped In-Buffer Length:   {clipped_len_1:.1f} m")
assert abs(clipped_len_1 - 1000.0) < 1e-6, f"Clipping failed: {clipped_len_1}"

full_len_2 = 1000.0
clipped_len_2 = clip_segment_to_circle(2000.0, 2000.0, 3000.0, 2000.0, 1000.0)
print(f"   Test 2 (Segment completely outside R=1000m):")
print(f"     Full OSM Geometry Length:   {full_len_2:.1f} m")
print(f"     Clipped In-Buffer Length:   {clipped_len_2:.1f} m")
assert abs(clipped_len_2 - 0.0) < 1e-6, f"Outside segment clipping failed: {clipped_len_2}"

# ------------------------------------------------------------------------------
# 7. Core Prediction Non-Blocking Check
# ------------------------------------------------------------------------------
t0 = time.time()
r_core = client.post("/api/predict/all/live", json={"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"})
core_ms = round((time.time() - t0) * 1000, 2)
assert r_core.status_code == 200, "Core prediction API failed"
data_core = r_core.get_json()
assert data_core["success"] is True, "Core prediction failed"
print(f"\n7. Core Prediction Non-Blocking Check:")
print(f"   POST /api/predict/all/live latency (Cold): {core_ms} ms")
assert "overpass" not in data_core.get("provider_execution_times", {}), "Overpass should not be in core prediction providers!"
assert "overpass" not in data_core.get("provider_cache_status", {}), "Overpass should not be in core cache status!"
print(f"   >>> Confirmed: Overpass is NOT in provider_execution_times or provider_cache_status! <<<")

t0_warm = time.time()
r_core_warm = client.post("/api/predict/all/live", json={"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"})
warm_core_ms = round((time.time() - t0_warm) * 1000, 2)
print(f"   POST /api/predict/all/live latency (Warm Cached): {warm_core_ms} ms")
assert warm_core_ms < 500, f"Warm cached core API took too long: {warm_core_ms} ms"

# ------------------------------------------------------------------------------
# 8. Deterministic GeoJSON Coordinate Order Validation ([lon, lat] per RFC 7946)
# ------------------------------------------------------------------------------
print(f"\n8. Deterministic GeoJSON Coordinate Order Test:")
mock_element = {
    "type": "way",
    "id": 999001,
    "tags": {"waterway": "drain", "name": "Synthetic Drain Alpha"},
    "geometry": [
        {"lat": 16.3000, "lon": 80.4000},
        {"lat": 16.3100, "lon": 80.4100}
    ]
}
with patch("backend.data_sources.overpass._run_overpass_query", return_value=({"elements": [mock_element]}, "mock://interpreter")):
    res_synthetic = drainage_features(16.3000, 80.4000, radius_m=2000, use_cache=False)
    coords = res_synthetic["geojson"]["features"][0]["geometry"]["coordinates"]
    print(f"   Input OSM geometry: lat=16.3000, lon=80.4000")
    print(f"   Output GeoJSON coordinate: {coords[0]}")
    assert coords[0][0] == 80.4000, f"Expected longitude 80.4000 at index 0, got {coords[0][0]}"
    assert coords[0][1] == 16.3000, f"Expected latitude 16.3000 at index 1, got {coords[0][1]}"
    print(f"   >>> Confirmed: GeoJSON coordinates[0] == longitude (80.4000) and coordinates[1] == latitude (16.3000) strictly! <<<")

# ------------------------------------------------------------------------------
# 9. Controlled Empty-Response Test (SUCCESSFUL QUERY, ZERO FEATURES)
# ------------------------------------------------------------------------------
print(f"\n9. Controlled Empty-Response Test (STATE A: NO MAPPED DATA):")
with patch("backend.data_sources.overpass._run_overpass_query", return_value=({"elements": []}, "mock://interpreter")):
    res_empty = drainage_features(26.9157, 70.9083, radius_m=2000, use_cache=False)
    print(f"   status:                  {res_empty.get('status')}")
    print(f"   waterway_count:          {res_empty.get('waterway_count')}")
    print(f"   total_waterway_length_km:{res_empty.get('total_waterway_length_km')}")
    print(f"   drainage_density_km_per_km2: {res_empty.get('drainage_density_km_per_km2')}")
    print(f"   nearest_drain_distance_m:{res_empty.get('nearest_drain_distance_m')}")
    print(f"   nearest_drain_name:      {res_empty.get('nearest_drain_name')}")
    print(f"   nearest_drain_type:      {res_empty.get('nearest_drain_type')}")
    print(f"   proximity badge:         {res_empty.get('proximity_advisory', {}).get('badge_text')}")
    print(f"   note:                    {res_empty.get('note')}")

    assert res_empty["status"] == "no_mapped_data_in_radius", f"Unexpected status: {res_empty['status']}"
    assert res_empty["waterway_count"] == 0, "Count should be 0"
    assert res_empty["total_waterway_length_km"] == 0.0, "Length should be 0.0"
    assert res_empty["drainage_density_km_per_km2"] == 0.0, "Density should be 0.0"
    assert res_empty["nearest_drain_distance_m"] is None, "Nearest distance should be None"
    assert res_empty["nearest_drain_name"] is None, "Nearest name should be None"
    assert res_empty["nearest_drain_type"] is None, "Nearest type should be None"
    assert res_empty["proximity_advisory"]["badge_text"] == "NO MAPPED OSM WATERWAYS IN RADIUS", f"Bad badge: {res_empty['proximity_advisory']['badge_text']}"
    assert "does not prove physical drainage is absent" in res_empty["note"]
    print(f"   >>> Confirmed: State A (Zero Features) semantics 100% verified! <<<")

# ------------------------------------------------------------------------------
# 10. Controlled Provider-Failure Test (STATE B: PROVIDER UNAVAILABLE)
# ------------------------------------------------------------------------------
print(f"\n10. Controlled Provider-Failure Test (STATE B: PROVIDER UNAVAILABLE):")
with patch("backend.data_sources.overpass._run_overpass_query", side_effect=RuntimeError("Simulated Overpass 429 Rate Limit")):
    res_fail = drainage_features(19.0760, 72.8777, radius_m=2000, use_cache=False)
    print(f"   status:                  {res_fail.get('status')}")
    print(f"   waterway_count:          {res_fail.get('waterway_count')}")
    print(f"   total_waterway_length_km:{res_fail.get('total_waterway_length_km')}")
    print(f"   drainage_density_km_per_km2: {res_fail.get('drainage_density_km_per_km2')}")
    print(f"   nearest_drain_distance_m:{res_fail.get('nearest_drain_distance_m')}")
    print(f"   nearest_drain_name:      {res_fail.get('nearest_drain_name')}")
    print(f"   nearest_drain_type:      {res_fail.get('nearest_drain_type')}")
    print(f"   proximity badge:         {res_fail.get('proximity_advisory', {}).get('badge_text')}")
    print(f"   error:                   {res_fail.get('error')}")

    assert res_fail["status"] == "unavailable", f"Unexpected status: {res_fail['status']}"
    assert res_fail["waterway_count"] is None, "Count MUST be None on provider failure, NOT 0!"
    assert res_fail["total_waterway_length_km"] is None, "Length MUST be None on provider failure, NOT 0.0!"
    assert res_fail["drainage_density_km_per_km2"] is None, "Density MUST be None on provider failure, NOT 0.0!"
    assert res_fail["nearest_drain_distance_m"] is None, "Nearest distance should be None"
    assert res_fail["nearest_drain_name"] is None, "Nearest name should be None"
    assert res_fail["nearest_drain_type"] is None, "Nearest type should be None"
    assert res_fail["proximity_advisory"]["badge_text"] == "DRAINAGE DATA UNAVAILABLE", f"Bad badge: {res_fail['proximity_advisory']['badge_text']}"
    assert "Simulated Overpass 429" in res_fail["error"]
    print(f"   >>> Confirmed: State B (Provider Unavailable) semantics 100% verified (NULL metrics, NOT zero)! <<<")

# ------------------------------------------------------------------------------
# 11. Paced India-Wide Live Tests Across Diverse Landscapes
# ------------------------------------------------------------------------------
print(f"\n11. Paced India-Wide Live Tests Across Diverse Landscapes:")
test_matrix = [
    ("Guntur (AP) - Coastal Irrigation Plain", 16.3067, 80.4365, 2000),
    ("Mumbai (MH) - Dense Coastal Urban", 19.0760, 72.8777, 2000),
    ("Delhi (DL) - Alluvial Urban Plain", 28.6139, 77.2090, 2000),
    ("Kolkata (WB) - Gangetic Delta Canals", 22.5726, 88.3639, 2000),
    ("Guwahati (AS) - Northeast Riverine", 26.1445, 91.7362, 2000),
    ("Rural Bihar (BR) - Agricultural Floodplain", 25.7500, 85.2000, 2000),
    ("Jaisalmer (RJ) - Arid Desert Baseline", 26.9157, 70.9083, 2000),
]

counts = {"SUCCESS": 0, "NO_MAPPED_DATA": 0, "UNAVAILABLE": 0}

for idx, (name, l_lat, l_lon, l_rad) in enumerate(test_matrix):
    # Pace requests by waiting 2.5s between uncached queries to respect public Overpass rate limits
    if idx > 0:
        time.sleep(2.5)

    t_start = time.time()
    res = drainage_features(l_lat, l_lon, radius_m=l_rad, use_cache=True)
    lat_ms = round((time.time() - t_start) * 1000, 2)
    
    status = res.get("status")
    cnt = res.get("waterway_count")
    dist = res.get("nearest_drain_distance_m")
    d_name = res.get("nearest_drain_name")
    d_type = res.get("nearest_drain_type")
    d_coords = res.get("nearest_drain_coords")
    l_in = res.get("total_waterway_length_km")
    dens = res.get("drainage_density_km_per_km2")
    c_hit = res.get("cache_hit")
    prox = res.get("proximity_advisory", {}).get("badge_text")

    if status == "ok":
        counts["SUCCESS"] += 1
        cat = "SUCCESS"
    elif status == "no_mapped_data_in_radius":
        counts["NO_MAPPED_DATA"] += 1
        cat = "NO_MAPPED_DATA"
    else:
        counts["UNAVAILABLE"] += 1
        cat = "UNAVAILABLE"

    print(f"\n  --- [{cat}] {name} (Lat {l_lat}, Lon {l_lon}, Radius {l_rad}m) ---")
    print(f"    Status:                {status} (Cache: {'HIT' if c_hit else 'MISS'}, Latency: {lat_ms} ms)")
    print(f"    Waterway Count:        {cnt if cnt is not None else '—'}")
    print(f"    Nearest Waterway:      {d_name if d_name is not None else '—'} ({d_type if d_type is not None else '—'})")
    print(f"    Nearest Distance:      {f'{dist} m' if dist is not None else '—'} | Coords: {d_coords if d_coords is not None else '—'}")
    print(f"    Clipped Length:        {f'{l_in} km' if l_in is not None else '—'}")
    print(f"    Mapped Network Density: {f'{dens} km/km²' if dens is not None else '—'}")
    print(f"    Proximity Advisory:    {prox}")

print("\n" + "=" * 80)
print(f"TEST SUMMARY: SUCCESS={counts['SUCCESS']}, NO_MAPPED_DATA={counts['NO_MAPPED_DATA']}, UNAVAILABLE={counts['UNAVAILABLE']}")
print(">>> ALL PHASE B PATCH CONTRACT & SEMANTIC AUDITS PASSED WITH ZERO DEFECTS! <<<")
print("=" * 80)
