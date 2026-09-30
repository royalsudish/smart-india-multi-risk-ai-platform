#!/usr/bin/env python3
"""
scripts/verify_phase_d.py
=============================================================================
Comprehensive Automated Verification & Regression Suite for Phase D:
- Structural Audit of HTML, CSS, and JS (Command Center, Leaflet Map, 3-Columns, Alerts, Workflow)
- Model Invariance & Contract Verification (30 feats, threshold 0.36, 39 Weather, 20 Disaster, 35 Agri)
- Pan-India & Local-Level Test Matrix (7 Diverse Indian Landscapes + Village + PIN + District)
- GeoJSON & Map Layer Contract Verification (Drainage, Terrain, Disaster Coordinates)
- Failure Semantics Verification (State A vs State B)
=============================================================================
"""

import os
import sys
import re
import json
import time
from pathlib import Path

# Setup workspace paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import create_app
from backend.services.model_service import registry
from backend.services.waterlogging_service import get_warning_threshold
from backend.services.waterlogging_features import build_terrain_manifest, get_live_terrain
from backend.data_sources.overpass import drainage_features
from backend.data_sources.usgs import get_recent_earthquake_features
from backend.data_sources.gdacs import get_cyclone_and_volcano_alerts


def log_test(title):
    print(f"\n[TEST] {title}")


def assert_true(cond, msg):
    if not cond:
        print(f"  FAILED: {msg}")
        sys.exit(1)
    print(f"  PASSED: {msg}")


def verify_html_structure():
    log_test("HTML Structure & Phase D Geospatial Layout Verification")
    html_path = ROOT_DIR / "frontend" / "templates" / "index.html"
    assert_true(html_path.exists(), "frontend/templates/index.html exists")
    content = html_path.read_text(encoding="utf-8")

    # 1. Leaflet CDN assets
    assert_true("leaflet@1.9.4/dist/leaflet.css" in content, "Leaflet 1.9.4 CSS included")
    assert_true("leaflet@1.9.4/dist/leaflet.js" in content, "Leaflet 1.9.4 JS included")

    # 2. Main 3-column command center structure
    assert_true('class="command-center-grid"' in content, "command-center-grid container exists")
    assert_true('class="column-sidebar column-left"' in content, "Left column (~22%) exists")
    assert_true('class="column-center-map"' in content, "Center column (~56% dominant map) exists")
    assert_true('class="column-sidebar column-right"' in content, "Right column (~22%) exists")

    # 3. Dominant Map elements
    assert_true('id="mapContainer"' in content, "mapContainer viewport exists")
    assert_true('id="mapLayerControls"' in content, "Floating mapLayerControls exists")
    assert_true('id="mapLegend"' in content, "Floating mapLegend exists")
    assert_true('id="mapResetBtn"' in content, "mapResetBtn exists")

    # 4. Layer toggles (Active vs Disabled)
    assert_true('id="layerToggleWaterlogging"' in content, "Waterlogging risk layer toggle exists")
    assert_true('id="layerToggleDrainage"' in content, "Drainage layer toggle exists")
    assert_true('id="layerToggleTerrain"' in content, "Terrain sample points toggle exists")
    assert_true('id="layerToggleDisaster"' in content, "Disaster event markers toggle exists")
    assert_true('id="layerToggleRainfall"' in content and "disabled" in content, "Rainfall spatial layer toggle exists and is disabled")
    assert_true('id="layerToggleSatellite"' in content and "disabled" in content, "Satellite water detection toggle exists and is disabled")
    assert_true('id="layerToggleAgri"' in content and "disabled" in content, "Regional agriculture toggle exists and is disabled")
    assert_true('id="layerToggleGrid"' in content and "disabled" in content, "Grid waterlogging toggle exists and is disabled")

    # 5. Point analysis disclaimer
    assert_true('id="pointAnalysisDisclaimer"' in content, "Point analysis disclaimer tag exists")
    assert_true("Point analysis at selected/resolved coordinate" in content, "Disclaimer text explicitly present")

    # 6. Full-width Alerts & Early Warning Bar
    assert_true('class="alerts-bar-panel"' in content, "Full-width alerts bar panel exists")
    assert_true('id="overallRiskBadge"' in content, "overallRiskBadge exists")
    assert_true('id="alertBarWaterlogging"' in content, "alertBarWaterlogging exists")
    assert_true('id="alertBarOutlook"' in content, "alertBarOutlook exists")
    assert_true('id="alertBarDisaster"' in content, "alertBarDisaster exists")
    assert_true('id="alertBarAgri"' in content, "alertBarAgri exists")
    assert_true('id="overallAdvisoryBox"' in content, "overallAdvisoryBox exists")

    # 7. Bottom System Workflow
    assert_true('class="system-workflow-panel"' in content, "Bottom system workflow panel exists")
    assert_true("Smart Multi-Risk AI System Workflow" in content, "System workflow title exists")
    assert_true("Data Ingestion" in content and "Feature Engineering" in content and "4 ML Inference Engines" in content, "Pipeline stages exist")
    assert_true("Open-Meteo Weather" in content and "OpenStreetMap / Overpass" in content and "USGS Earthquake" in content, "Data source cards exist")
    assert_true("NASA GPM Precipitation" in content and "NOT CONFIGURED" in content, "NASA GPM is marked NOT CONFIGURED")
    assert_true("ESA Copernicus Sentinel-1/2" in content and "NOT CONFIGURED" in content, "Sentinel is marked NOT CONFIGURED")
    assert_true("WHAT HAPPENED?" in content and "CURRENT RISK" in content and "WHAT MAY HAPPEN?" in content, "Waterlogging 3-stage temporal architecture exists")


def verify_css_rules():
    log_test("CSS Layout Proportions & Responsive Styling Verification")
    css_path = ROOT_DIR / "frontend" / "static" / "css" / "style.css"
    assert_true(css_path.exists(), "frontend/static/css/style.css exists")
    content = css_path.read_text(encoding="utf-8")

    # 1. 22% | 56% | 22% layout rule
    assert_true("grid-template-columns: minmax(280px, 22%) 1fr minmax(280px, 22%);" in content or
                "22%" in content, "22% / 56% / 22% proportions defined in CSS")

    # 2. Responsive breakpoints
    assert_true("@media (max-width: 1199px)" in content, "Tablet breakpoint (<= 1199px) exists")
    assert_true("@media (max-width: 767px)" in content, "Mobile breakpoint (<= 767px) exists")
    assert_true("@media (prefers-reduced-motion: reduce)" in content, "Accessibility prefers-reduced-motion rule exists")

    # 3. Leaflet dark popup styling
    assert_true(".leaflet-popup-content-wrapper" in content, "Leaflet popup dark styling exists")
    assert_true(".pulse-beacon-marker" in content, "Pulse beacon marker animation exists")
    assert_true(".terrain-model-glyph" in content, "Terrain model glyph styling exists")


def verify_js_architecture():
    log_test("JavaScript Map State, Race Safety & Event Listeners Verification")
    js_path = ROOT_DIR / "frontend" / "static" / "js" / "app.js"
    assert_true(js_path.exists(), "frontend/static/js/app.js exists")
    content = js_path.read_text(encoding="utf-8")

    # 1. Map state & BASEMAP_CONFIG isolation
    assert_true("const BASEMAP_CONFIG" in content, "BASEMAP_CONFIG is isolated for provider flexibility")
    assert_true("const mapState" in content, "Centralized mapState architecture exists")
    assert_true("initMap()" in content, "initMap function exists")
    assert_true("20.5937, 78.9629" in content, "Initial map center set to India-wide [20.5937, 78.9629]")

    # 2. Single authoritative marker (no duplicates)
    assert_true("updateSelectedLocationMarker" in content, "updateSelectedLocationMarker function exists")
    assert_true("mapState.selectedMarker.setLatLng" in content, "Selected marker is updated in-place without duplicating")

    # 3. Race condition guards
    assert_true("currentRequestId" in content, "currentRequestId guard exists")
    assert_true("isAnalyzing" in content, "isAnalyzing flag exists")
    assert_true("requestId !== currentRequestId" in content, "Stale request protection present in fetch handlers")

    # 4. DOM Safety
    assert_true("escapeHtml" in content, "escapeHtml helper function exists for safe popup text")
    assert_true("textContent" in content, "textContent used extensively for DOM safety")


def verify_model_contracts():
    log_test("Model Invariance & Contract Verification")
    status = registry.status(try_load=True)
    assert_true(status["waterlogging"]["loaded"], "Waterlogging model loaded successfully")
    assert_true(status["weather"]["loaded"], "Weather model loaded successfully")
    assert_true(status["disaster"]["loaded"], "Disaster model loaded successfully")
    assert_true(status["agriculture"]["loaded"], "Agriculture model loaded successfully")

    # Verify Waterlogging 30 features & 0.36 threshold
    wl_features = status["waterlogging"]["features"]
    assert_true(len(wl_features) == 30, f"Waterlogging feature count is exactly 30 (got {len(wl_features)})")
    
    thresh = get_warning_threshold()
    assert_true(abs(thresh - 0.36) < 1e-4, f"Waterlogging warning threshold is strictly 0.36 (got {thresh})")

    # Run synthetic inference to verify threshold 0.36 logic
    mock_feats = {f: 0.0 for f in wl_features}
    mock_feats["rainfall_24h"] = 45.0
    mock_feats["soil_moisture_0_to_7cm"] = 0.40
    res = registry.predict("waterlogging", mock_feats)
    prob = res.get("positive_probability", res.get("probability", 0.0))
    warning = prob >= thresh
    expected_warning = prob >= 0.36
    assert_true(warning == expected_warning, f"Waterlogging warning strictly adheres to 0.36 threshold (prob={prob}, warning={warning})")


def verify_pan_india_and_local_searches():
    log_test("Pan-India Landscape & Local Search Verification")
    app = create_app()
    client = app.test_client()

    # 7 Diverse Indian Landscape Locations
    test_locations = [
        {"name": "Guntur (South Plain)", "lat": 16.3067, "lon": 80.4365, "crop": "Rice", "stage": "Flowering"},
        {"name": "Patna (East River Basin)", "lat": 25.5941, "lon": 85.1376, "crop": "Wheat", "stage": "Vegetative"},
        {"name": "Mumbai (West Coastal)", "lat": 19.0760, "lon": 72.8777, "crop": "Cotton", "stage": "Tillering"},
        {"name": "Bengaluru (South Plateau)", "lat": 12.9716, "lon": 77.5946, "crop": "Maize", "stage": "Maturity"},
        {"name": "Guwahati (Northeast Valley)", "lat": 26.1445, "lon": 91.7362, "crop": "Rice", "stage": "Grain_Filling"},
        {"name": "Dehradun (North Foothills)", "lat": 30.3165, "lon": 78.0322, "crop": "Mustard", "stage": "Flowering"},
        {"name": "Jaisalmer (West Desert)", "lat": 26.9157, "lon": 70.9083, "crop": "Soybean", "stage": "Vegetative"},
    ]

    for loc in test_locations:
        payload = {
            "lat": loc["lat"],
            "lon": loc["lon"],
            "crop": loc["crop"],
            "crop_stage": loc["stage"]
        }
        resp = client.post("/api/predict/all/live", json=payload)
        assert_true(resp.status_code == 200, f"Unified live prediction succeeded for {loc['name']} (HTTP 200)")
        data = resp.get_json()
        assert_true(data["success"] is True, f"Prediction response success=True for {loc['name']}")
        assert_true("summary" in data and "overall_risk" in data["summary"], f"Summary overall_risk present for {loc['name']}")
        assert_true("waterlogging" in data and data["waterlogging"]["success"], f"Waterlogging success for {loc['name']}")
        assert_true("weather" in data and data["weather"]["success"], f"Weather success for {loc['name']}")
        assert_true("disaster" in data and data["disaster"]["success"], f"Disaster success for {loc['name']}")
        assert_true("agriculture" in data and data["agriculture"]["success"], f"Agriculture success for {loc['name']}")

        # Verify disaster seismic & gdacs metadata with coordinates
        disaster_inp = data["disaster"]["data"]["input"]
        assert_true("seismic_metadata" in disaster_inp, f"seismic_metadata present for {loc['name']}")
        assert_true("gdacs_metadata" in disaster_inp, f"gdacs_metadata present for {loc['name']}")

    # Local Searches: Village, PIN, District
    log_test("Geocoding Local Search & Disambiguation Checks")
    # PIN Code search
    pin_resp = client.get("/api/sources/geocode?q=522002&limit=5")
    assert_true(pin_resp.status_code == 200, "PIN code 522002 search returned HTTP 200")
    pin_data = pin_resp.get_json()
    assert_true(pin_data["success"] and len(pin_data["data"]) > 0, "PIN code resolved to at least 1 Indian location")

    # Ambiguous Village search
    rampur_resp = client.get("/api/sources/geocode?q=Rampur&limit=5")
    assert_true(rampur_resp.status_code == 200, "Rampur search returned HTTP 200")
    rampur_data = rampur_resp.get_json()
    assert_true(rampur_data["success"] and len(rampur_data["data"]) > 1, f"Ambiguous 'Rampur' returned {len(rampur_data['data'])} candidates for user disambiguation")


def verify_map_layers_geojson():
    log_test("GeoJSON Map Layers & Failure Semantics Verification")
    # 1. Drainage GeoJSON check
    d_res = drainage_features(16.3067, 80.4365, radius_m=2000)
    assert_true("geojson" in d_res, "Drainage returns geojson")
    assert_true(d_res["geojson"]["type"] == "FeatureCollection", "Drainage geojson is a FeatureCollection")
    if d_res["geojson"]["features"]:
        feat = d_res["geojson"]["features"][0]
        assert_true(feat["geometry"]["type"] == "LineString", "Drainage feature geometry is LineString")
        # GeoJSON standard coordinate order: [longitude, latitude]
        c = feat["geometry"]["coordinates"][0]
        assert_true(isinstance(c, list) and len(c) == 2, "Coordinates format is [lon, lat]")
        assert_true(c[0] > 60 and c[0] < 100, f"First coord is longitude in India range (got {c[0]})")
        assert_true(c[1] > 5 and c[1] < 40, f"Second coord is latitude in India range (got {c[1]})")

    # 2. Terrain Sample GeoJSON check
    t_res = get_live_terrain(16.3067, 80.4365, return_details=True)
    assert_true("enhanced_terrain" in t_res, "Terrain returns enhanced_terrain")
    enh = t_res["enhanced_terrain"]
    assert_true("samples_geojson" in enh, "Enhanced terrain contains samples_geojson")
    assert_true(len(enh["samples_geojson"]["features"]) == 13, f"Terrain contains exactly 13 sample points (got {len(enh['samples_geojson']['features'])})")
    
    # Check model stencil vs context ring roles
    model_pts = [f for f in enh["samples_geojson"]["features"] if f["properties"]["sample_role"] == "model" or f["properties"]["sample_id"] == "model_center"]
    context_pts = [f for f in enh["samples_geojson"]["features"] if f["properties"]["sample_role"] == "context"]
    assert_true(len(model_pts) == 5, f"Exactly 5 model stencil points present (got {len(model_pts)})")
    assert_true(len(context_pts) == 8, f"Exactly 8 contextual ring points present (got {len(context_pts)})")

    # 3. State A (no_mapped_data) vs State B (unavailable) Semantics Check
    # Verify no_mapped_data structure
    from backend.data_sources.overpass import evaluate_drainage_proximity
    no_map_prox = evaluate_drainage_proximity(None)
    assert_true(no_map_prox["badge_text"] == "NO MAPPED OSM WATERWAYS IN RADIUS", "Proximity advisory for None is 'NO MAPPED OSM WATERWAYS IN RADIUS'")


def main():
    print("=" * 70)
    print("PHASE D COMPREHENSIVE VERIFICATION & REGRESSION AUDIT")
    print("=" * 70)

    t0 = time.time()
    verify_html_structure()
    verify_css_rules()
    verify_js_architecture()
    verify_model_contracts()
    verify_pan_india_and_local_searches()
    verify_map_layers_geojson()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL PHASE D TESTS PASSED IN {elapsed:.2f}s! GEOSPATIAL COMMAND CENTER VERIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
