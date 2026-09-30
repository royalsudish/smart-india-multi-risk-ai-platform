"""
Pan-India Hierarchical Spatial Risk Mapping Test Suite.

Validates:
1. Pan-India State/UT catalog (28 States + 8 UTs = 36 total).
2. Geographic lookup and coordinate resolution across India.
3. GeoJSON APIs:
   - GET /api/risk/states
   - GET /api/risk/india (state-level synthesized risk)
   - GET /api/risk/state (district-level summarized risk)
   - GET /api/risk/district (district resolution and bounds)
   - POST /api/risk/grid (vectorized batch ML inference on Cartesian grid)
4. Model contract preservation:
   - Waterlogging Random Forest (30 features, exact scaler, [0, 1] probability)
   - Drainage Random Forest (6 features, exact order, [0, 1] probability)
5. RFC 7946 GeoJSON polygon compliance (5-vertex closed ring, [lon, lat]).
6. Multi-criteria hotspot detection logic.
7. Data quality transparency (FULL_DATA, PARTIAL_DATA, FALLBACK_USED).
8. Pan-India 11-region validation:
   - Guntur, Patna, Delhi, Mumbai, Chennai, Kochi, Jaipur, Guwahati, Hyderabad, Dehradun, Srinagar.
9. Non-regression of existing core endpoints.
"""

import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from app import create_app
from backend.services.pan_india_catalog import (
    INDIA_STATES,
    get_all_states,
    find_state,
    get_districts_for_state,
    find_district,
    resolve_location,
)
from backend.services.spatial_risk_service import (
    generate_spatial_grid_cells,
    sample_batch_terrain,
    classify_combined_hotspot,
    compute_spatial_grid_risk,
    get_india_risk_summary,
    get_state_risk_summary,
    get_district_context,
)
from backend.services.model_service import registry


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# =============================================================================
# 1. CATALOG INTEGRITY & HIERARCHY
# =============================================================================

def test_pan_india_catalog_completeness():
    """Verify that all 28 States and 8 Union Territories are covered (36 total)."""
    assert len(INDIA_STATES) == 36, f"Expected 36 States/UTs, found {len(INDIA_STATES)}"

    states = get_all_states()
    assert len(states) == 36, f"Expected 36 States/UTs, found {len(states)}"

    # Verify key metadata exists on all entities
    for s in states:
        assert "state" in s
        assert "code" in s
        assert "capital" in s
        assert "lat" in s and "lon" in s
        assert "bbox" in s
        assert len(s["bbox"]) == 4
        # bbox is [min_lon, min_lat, max_lon, max_lat]
        assert s["bbox"][1] < s["bbox"][3]  # min_lat < max_lat
        assert s["bbox"][0] < s["bbox"][2]  # min_lon < max_lon
        assert s["district_count"] > 0


def test_catalog_lookup_and_aliases():
    """Verify state and district lookups handle case, whitespace, and codes."""
    # State lookups
    ap = find_state("andhra pradesh")
    assert ap is not None and ap["code"] == "AP"

    dl = find_state("Delhi")
    assert dl is not None and dl["code"] == "DL"

    jk = find_state("Jammu & Kashmir")
    assert jk is not None and "Jammu" in jk["state"]

    # District lookups across India
    d_patna = find_district("Bihar", "Patna")
    assert d_patna is not None and abs(d_patna["lat"] - 25.5941) < 0.1

    d_guntur = find_district("Andhra Pradesh", "Guntur")
    assert d_guntur is not None and abs(d_guntur["lat"] - 16.3067) < 0.1

    d_srinagar = find_district("Jammu and Kashmir", "Srinagar")
    assert d_srinagar is not None and abs(d_srinagar["lat"] - 34.0837) < 0.1


def test_resolve_location_coordinates():
    """Verify reverse coordinate resolution into state and district."""
    # Patna coordinates -> Bihar, Patna
    res_patna = resolve_location(25.5941, 85.1376)
    assert res_patna["state"] == "Bihar"
    assert res_patna["district"] == "Patna"

    # Guntur coordinates -> Andhra Pradesh, Guntur
    res_guntur = resolve_location(16.3067, 80.4365)
    assert res_guntur["state"] == "Andhra Pradesh"
    assert res_guntur["district"] == "Guntur"


# =============================================================================
# 2. CARTESIAN GRID GENERATION & RFC 7946 GEOJSON
# =============================================================================

def test_generate_spatial_grid_cells_geometry():
    """Verify Cartesian grid generation, circular disk clipping, and RFC 7946 GeoJSON."""
    lat, lon = 25.5941, 85.1376  # Patna
    radius_m = 5000
    grid_size_m = 500

    grid_res = generate_spatial_grid_cells(lat, lon, radius_m=radius_m, grid_size_m=grid_size_m)
    assert "cells" in grid_res
    cells = grid_res["cells"]
    assert len(cells) > 0

    # Area of disk = pi * 5^2 = 78.5 km2; cell area = 0.5 * 0.5 = 0.25 km2 => ~314 cells
    assert 250 <= len(cells) <= 380

    for c in cells:
        assert "grid_id" in c
        assert "center_lat" in c
        assert "center_lon" in c
        assert "polygon_ring" in c
        ring = c["polygon_ring"]

        # RFC 7946: 5 vertices for a closed bounding box polygon
        assert len(ring) == 5
        # Start and end vertex must match exactly to form a closed ring
        assert ring[0] == ring[4]
        # Coordinates must be [lon, lat] order
        for pt in ring:
            assert len(pt) == 2
            p_lon, p_lat = pt[0], pt[1]
            assert 60.0 <= p_lon <= 100.0  # Valid India longitudes
            assert 6.0 <= p_lat <= 38.0    # Valid India latitudes

        # Cell center must be within radius + margin
        dist_m = math.sqrt(
            ((c["center_lat"] - lat) * 111000) ** 2 +
            ((c["center_lon"] - lon) * (111000 * math.cos(math.radians(lat)))) ** 2
        )
        assert dist_m <= (radius_m + grid_size_m * 1.5)


# =============================================================================
# 3. VECTORIZED BATCH MODEL INFERENCE CONTRACTS
# =============================================================================

def test_waterlogging_batch_inference_contract():
    """Verify Waterlogging model batch inference preserves 30 features and produces valid probabilities."""
    wl_slot = registry.get("waterlogging")
    assert wl_slot is not None
    wl_slot.ensure_loaded()
    assert wl_slot.is_loaded

    expected_features = wl_slot.metadata.get("features", [])
    assert len(expected_features) == 30

    # Generate synthetic batch dataframe of 10 rows
    dummy_data = {feat: [0.0] * 10 for feat in expected_features}
    # Set realistic varied values
    dummy_data["elevation"] = [10.0, 15.0, 20.0, 25.0, 30.0, 5.0, 8.0, 12.0, 50.0, 100.0]
    dummy_data["slope"] = [0.5, 1.2, 0.8, 2.5, 4.0, 0.2, 0.4, 1.0, 5.0, 12.0]
    dummy_data["rainfall_24h"] = [50.0, 30.0, 10.0, 5.0, 0.0, 80.0, 60.0, 20.0, 0.0, 0.0]

    df = pd.DataFrame(dummy_data)
    batch_out = wl_slot.predict_batch(df)

    assert batch_out["row_count"] == 10
    assert len(batch_out["predictions"]) == 10
    assert len(batch_out["positive_probabilities"]) == 10
    for p in batch_out["positive_probabilities"]:
        assert 0.0 <= p <= 1.0


def test_drainage_batch_inference_contract():
    """Verify Drainage Random Forest batch inference preserves exact 6 features in order."""
    dr_slot = registry.get("drainage")
    assert dr_slot is not None
    dr_slot.ensure_loaded()
    assert dr_slot.is_loaded

    expected_features = dr_slot.metadata.get("features", [])
    assert expected_features == ["elevation", "slope", "rain_max", "sentinel1_vv", "NDVI", "NDBI"]

    dummy_data = {
        "elevation": [12.0, 18.0, 25.0, 8.0, 4.0],
        "slope": [0.6, 1.5, 3.2, 0.4, 0.1],
        "rain_max": [math.log10(51.0), math.log10(21.0), 0.0, math.log10(81.0), math.log10(121.0)],
        "sentinel1_vv": [-11.2, -10.5, -9.8, -13.5, -15.0],
        "NDVI": [0.35, 0.40, 0.28, 0.22, 0.15],
        "NDBI": [-0.02, -0.05, 0.05, 0.12, 0.18],
    }

    df = pd.DataFrame(dummy_data)
    batch_out = dr_slot.predict_batch(df)

    assert batch_out["row_count"] == 5
    assert len(batch_out["predictions"]) == 5
    assert len(batch_out["positive_probabilities"]) == 5
    for p in batch_out["positive_probabilities"]:
        assert 0.0 <= p <= 1.0


# =============================================================================
# 4. HOTSPOT LOGIC & PROVENANCE
# =============================================================================

def test_hotspot_classification_logic():
    """Verify deterministic multi-criteria hotspot classification."""
    # Critical hotspot: high waterlogging AND high drainage
    crit = classify_combined_hotspot(wl_prob=0.55, dr_prob=0.60, slope=0.8, rain_max=35.0, elevation=15.0)
    assert crit == "CRITICAL_HOTSPOT"

    # Extreme waterlogging alone creates critical hotspot
    crit2 = classify_combined_hotspot(wl_prob=0.72, dr_prob=0.20, slope=1.0, rain_max=20.0, elevation=15.0)
    assert crit2 == "CRITICAL_HOTSPOT"

    # High risk hotspot: warning waterlogging with flat slope or rain
    high = classify_combined_hotspot(wl_prob=0.40, dr_prob=0.30, slope=0.5, rain_max=10.0, elevation=15.0)
    assert high == "HIGH_RISK_HOTSPOT"

    # Potential hotspot: moderate probabilities
    pot = classify_combined_hotspot(wl_prob=0.25, dr_prob=0.40, slope=3.0, rain_max=5.0, elevation=100.0)
    assert pot == "POTENTIAL_HOTSPOT"

    # Normal: low probabilities
    norm = classify_combined_hotspot(wl_prob=0.08, dr_prob=0.12, slope=5.0, rain_max=0.0, elevation=250.0)
    assert norm == "NORMAL"


# =============================================================================
# 5. API ENDPOINTS INTEGRATION
# =============================================================================

def test_api_risk_states(client):
    """Verify GET /api/risk/states returns all 36 entities."""
    res = client.get("/api/risk/states")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert len(json_data["data"]) == 36


def test_api_risk_india(client):
    """Verify GET /api/risk/india returns state summaries as GeoJSON."""
    res = client.get("/api/risk/india")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["total_states"] == 36
    assert "summary" in data
    assert "geojson" in data
    fc = data["geojson"]
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 36

    # Verify properties of features
    first_feat = fc["features"][0]
    p = first_feat["properties"]
    assert "state" in p
    assert p["overall_risk"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "disaster_context" in p


def test_api_risk_state(client):
    """Verify GET /api/risk/state?state=Bihar returns district breakdown."""
    res = client.get("/api/risk/state?state=Bihar")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["state"] == "Bihar"
    assert "bbox" in data
    assert len(data["districts"]) > 0

    patna_d = next((d for d in data["districts"] if d["district"] == "Patna"), None)
    assert patna_d is not None
    assert patna_d["overall_risk"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def test_api_risk_district(client):
    """Verify GET /api/risk/district?state=Bihar&district=Patna."""
    res = client.get("/api/risk/district?state=Bihar&district=Patna")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["district"] == "Patna"
    assert data["state"] == "Bihar"
    assert abs(data["lat"] - 25.5941) < 0.1
    assert abs(data["lon"] - 85.1376) < 0.1
    assert data["recommended_grid"]["grid_size_m"] == 500


def test_api_risk_grid_post(client):
    """Verify POST /api/risk/grid produces complete GeoJSON with batch ML inference."""
    payload = {
        "lat": 25.5941,
        "lon": 85.1376,
        "radius_m": 2000,   # 2km radius (~50 cells) for fast unit test
        "grid_size_m": 500,
        "state": "Bihar",
        "district": "Patna"
    }
    res = client.post("/api/risk/grid", json=payload)
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]

    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) > 0
    assert "summary" in data

    # Verify summary structure
    s = data["summary"]
    assert s["total_cells"] == len(data["geojson"]["features"])
    assert "waterlogging" in s
    assert "drainage" in s
    assert "hotspots_count" in s
    assert "top_hotspots" in s

    # Verify individual cell properties
    for feat in data["geojson"]["features"]:
        p = feat["properties"]
        assert "grid_id" in p
        assert "center_lat" in p
        assert "center_lon" in p
        assert 0.0 <= p["waterlogging_probability"] <= 1.0
        assert p["waterlogging_risk"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert 0.0 <= p["drainage_probability"] <= 1.0
        assert p["drainage_risk"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert p["combined_hotspot_status"] in ["CRITICAL_HOTSPOT", "HIGH_RISK_HOTSPOT", "POTENTIAL_HOTSPOT", "NORMAL"]
        assert p["data_quality"] in ["FULL_DATA", "PARTIAL_DATA", "FALLBACK_USED"]


# =============================================================================
# 6. PAN-INDIA MULTI-REGION VALIDATION (11 MANDATORY LOCATIONS)
# =============================================================================

TEST_REGIONS = [
    ("Guntur", "Andhra Pradesh", 16.3067, 80.4365),
    ("Patna", "Bihar", 25.5941, 85.1376),
    ("Delhi", "Delhi", 28.6139, 77.2090),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Kochi", "Kerala", 9.9312, 76.2673),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Guwahati", "Assam", 26.1445, 91.7362),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
]


@pytest.mark.parametrize("city,state,lat,lon", TEST_REGIONS)
def test_pan_india_multi_region_grid_execution(city, state, lat, lon):
    """Verify compute_spatial_grid_risk executes smoothly across diverse Indian regions."""
    # Use 1000m radius (~12 cells) for fast multi-region testing
    result = compute_spatial_grid_risk(
        center_lat=lat,
        center_lon=lon,
        radius_m=1000,
        grid_size_m=500,
        state=state,
        district=city
    )

    assert "geojson" in result
    assert result["geojson"]["type"] == "FeatureCollection"
    features = result["geojson"]["features"]
    assert len(features) > 0
    assert result["summary"]["total_cells"] == len(features)

    # Ensure no fabricated or empty risk predictions
    for feat in features:
        p = feat["properties"]
        assert p["waterlogging_probability"] is not None
        assert p["drainage_probability"] is not None
        assert p["elevation"] is not None
        assert p["slope"] is not None
        assert p["data_quality"] in ["FULL_DATA", "PARTIAL_DATA", "FALLBACK_USED"]


# =============================================================================
# 7. REGRESSION SAFETY ON EXISTING CORE PIPELINE
# =============================================================================

def test_existing_point_predict_live_unaffected(client):
    """Ensure POST /api/predict/all/live continues to work perfectly."""
    payload = {
        "lat": 16.3067,
        "lon": 80.4365,
        "crop": "Cotton",
        "crop_stage": "Vegetative"
    }
    res = client.post("/api/predict/all/live", json=payload)
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert "weather" in json_data
    assert "disaster" in json_data
    assert "agriculture" in json_data
    assert "waterlogging" in json_data
    assert "summary" in json_data
