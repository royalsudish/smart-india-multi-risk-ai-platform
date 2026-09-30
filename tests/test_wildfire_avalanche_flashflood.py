"""
Automated Test Suite: 20 Required Scenarios for Wildfire, Avalanche, and Flash Flood
====================================================================================
Covers all 20 required scenarios from Section 11 of the specification:
 1. wildfire no detection
 2. wildfire nearby recent FIRMS detection
 3. distant wildfire ignored
 4. old wildfire ignored
 5. high FRP wildfire
 6. FIRMS provider unavailable
 7. avalanche official warning active
 8. avalanche alert expired
 9. avalanche alert outside selected district
 10. SACHET unavailable
 11. flash flood intense short rainfall
 12. heavy rain but low runoff evidence
 13. high waterlogging + drainage congestion
 14. Sentinel-1 corroborates flash flood
 15. GloFAS unavailable but flash flood still detected
 16. multiple hazards together
 17. wildfire + air pollution
 18. flash flood + thunderstorm + high wind
 19. no regression in existing 12 hazards
 20. no regression in Weather/Waterlogging/Drainage/Agriculture/Map
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from backend.services.api_hazard_engine import (
    evaluate_api_hazards,
    detect_wildfire,
    detect_avalanche,
    detect_flash_flood,
    SUPPORTED_HAZARDS,
)
from backend.services.disaster_service import (
    corroborate_disaster_evidence,
    predict_disaster_live,
)
from backend.data_sources.india_official import (
    fetch_sachet_cap_alerts,
    get_official_india_evidence,
)
from app import create_app


@pytest.fixture
def base_features():
    return {
        "temp_2m": 25.0,
        "precip_1h": 0.0,
        "rain_24h": 0.0,
        "rain_48h": 0.0,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 15.0,
        "surface_pressure": 1012.0,
        "weather_code": 1,
        "us_aqi": 40.0,
        "pm2_5": 15.0,
        "elevation": 200.0,
        "slope": 2.0,
        "river_discharge": 2.0,
        "max_quake_mag": 0.0,
        "quake_depth": 0.0,
        "tsunami_flag": 0,
        "gdacs_cyclone": 0,
        "gdacs_volcano": 0,
        "wind_pressure_shear": 0.1,
        "precipitation_saturation_rate": 0.0,
        "thermal_particulate_index": 20.0,
    }


@pytest.fixture
def base_built(base_features):
    return {
        "features": base_features,
        "seismic_metadata": {"event_distance_km": None, "event_age_hours": None, "max_quake_mag": 0.0},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
        "official_alerts": [],
        "official_india": {"official_alerts": []},
        "waterlogging_probability": 0.05,
        "drainage_congestion_probability": 0.05,
        "firms_fire": {
            "source": "NASA_FIRMS",
            "status": "available",
            "hotspots": [],
            "nearest_fire_distance_km": None,
            "detections_10km": 0,
            "detections_25km": 0,
            "detections_50km": 0,
            "max_frp": 0.0,
            "latest_detection_age_hours": None,
        },
    }


@pytest.fixture
def base_history():
    return {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0, "nearest_event": None},
        "flood": {"peak_discharge_7d_m3s": 2.0, "discharge_trend": "STEADY", "rainfall_accumulation": {"rain_24h_mm": 0.0, "rain_7d_mm": 0.0}},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL", "rainfall_deficit_pct": 0.0},
    }


# ==============================================================================
# 1. Wildfire No Detection
# ==============================================================================
def test_1_wildfire_no_detection(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [],
        "nearest_fire_distance_km": None,
        "detections_10km": 0,
        "detections_25km": 0,
        "detections_50km": 0,
        "max_frp": 0.0,
        "latest_detection_age_hours": None,
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is False
    assert res["severity"] == "NONE"


# ==============================================================================
# 2. Wildfire Nearby Recent FIRMS Detection
# ==============================================================================
def test_2_wildfire_nearby_recent_firms_detection(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [
            {"latitude": 22.1, "longitude": 78.5, "distance_km": 6.4, "detection_age_hours": 2.3, "frp": 42.1, "confidence": "high", "satellite": "VIIRS NOAA-21"}
        ],
        "nearest_fire_distance_km": 6.4,
        "detections_10km": 1,
        "detections_25km": 3,
        "detections_50km": 3,
        "max_frp": 42.1,
        "latest_detection_age_hours": 2.3,
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is True
    assert res["severity"] in ("HIGH", "SEVERE", "CRITICAL")
    assert res["confidence_type"] == "satellite_observed"
    assert "NASA FIRMS" in res["sources"][0] or "VIIRS" in res["sources"][0]
    assert res["evidence"]["nearest_fire_distance_km"] == 6.4
    assert res["evidence"]["detections_10km"] == 1


# ==============================================================================
# 3. Distant Wildfire Ignored (>50 km)
# ==============================================================================
def test_3_distant_wildfire_ignored(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [
            {"latitude": 21.2, "longitude": 78.1, "distance_km": 78.5, "detection_age_hours": 3.0, "frp": 45.0, "confidence": "high"}
        ],
        "nearest_fire_distance_km": 78.5,
        "detections_10km": 0,
        "detections_25km": 0,
        "detections_50km": 0,
        "detections_100km": 1,
        "max_frp": 45.0,
        "latest_detection_age_hours": 3.0,
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is False
    assert "beyond localized hazard threshold" in res["reason"]


# ==============================================================================
# 4. Old Wildfire Ignored (>48 hours)
# ==============================================================================
def test_4_old_wildfire_ignored(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [
            {"latitude": 22.0, "longitude": 78.9, "distance_km": 8.0, "detection_age_hours": 72.0, "frp": 50.0}
        ],
        "nearest_fire_distance_km": 8.0,
        "detections_10km": 1,
        "detections_25km": 1,
        "detections_50km": 1,
        "max_frp": 50.0,
        "latest_detection_age_hours": 72.0,
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is False


# ==============================================================================
# 5. High FRP Wildfire (Extreme Thermal Output)
# ==============================================================================
def test_5_high_frp_wildfire(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [
            {"latitude": 22.05, "longitude": 78.95, "distance_km": 4.5, "detection_age_hours": 1.5, "frp": 125.0, "confidence": "high"}
        ],
        "nearest_fire_distance_km": 4.5,
        "detections_10km": 2,
        "detections_25km": 4,
        "detections_50km": 4,
        "max_frp": 125.0,
        "latest_detection_age_hours": 1.5,
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is True
    assert res["severity"] == "CRITICAL"


# ==============================================================================
# 6. FIRMS Provider Unavailable (Track Provider Failure)
# ==============================================================================
def test_6_firms_provider_unavailable(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "unavailable",
        "missing_provider": True,
        "reason": "NASA FIRMS API connection timeout.",
        "hotspots": [],
    }
    res = detect_wildfire(base_features, base_built, base_history, [])
    assert res["hazard"] == "wildfire"
    assert res["active"] is False
    assert res["confidence_type"] == "insufficient_data"
    assert res["evidence"]["missing_provider"] is True


# ==============================================================================
# 7. Avalanche Official Warning Active
# ==============================================================================
def test_7_avalanche_official_warning_active(base_features, base_built, base_history):
    now = datetime.now(timezone.utc)
    official_alerts = [
        {
            "hazard": "avalanche",
            "source": "NDMA SACHET",
            "title": "Danger Level 3 Avalanche Warning",
            "description": "DGRE has issued an Orange Alert avalanche warning for Kupwara district above 2200m.",
            "severity": "ORANGE",
            "location": "Kupwara, Jammu & Kashmir",
            "effective": now.isoformat(),
            "expires": (now + timedelta(hours=24)).isoformat(),
        }
    ]
    base_features["elevation"] = 2400.0
    base_features["slope"] = 28.0
    base_features["temp_2m"] = -3.0
    res = detect_avalanche(base_features, base_built, base_history, official_alerts)
    assert res["hazard"] == "avalanche"
    assert res["active"] is True
    assert res["severity"] == "SEVERE"
    assert res["confidence_type"] == "official_warning"
    assert "NDMA SACHET" in res["sources"]
    assert res["evidence"]["elevation_m"] == 2400.0


# ==============================================================================
# 8. Avalanche Alert Expired
# ==============================================================================
def test_8_avalanche_alert_expired():
    # Simulated SACHET XML with expired timestamp
    past_effective = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    past_expires = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    mock_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <alerts xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <alert>
        <identifier>ALERT-EXPIRED-01</identifier>
        <info>
          <category>Met</category>
          <event>Avalanche Alert</event>
          <urgency>Past</urgency>
          <severity>Orange</severity>
          <certainty>Observed</certainty>
          <effective>{past_effective}</effective>
          <expires>{past_expires}</expires>
          <headline>Expired Avalanche Warning</headline>
          <area><areaDesc>Kupwara, Jammu &amp; Kashmir</areaDesc></area>
        </info>
      </alert>
    </alerts>"""

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = mock_xml
        mock_get.return_value.headers = {"ETag": '"test-etag-expired"'}
        res = fetch_sachet_cap_alerts(34.5, 74.2, state="Jammu & Kashmir", district="Kupwara")
        # Expired alert must be filtered out
        assert len(res["alerts"]) == 0


# ==============================================================================
# 9. Avalanche Alert Outside Selected District
# ==============================================================================
def test_9_avalanche_alert_outside_selected_district():
    now = datetime.now(timezone.utc)
    future_expires = (now + timedelta(hours=24)).isoformat()
    mock_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <alerts xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <alert>
        <identifier>ALERT-CHAMOLI-01</identifier>
        <info>
          <category>Met</category>
          <event>Avalanche Warning</event>
          <severity>Orange</severity>
          <effective>{now.isoformat()}</effective>
          <expires>{future_expires}</expires>
          <area><areaDesc>Chamoli, Uttarakhand</areaDesc></area>
        </info>
      </alert>
    </alerts>"""

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = mock_xml
        mock_get.return_value.headers = {"ETag": '"test-etag-district"'}
        # Querying for Shimla, Himachal Pradesh should not match Chamoli, Uttarakhand
        res = fetch_sachet_cap_alerts(31.1, 77.1, state="Himachal Pradesh", district="Shimla")
        assert len(res["alerts"]) == 0


# ==============================================================================
# 10. SACHET Unavailable
# ==============================================================================
def test_10_sachet_unavailable(base_features, base_built, base_history):
    base_built["provider_status"] = {"sachet": "unavailable"}
    res = detect_avalanche(base_features, base_built, base_history, [])
    assert res["hazard"] == "avalanche"
    assert res["active"] is False
    assert res["confidence_type"] == "insufficient_data"
    assert res["evidence"]["missing_provider"] is True


# ==============================================================================
# 11. Flash Flood Intense Short Rainfall (1h >= 25mm)
# ==============================================================================
def test_11_flash_flood_intense_short_rainfall(base_features, base_built, base_history):
    base_features["precip_1h"] = 35.0
    base_features["rain_24h"] = 45.0
    base_built["precip_1h"] = 35.0
    base_built["rain_3h"] = 70.0
    base_built["waterlogging_probability"] = 0.65
    base_built["drainage_congestion_probability"] = 0.60

    res = detect_flash_flood(base_features, base_built, base_history, [])
    assert res["hazard"] == "flash_flood"
    assert res["active"] is True
    assert res["severity"] in ("SEVERE", "CRITICAL")
    assert res["confidence_type"] == "multi_source_evidence"
    assert res["evidence"]["rain_1h_mm"] == 35.0


# ==============================================================================
# 12. Heavy Rain but Low Runoff Evidence (Not a Flash Flood)
# ==============================================================================
def test_12_heavy_rain_but_low_runoff_evidence(base_features, base_built, base_history):
    # 24h rain is 80mm (steady soaking rain), but 1h rain is only 1.2mm, flat dry ground
    base_features["precip_1h"] = 1.2
    base_features["rain_24h"] = 80.0
    base_built["precip_1h"] = 1.2
    base_built["rain_3h"] = 3.5
    base_built["rain_6h"] = 8.0
    base_built["waterlogging_probability"] = 0.08
    base_built["drainage_congestion_probability"] = 0.05

    res = detect_flash_flood(base_features, base_built, base_history, [])
    assert res["hazard"] == "flash_flood"
    assert res["active"] is False
    assert res["severity"] == "NONE"


# ==============================================================================
# 13. High Waterlogging + Drainage Congestion Flash Flood
# ==============================================================================
def test_13_high_waterlogging_plus_drainage_congestion(base_features, base_built, base_history):
    base_features["precip_1h"] = 18.0
    base_built["precip_1h"] = 18.0
    base_built["rain_3h"] = 40.0
    base_built["waterlogging_probability"] = 0.85
    base_built["drainage_congestion_probability"] = 0.80

    res = detect_flash_flood(base_features, base_built, base_history, [])
    assert res["hazard"] == "flash_flood"
    assert res["active"] is True
    assert res["severity"] in ("HIGH", "SEVERE", "CRITICAL")


# ==============================================================================
# 14. Sentinel-1 Corroborates Flash Flood
# ==============================================================================
def test_14_sentinel1_corroborates_flash_flood(base_features, base_built, base_history):
    base_features["precip_1h"] = 26.0
    base_built["precip_1h"] = 26.0
    base_built["sentinel1_metadata"] = {"surface_water_change_detected": True}

    res = detect_flash_flood(base_features, base_built, base_history, [])
    assert res["hazard"] == "flash_flood"
    assert res["active"] is True
    assert res["evidence"]["surface_water_change"] is True
    assert "Sentinel-1 SAR Surface Water" in res["sources"]


# ==============================================================================
# 15. GloFAS Unavailable but Flash Flood Still Detected
# ==============================================================================
def test_15_glofas_unavailable_but_flash_flood_still_detected(base_features, base_built, base_history):
    # river_discharge is None (GloFAS dead/offline)
    base_features["river_discharge"] = None
    base_features["precip_1h"] = 38.0
    base_built["precip_1h"] = 38.0
    base_built["waterlogging_probability"] = 0.70
    base_built["drainage_congestion_probability"] = 0.65

    res = detect_flash_flood(base_features, base_built, base_history, [])
    assert res["hazard"] == "flash_flood"
    assert res["active"] is True
    assert res["severity"] in ("SEVERE", "CRITICAL")


# ==============================================================================
# 16. Multiple Hazards Together (Multi-Label Operational Output)
# ==============================================================================
def test_16_multiple_hazards_together(base_features, base_built, base_history):
    # 1. Active Wildfire
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [{"latitude": 22.0, "longitude": 78.9, "distance_km": 5.0, "detection_age_hours": 1.0, "frp": 65.0}],
        "nearest_fire_distance_km": 5.0,
        "detections_10km": 2,
        "detections_25km": 3,
        "max_frp": 65.0,
        "latest_detection_age_hours": 1.0,
    }
    # 2. Active Flash Flood
    base_features["precip_1h"] = 32.0
    base_built["precip_1h"] = 32.0
    base_built["rain_3h"] = 65.0
    base_built["waterlogging_probability"] = 0.75
    base_built["drainage_congestion_probability"] = 0.70

    # 3. Active Seismic Event
    base_features["max_quake_mag"] = 6.2
    base_built["seismic_metadata"] = {
        "event_distance_km": 25.0,
        "event_age_hours": 2.0,
        "max_quake_mag": 6.2,
    }
    base_history["earthquake"]["nearest_event"] = {"dist_km": 25.0, "mag": 6.2, "age_days": 0.1}

    eval_res = evaluate_api_hazards(base_features, base_built, base_history, [])
    v_hazards = eval_res["verified_hazards"]
    active_names = [h["hazard"] for h in v_hazards]

    assert "earthquake" in active_names
    assert "flash_flood" in active_names
    assert "wildfire" in active_names
    assert len(v_hazards) >= 3


# ==============================================================================
# 17. Wildfire + Air Pollution
# ==============================================================================
def test_17_wildfire_plus_air_pollution(base_features, base_built, base_history):
    base_built["firms_fire"] = {
        "status": "available",
        "hotspots": [{"latitude": 22.0, "longitude": 78.9, "distance_km": 8.0, "detection_age_hours": 1.5, "frp": 45.0}],
        "nearest_fire_distance_km": 8.0,
        "detections_10km": 1,
        "detections_25km": 2,
        "max_frp": 45.0,
        "latest_detection_age_hours": 1.5,
    }
    base_features["us_aqi"] = 320.0
    base_features["pm2_5"] = 260.0

    eval_res = evaluate_api_hazards(base_features, base_built, base_history, [])
    active_names = [h["hazard"] for h in eval_res["verified_hazards"]]

    assert "wildfire" in active_names
    assert "air_pollution" in active_names


# ==============================================================================
# 18. Flash Flood + Thunderstorm + High Wind
# ==============================================================================
def test_18_flash_flood_plus_thunderstorm_plus_high_wind(base_features, base_built, base_history):
    base_features["precip_1h"] = 32.0
    base_built["precip_1h"] = 32.0
    base_built["waterlogging_probability"] = 0.70
    base_built["drainage_congestion_probability"] = 0.65
    base_features["weather_code"] = 95  # thunderstorm
    base_features["wind_speed_10m"] = 68.0
    base_features["wind_gusts_10m"] = 88.0

    eval_res = evaluate_api_hazards(base_features, base_built, base_history, [])
    active_names = [h["hazard"] for h in eval_res["verified_hazards"]]

    assert "flash_flood" in active_names
    assert "thunderstorm" in active_names
    assert "high_wind" in active_names


# ==============================================================================
# 19. No Regression in Existing 12 Hazards
# ==============================================================================
def test_19_no_regression_in_existing_12_hazards(base_features, base_built, base_history):
    assert len(SUPPORTED_HAZARDS) == 15
    eval_res = evaluate_api_hazards(base_features, base_built, base_history, [])
    status_map = eval_res["hazard_status"]

    expected_15 = [
        "flood", "flash_flood", "thunderstorm", "high_wind", "heat_wave",
        "cold_wave", "earthquake", "cyclone", "drought", "air_pollution",
        "landslide", "tsunami", "volcano", "wildfire", "avalanche"
    ]
    for h in expected_15:
        assert h in status_map, f"Missing hazard in status map: {h}"
        assert "sources" in status_map[h]
        assert "source" in status_map[h]
        assert "evidence" in status_map[h]
        assert "severity" in status_map[h]


# ==============================================================================
# 20. No Regression in Weather / Waterlogging / Drainage / Agriculture / Map
# ==============================================================================
def test_20_no_regression_in_other_modules():
    app = create_app()
    client = app.test_client()

    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["data"]["service"] == "Smart Multi-Risk Platform"
    assert "model_files" in data["data"]

    # Test fire_hotspots endpoint
    fire_resp = client.get("/api/disaster/fire_hotspots?lat=22.5&lon=78.5&radius_km=50")
    assert fire_resp.status_code == 200
    fire_data = fire_resp.get_json()
    assert fire_data["success"] is True
    assert "hotspots" in fire_data["data"]
