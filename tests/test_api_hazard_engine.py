"""
Test Suite: API-First Multi-Hazard Operational Detection Engine
Covers all 20 required operational scenarios:
 1. No active hazard
 2. Flood only
 3. Thunderstorm only
 4. Flood + Thunderstorm simultaneously
 5. Cyclone + heavy rain
 6. Active earthquake
 7. Distant old earthquake
 8. Heat wave
 9. Cold wave
 10. Drought
 11. Air pollution
 12. Tsunami evidence
 13. Volcano evidence
 14. XGBoost disagrees with API hazard
 15. Official flood alert overrides unrelated raw ML class
 16. Multiple verified hazards
 17. Provider timeout / fallback handling
 18. Partial data handling
 19. Cache isolation
 20. No regression in other modules
"""

import pytest
from unittest.mock import patch
from backend.services.api_hazard_engine import (
    evaluate_api_hazards,
    detect_earthquake,
    detect_flood,
    detect_cyclone,
    detect_drought,
    detect_heat_wave,
    detect_cold_wave,
    detect_thunderstorm,
    detect_high_wind,
    detect_tsunami,
    detect_volcano,
    detect_air_pollution,
    detect_landslide,
)
from backend.services.disaster_service import (
    corroborate_disaster_evidence,
    predict_disaster_live,
)
from app import create_app


@pytest.fixture
def nominal_features():
    return {
        "temp_2m": 27.0,
        "precip_1h": 0.0,
        "rain_24h": 0.0,
        "rain_48h": 0.0,
        "wind_speed_10m": 12.0,
        "wind_gusts_10m": 18.0,
        "surface_pressure": 1012.0,
        "weather_code": 1,
        "us_aqi": 45.0,
        "pm2_5": 18.0,
        "elevation": 150.0,
        "river_discharge": 2.5,
        "max_quake_mag": 0.0,
        "quake_depth": 0.0,
        "tsunami_flag": 0,
        "gdacs_cyclone": 0,
        "gdacs_volcano": 0,
        "wind_pressure_shear": 0.1,
        "precipitation_saturation_rate": 0.0,
        "thermal_particulate_index": 25.0,
    }


@pytest.fixture
def nominal_built(nominal_features):
    return {
        "features": nominal_features,
        "seismic_metadata": {"event_distance_km": None, "event_age_hours": None, "max_quake_mag": 0.0},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
        "official_alerts": [],
        "official_india": {"official_alerts": []},
        "waterlogging_probability": 0.05,
        "drainage_congestion_probability": 0.05,
    }


@pytest.fixture
def nominal_history():
    return {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0, "nearest_event": None},
        "flood": {"peak_discharge_7d_m3s": 3.0, "discharge_trend": "STEADY", "rainfall_accumulation": {"rain_24h_mm": 0.0, "rain_7d_mm": 0.0, "rain_30d_mm": 20.0}},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL", "rainfall_deficit_pct": 0.0},
    }


# ==============================================================================
# SCENARIO 1: No Active Hazard
# ==============================================================================
def test_scenario_1_no_active_hazard(nominal_features, nominal_built, nominal_history):
    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    assert eval_res["verified_hazards"] == []
    assert eval_res["active_hazard_count"] == 0
    assert eval_res["primary_hazard"]["active"] is False
    assert eval_res["primary_hazard"]["status"] == "NO_ACTIVE_HAZARD"

    raw_res = {"prediction": "nominal", "confidence": 0.95, "probabilities": {"nominal": 0.95, "flood": 0.02}}
    corrob = corroborate_disaster_evidence(raw_res, nominal_built, nominal_history)
    assert corrob["verified_hazards"] == []
    assert corrob["verified_hazard"]["active"] is False
    assert corrob["verified_hazard"]["status"] == "NO_ACTIVE_HAZARD"
    assert "No Active Disaster Threat" in corrob["verified_hazard"]["alert_title"]


# ==============================================================================
# SCENARIO 2: Flood Only
# ==============================================================================
def test_scenario_2_flood_only(nominal_features, nominal_built, nominal_history):
    nominal_features["river_discharge"] = 85.0
    nominal_features["rain_24h"] = 40.0
    nominal_history["flood"]["discharge_trend"] = "RISING"

    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    assert len(eval_res["verified_hazards"]) == 1
    assert eval_res["verified_hazards"][0]["hazard"] == "flood"
    assert eval_res["verified_hazards"][0]["active"] is True
    assert eval_res["verified_hazards"][0]["severity"] in ("HIGH", "CRITICAL")
    assert "flood" in eval_res["hazard_status"]
    assert eval_res["hazard_status"]["flood"]["active"] is True
    assert eval_res["hazard_status"]["earthquake"]["active"] is False


# ==============================================================================
# SCENARIO 3: Thunderstorm Only
# ==============================================================================
def test_scenario_3_thunderstorm_only(nominal_features, nominal_built, nominal_history):
    nominal_features["weather_code"] = 95  # WMO Thunderstorm
    nominal_features["wind_gusts_10m"] = 38.0
    nominal_features["precip_1h"] = 12.0

    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    assert len(eval_res["verified_hazards"]) == 1
    assert eval_res["verified_hazards"][0]["hazard"] == "thunderstorm"
    assert eval_res["verified_hazards"][0]["active"] is True
    assert eval_res["verified_hazards"][0]["severity"] in ("MODERATE", "HIGH", "SEVERE")


# ==============================================================================
# SCENARIO 4: Flood + Thunderstorm Simultaneously (Multi-label)
# ==============================================================================
def test_scenario_4_flood_and_thunderstorm_simultaneously(nominal_features, nominal_built, nominal_history):
    # Flood evidence
    nominal_features["river_discharge"] = 90.0
    nominal_features["rain_24h"] = 45.0
    nominal_history["flood"]["discharge_trend"] = "RISING"
    # Thunderstorm evidence
    nominal_features["weather_code"] = 96  # WMO Thunderstorm with slight hail
    nominal_features["wind_gusts_10m"] = 65.0
    nominal_features["precip_1h"] = 18.0

    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    active_names = [h["hazard"] for h in eval_res["verified_hazards"]]
    assert "flood" in active_names
    assert "thunderstorm" in active_names
    assert len(eval_res["verified_hazards"]) >= 2

    # Check via corroborate_disaster_evidence
    raw_res = {"prediction": "flood", "confidence": 0.60, "probabilities": {"flood": 0.60, "thunderstorm": 0.30}}
    corrob = corroborate_disaster_evidence(raw_res, nominal_built, nominal_history)
    c_names = [h["hazard"] for h in corrob["verified_hazards"]]
    assert "flood" in c_names
    assert "thunderstorm" in c_names


# ==============================================================================
# SCENARIO 5: Cyclone + Heavy Rain
# ==============================================================================
def test_scenario_5_cyclone_and_heavy_rain(nominal_features, nominal_built, nominal_history):
    nominal_features["gdacs_cyclone"] = 1
    nominal_features["wind_speed_10m"] = 72.0
    nominal_features["wind_gusts_10m"] = 98.0
    nominal_features["surface_pressure"] = 982.0
    nominal_features["rain_24h"] = 65.0
    nominal_built["gdacs_metadata"] = {
        "gdacs_cyclone": 1,
        "active_cyclone_details": {"name": "Cyclone Vardah", "distance_km": 140.0}
    }

    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    active_names = [h["hazard"] for h in eval_res["verified_hazards"]]
    assert "cyclone" in active_names
    assert eval_res["verified_hazards"][0]["hazard"] == "cyclone"
    assert eval_res["verified_hazards"][0]["severity"] == "CRITICAL"


# ==============================================================================
# SCENARIO 6: Active Earthquake
# ==============================================================================
def test_scenario_6_active_earthquake(nominal_features, nominal_built, nominal_history):
    nominal_features["max_quake_mag"] = 6.2
    nominal_features["quake_depth"] = 14.0
    nominal_built["seismic_metadata"] = {
        "event_distance_km": 42.0,
        "event_age_hours": 1.2,
        "event_age_days": 0.05,
        "max_quake_mag": 6.2,
    }
    nominal_history["earthquake"]["events_24h"] = 1
    nominal_history["earthquake"]["nearest_event"] = {"distance_km": 42.0, "age_days": 0.05, "mag": 6.2}

    h_eq = detect_earthquake(nominal_features, nominal_built, nominal_history, [])
    assert h_eq["active"] is True
    assert h_eq["severity"] in ("CRITICAL", "SEVERE", "HIGH")
    assert "M6.2" in h_eq["reason"] or "42km" in h_eq["reason"]


# ==============================================================================
# SCENARIO 7: Distant Old Earthquake
# ==============================================================================
def test_scenario_7_distant_old_earthquake(nominal_features, nominal_built, nominal_history):
    nominal_features["max_quake_mag"] = 4.8
    nominal_features["quake_depth"] = 10.0
    nominal_built["seismic_metadata"] = {
        "event_distance_km": 480.0,
        "event_age_hours": 480.0,
        "event_age_days": 20.0,
        "max_quake_mag": 4.8,
    }
    nominal_history["earthquake"]["events_24h"] = 0
    nominal_history["earthquake"]["events_30d"] = 2
    nominal_history["earthquake"]["nearest_event"] = {"distance_km": 480.0, "age_days": 20.0, "mag": 4.8}

    h_eq = detect_earthquake(nominal_features, nominal_built, nominal_history, [])
    assert h_eq["active"] is False
    assert h_eq["severity"] in ("NONE", "LOW")


# ==============================================================================
# SCENARIO 8: Heat Wave
# ==============================================================================
def test_scenario_8_heat_wave(nominal_features, nominal_built, nominal_history):
    nominal_features["temp_2m"] = 44.8
    nominal_features["elevation"] = 80.0  # Plains

    h_hw = detect_heat_wave(nominal_features, nominal_built, nominal_history, [])
    assert h_hw["active"] is True
    assert h_hw["severity"] in ("HIGH", "SEVERE", "CRITICAL")
    assert "44.8" in h_hw["reason"]


# ==============================================================================
# SCENARIO 9: Cold Wave
# ==============================================================================
def test_scenario_9_cold_wave(nominal_features, nominal_built, nominal_history):
    nominal_features["temp_2m"] = 2.1
    nominal_features["elevation"] = 120.0  # Plains

    h_cw = detect_cold_wave(nominal_features, nominal_built, nominal_history, [])
    assert h_cw["active"] is True
    assert h_cw["severity"] in ("SEVERE", "HIGH")
    assert "2.1" in h_cw["reason"]


# ==============================================================================
# SCENARIO 10: Drought
# ==============================================================================
def test_scenario_10_drought(nominal_features, nominal_built, nominal_history):
    nominal_features["temp_2m"] = 36.5
    nominal_history["drought"] = {
        "status": "SEVERE_DEFICIT",
        "rainfall_deficit_pct": -78.0,
        "accumulation_30d_mm": 3.5,
        "is_drought_indicated": True,
        "soil_moisture": 0.11,
    }
    nominal_history["flood"]["rainfall_accumulation"] = {
        "rain_7d_mm": 0.0,
        "rain_30d_mm": 3.5,
    }

    h_dr = detect_drought(nominal_features, nominal_built, nominal_history, [])
    assert h_dr["active"] is True
    assert h_dr["severity"] in ("MODERATE", "HIGH")


# ==============================================================================
# SCENARIO 11: Air Pollution
# ==============================================================================
def test_scenario_11_air_pollution(nominal_features, nominal_built, nominal_history):
    nominal_features["us_aqi"] = 340.0
    nominal_features["pm2_5"] = 290.0

    h_ap = detect_air_pollution(nominal_features, nominal_built, nominal_history, [])
    assert h_ap["active"] is True
    assert h_ap["severity"] == "CRITICAL"


# ==============================================================================
# SCENARIO 12: Tsunami Evidence
# ==============================================================================
def test_scenario_12_tsunami_evidence(nominal_features, nominal_built, nominal_history):
    nominal_features["tsunami_flag"] = 1
    nominal_built["seismic_metadata"]["tsunami_flag"] = 1

    h_ts = detect_tsunami(nominal_features, nominal_built, nominal_history, [])
    assert h_ts["active"] is True
    assert h_ts["severity"] == "CRITICAL"


# ==============================================================================
# SCENARIO 13: Volcano Evidence
# ==============================================================================
def test_scenario_13_volcano_evidence(nominal_features, nominal_built, nominal_history):
    nominal_features["gdacs_volcano"] = 1
    nominal_built["gdacs_metadata"] = {
        "gdacs_volcano": 1,
        "active_volcano_details": {"name": "Barren Island Volcano", "distance_km": 45.0}
    }

    h_vo = detect_volcano(nominal_features, nominal_built, nominal_history, [])
    assert h_vo["active"] is True
    assert h_vo["severity"] == "CRITICAL"


# ==============================================================================
# SCENARIO 14: XGBoost Disagrees with API Hazard
# ==============================================================================
def test_scenario_14_xgboost_disagrees_with_api_hazard(nominal_features, nominal_built, nominal_history):
    # APIs report flood
    nominal_features["river_discharge"] = 120.0
    nominal_features["rain_24h"] = 45.0
    nominal_history["flood"]["discharge_trend"] = "RISING"

    # Raw model predicts earthquake with 58% confidence
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.58,
        "probabilities": {"earthquake": 0.58, "flood": 0.12, "nominal": 0.20}
    }

    corrob = corroborate_disaster_evidence(raw_res, nominal_built, nominal_history)
    # Operational verified hazard must be FLOOD from live APIs
    assert corrob["verified_hazard"]["active"] is True
    assert corrob["verified_hazard"]["hazard"] == "flood"

    # Raw ML signal must remain unchanged and visible for transparency
    assert corrob["raw_model"]["prediction"] == "earthquake"
    assert corrob["raw_model"]["confidence"] == 0.58
    assert corrob["raw_model"]["raw_model_prediction"] == "earthquake"


# ==============================================================================
# SCENARIO 15: Official Flood Alert Overrides Unrelated Raw ML Class
# ==============================================================================
def test_scenario_15_official_flood_alert_overrides_unrelated_raw_ml_class(nominal_features, nominal_built, nominal_history):
    # Official CWC alert
    official_alerts = [{
        "hazard": "flood",
        "source": "Central Water Commission (CWC)",
        "severity": "CRITICAL",
        "title": "Severe Riverine Flood Warning on Ganga River",
        "description": "Water level exceeding danger mark by 1.2 meters.",
    }]

    # Raw ML predicts heat wave
    raw_res = {
        "prediction": "heat_wave",
        "confidence": 0.62,
        "probabilities": {"heat_wave": 0.62, "nominal": 0.20}
    }

    corrob = corroborate_disaster_evidence(
        raw_res, nominal_built, nominal_history, official_alerts=official_alerts
    )
    # Operational hazard is the official Flood
    assert corrob["verified_hazard"]["active"] is True
    assert corrob["verified_hazard"]["hazard"] == "flood"
    assert corrob["verified_hazard"]["severity"] == "CRITICAL"
    assert "CWC" in str(corrob["verified_hazard"]["sources"])

    # Raw model remains heat wave
    assert corrob["raw_model"]["prediction"] == "heat_wave"


# ==============================================================================
# SCENARIO 16: Multiple Verified Hazards
# ==============================================================================
def test_scenario_16_multiple_verified_hazards(nominal_features, nominal_built, nominal_history):
    # Hazardous air pollution + gale force wind
    nominal_features["us_aqi"] = 310.0
    nominal_features["pm2_5"] = 260.0
    nominal_features["wind_speed_10m"] = 58.0
    nominal_features["wind_gusts_10m"] = 78.0

    eval_res = evaluate_api_hazards(nominal_features, nominal_built, nominal_history, [])
    active_hazards = [h["hazard"] for h in eval_res["verified_hazards"]]
    assert "air_pollution" in active_hazards
    assert "high_wind" in active_hazards
    assert len(eval_res["verified_hazards"]) >= 2


# ==============================================================================
# SCENARIO 17: Provider Timeout / Fallback Handling
# ==============================================================================
def test_scenario_17_provider_timeout():
    # If a critical provider fails/times out during feature building
    with patch("backend.services.disaster_features.forecast", side_effect=TimeoutError("Connection timed out")):
        from backend.services.disaster_features import build_disaster_features
        d_built = build_disaster_features(lat=25.5941, lon=85.1376)
        assert d_built is not None
        assert "open_meteo_weather" in d_built["fallback_sources"]
        assert d_built["model_feature_quality"] in ("PARTIAL_DATA", "FALLBACK_USED")


# ==============================================================================
# SCENARIO 18: Partial Data
# ==============================================================================
def test_scenario_18_partial_data(nominal_built):
    nominal_built["model_feature_quality"] = "PARTIAL_DATA"
    nominal_built["fallback_sources"] = ["glofas_flood"]

    # When predicting disaster live with partial data
    with patch("backend.services.disaster_service.build_disaster_features", return_value=nominal_built):
        res = predict_disaster_live(lat=28.6139, lon=77.2090)
        assert res["success"] is True
        assert res["data_quality"]["status"] == "PARTIAL_DATA"
        assert "glofas_flood" in res["data_quality"]["fallback_sources"]


# ==============================================================================
# SCENARIO 19: Cache Isolation by Coordinate
# ==============================================================================
def test_scenario_19_cache_isolation():
    from backend.services.disaster_history_service import get_disaster_history
    hist1 = get_disaster_history(25.5941, 85.1376)  # Patna
    hist2 = get_disaster_history(17.3850, 78.4867)  # Hyderabad
    assert hist1["location"]["lat"] == 25.5941
    assert hist2["location"]["lat"] == 17.3850
    assert hist1 is not hist2


# ==============================================================================
# SCENARIO 20: No Regression in Other Modules
# ==============================================================================
def test_scenario_20_no_regression_in_other_modules():
    app = create_app()
    client = app.test_client()

    payload = {"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Vegetative"}
    res = client.post("/api/predict/all/live", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    # All 4 modules must succeed
    assert data["success"] is True
    assert data["weather"]["success"] is True
    assert data["waterlogging"]["success"] is True
    assert data["agriculture"]["success"] is True
    assert data["disaster"]["success"] is True

    # Check disaster response includes all new multi-hazard operational fields
    d_data = data["disaster"]["data"]
    assert "raw_model" in d_data
    assert "verified_hazards" in d_data
    assert "hazard_status" in d_data
    assert "official_live_evidence" in d_data
    assert "historical_context" in d_data
    assert "data_quality" in d_data
