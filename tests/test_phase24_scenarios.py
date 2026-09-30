"""
Comprehensive Test Suite for Phase 24: Live Hybrid Prediction System Verification
================================================================================
Explicitly covers all 20 required scenarios from Phase 24:
1.  Exact 20-feature contract
2.  Live feature transformations
3.  Model predict/predict_proba
4.  Earthquake old/distant rejection
5.  Active local earthquake support
6.  Flood with official warning
7.  Flood from hydrology + rainfall support
8.  Flood with waterlogging/drainage support
9.  Cyclone active/inactive
10. Drought corroborated/unverified
11. Heat wave
12. Thunderstorm
13. Multiple simultaneous hazards
14. Provider timeout
15. Historical data unavailable
16. Official alert with different raw model class
17. Patna regression
18. Multi-city regression
19. Cache isolation
20. No regression in other platform modules
"""

import math
import pytest
from unittest.mock import patch
import requests

from backend.data_sources.open_meteo import flood, _extract_valid_discharge
from backend.data_sources.india_official import (
    fetch_cwc_flood_alerts,
    fetch_imd_official_warnings,
    fetch_ncs_earthquakes,
    get_official_india_evidence,
)
from backend.services.disaster_features import (
    build_disaster_features,
    EXTERNAL_HAZARD_NAMES,
    FEATURE_CONTRACT_AUDIT,
    get_feature_contract_audit,
)
from backend.services.disaster_service import (
    corroborate_disaster_evidence,
    predict_disaster_live,
    SEISMIC_POLICY,
)
from backend.services.disaster_history_service import get_disaster_history
from backend.services.model_service import registry
from backend.utils.cache import provider_cache
from app import create_app


# ==============================================================================
# 1. EXACT 20-FEATURE CONTRACT
# ==============================================================================
def test_01_exact_20_feature_contract():
    """Requirement 1: Verify exact 20 feature names, exact order, types, and contract audit."""
    expected_order = [
        "temp_2m", "precip_1h", "rain_24h", "rain_48h", "wind_speed_10m",
        "wind_gusts_10m", "surface_pressure", "weather_code", "us_aqi", "pm2_5",
        "river_discharge", "elevation", "max_quake_mag", "quake_depth", "tsunami_flag",
        "gdacs_cyclone", "gdacs_volcano", "wind_pressure_shear",
        "precipitation_saturation_rate", "thermal_particulate_index"
    ]

    mock_precomputed = {
        "forecast": {
            "current": {
                "temperature_2m": 27.5,
                "precipitation": 1.2,
                "wind_speed_10m": 9.5,
                "wind_gusts_10m": 15.0,
                "surface_pressure": 1010.5,
                "weather_code": 3,
            },
            "hourly": {"rain": [0.5] * 48}
        },
        "air_quality": {"current": {"us_aqi": 72.0, "pm2_5": 28.4}},
        "elevation": 55.0,
        "flood": 1.45,
        "usgs": {"max_quake_mag": 0.0, "quake_depth": 0.0, "tsunami_flag": 0},
        "gdacs": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }

    built = build_disaster_features(25.5941, 85.1376, precomputed_sources=mock_precomputed)
    features = built["features"]

    # 1. Exact count and order
    assert list(features.keys()) == expected_order
    assert len(features) == 20

    # 2. Every feature has metadata with source, timestamp, quality, transformation
    meta = built["feature_metadata"]
    assert len(meta) == 20
    for name in expected_order:
        m = meta[name]
        assert "value" in m
        assert "source" in m
        assert "timestamp" in m
        assert m["quality"] in ["LIVE", "DERIVED", "FALLBACK"]
        assert "transformation" in m

    # 3. Contract audit dictionary exists and contains all 20 features
    audit = get_feature_contract_audit()
    assert len(audit) == 20
    for name in expected_order:
        assert name in audit
        assert "training" in audit[name]
        assert "live" in audit[name]
        assert "unit" in audit[name]["training"]
        assert "unit" in audit[name]["live"]


# ==============================================================================
# 2. LIVE FEATURE TRANSFORMATIONS
# ==============================================================================
def test_02_live_feature_transformations():
    """Requirement 2: Audit and verify training/live transformations for engineered and raw features."""
    temp = 30.0
    precip_1h = 5.0
    rain_48h = 20.0
    wind_speed = 12.0
    pressure = 1005.0
    aqi = 110.0
    pm25 = 45.0

    mock_precomputed = {
        "forecast": {
            "current": {
                "temperature_2m": temp,
                "precipitation": precip_1h,
                "wind_speed_10m": wind_speed,
                "wind_gusts_10m": 18.0,
                "surface_pressure": pressure,
                "weather_code": 61,
            },
            "hourly": {"rain": [rain_48h / 48.0] * 48}
        },
        "air_quality": {"current": {"us_aqi": aqi, "pm2_5": pm25}},
        "elevation": 50.0,
        "flood": 3.2,
        "usgs": {"max_quake_mag": 0.0, "quake_depth": 0.0, "tsunami_flag": 0},
        "gdacs": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }

    built = build_disaster_features(20.0, 80.0, precomputed_sources=mock_precomputed)
    f = built["features"]

    # Verify exact mathematical formulas used during model training
    expected_shear = wind_speed / (pressure + 1e-5)
    expected_sat_rate = precip_1h / (rain_48h + 1.0)
    expected_tpi = (pm25 * aqi) / (max(1.0, temp) + 10.0)

    assert abs(f["wind_pressure_shear"] - expected_shear) < 1e-6
    assert abs(f["precipitation_saturation_rate"] - expected_sat_rate) < 1e-6
    assert abs(f["thermal_particulate_index"] - expected_tpi) < 1e-6

    # Verify non-negative, finite numbers
    assert f["river_discharge"] == 3.2
    assert f["elevation"] == 50.0
    assert not math.isnan(f["wind_pressure_shear"])
    assert not math.isinf(f["wind_pressure_shear"])


# ==============================================================================
# 3. MODEL PREDICT / PREDICT_PROBA
# ==============================================================================
def test_03_model_predict_and_predict_proba():
    """Requirement 3: Verify model prediction, predict_proba mapping, and class coverage."""
    sample_features = {
        "temp_2m": 28.5,
        "precip_1h": 0.0,
        "rain_24h": 0.0,
        "rain_48h": 0.0,
        "wind_speed_10m": 8.0,
        "wind_gusts_10m": 12.0,
        "surface_pressure": 1008.0,
        "weather_code": 1,
        "us_aqi": 65.0,
        "pm2_5": 25.0,
        "river_discharge": 0.55,
        "elevation": 42.0,
        "max_quake_mag": 0.0,
        "quake_depth": 0.0,
        "tsunami_flag": 0,
        "gdacs_cyclone": 0,
        "gdacs_volcano": 0,
        "wind_pressure_shear": 8.0 / (1008.0 + 1e-5),
        "precipitation_saturation_rate": 0.0,
        "thermal_particulate_index": (25.0 * 65.0) / (28.5 + 10.0),
    }

    res = registry.predict("disaster", sample_features)

    assert "prediction" in res
    assert isinstance(res["prediction"], str)
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0

    assert "probabilities" in res
    probs = res["probabilities"]
    assert len(probs) == 18

    # Sum of probabilities must equal ~1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3

    # Prediction must match class with maximum probability
    max_class = max(probs, key=probs.get)
    assert res["prediction"] == max_class
    assert abs(res["confidence"] - probs[max_class]) < 1e-4


# ==============================================================================
# 4. EARTHQUAKE OLD / DISTANT REJECTION
# ==============================================================================
def test_04_earthquake_old_distant_rejection():
    """Requirement 4: Distant (>250km) or old (>7 days) earthquakes must be UNCORROBORATED."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.68,
        "probabilities": {"earthquake": 0.68, "nominal": 0.15}
    }

    # Case A: Distant earthquake (320 km away, even if recent)
    d_built_distant = {
        "features": {"max_quake_mag": 5.2, "quake_depth": 15.0, "tsunami_flag": 0, "river_discharge": 0.5},
        "seismic_metadata": {"event_distance_km": 320.0, "event_age_hours": 3.0, "max_quake_mag": 5.2},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 1, "events_7d": 1, "events_30d": 2, "events_90d": 5},
        "flood": {"peak_discharge_7d": 0.5, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_distant = corroborate_disaster_evidence(raw_res, d_built_distant, history)
    v_dist = corrob_distant["verified_hazard"]
    assert v_dist["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v_dist["active"] is False
    assert v_dist["hazard"] is None

    # Case B: Old earthquake (35 days old, even if within 100km)
    d_built_old = {
        "features": {"max_quake_mag": 5.4, "quake_depth": 10.0, "tsunami_flag": 0, "river_discharge": 0.5},
        "seismic_metadata": {"event_distance_km": 80.0, "event_age_hours": 840.0, "max_quake_mag": 5.4},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    corrob_old = corroborate_disaster_evidence(raw_res, d_built_old, history)
    v_old = corrob_old["verified_hazard"]
    assert v_old["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v_old["active"] is False
    assert v_old["hazard"] is None


# ==============================================================================
# 5. ACTIVE LOCAL EARTHQUAKE SUPPORT
# ==============================================================================
def test_05_active_local_earthquake_support():
    """Requirement 5: Active local earthquake (M>=4.5, <250km, <24h) verifies active hazard."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.74,
        "probabilities": {"earthquake": 0.74, "nominal": 0.10}
    }
    d_built = {
        "features": {"max_quake_mag": 5.8, "quake_depth": 12.0, "tsunami_flag": 0, "river_discharge": 0.5},
        "seismic_metadata": {"event_distance_km": 45.0, "event_age_hours": 1.5, "max_quake_mag": 5.8},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 1, "events_7d": 1, "events_30d": 2, "events_90d": 4},
        "flood": {"peak_discharge_7d": 0.5, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history)
    v = corrob["verified_hazard"]
    assert v["status"] in ["VERIFIED_ACTIVE_HAZARD", "SUPPORTED_HAZARD_SIGNAL"]
    assert v["active"] is True
    assert v["hazard"] == "earthquake"
    assert v["severity"] in ["HIGH", "SEVERE", "CRITICAL"]


# ==============================================================================
# 6. FLOOD WITH OFFICIAL WARNING
# ==============================================================================
def test_06_flood_with_official_warning():
    """Requirement 6: Official CWC / State flood alert elevates flood to verified hazard."""
    raw_res = {
        "prediction": "nominal",
        "confidence": 0.85,
        "probabilities": {"nominal": 0.85, "flood": 0.05}
    }
    d_built = {
        "features": {"river_discharge": 5.0, "rain_24h": 5.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 5.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 10.0},
        "drought": {"status": "NORMAL"}
    }
    official_alerts = [
        {
            "hazard": "flood",
            "severity": "ORANGE",
            "source": "Central Water Commission (CWC)",
            "title": "Severe Flood Warning - Ganga Basin at Patna",
            "location": "Patna",
            "issued_at": "2026-09-29T10:00:00Z"
        }
    ]

    corrob = corroborate_disaster_evidence(raw_res, d_built, history, official_alerts=official_alerts)
    v = corrob["verified_hazard"]
    assert v["status"] == "VERIFIED_ACTIVE_HAZARD"
    assert v["active"] is True
    assert v["hazard"] == "flood"
    assert "cwc" in v["reason"].lower() or "official" in v["reason"].lower()


# ==============================================================================
# 7. FLOOD FROM HYDROLOGY + RAINFALL SUPPORT
# ==============================================================================
def test_07_flood_hydrology_and_rainfall_support():
    """Requirement 7: Live discharge surge + heavy rainfall corroborates flood."""
    raw_res = {
        "prediction": "flood",
        "confidence": 0.76,
        "probabilities": {"flood": 0.76, "nominal": 0.12}
    }
    d_built = {
        "features": {"river_discharge": 28.5, "rain_24h": 45.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 28.5, "discharge_trend": "RISING"},
        "rainfall": {"accumulation_7d_mm": 85.0},
        "drought": {"status": "NORMAL"}
    }

    corrob = corroborate_disaster_evidence(raw_res, d_built, history)
    v = corrob["verified_hazard"]
    assert v["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert v["active"] is True
    assert v["hazard"] == "flood"
    assert "discharge" in v["reason"].lower() or "rain" in v["reason"].lower()


# ==============================================================================
# 8. FLOOD WITH WATERLOGGING / DRAINAGE SUPPORT
# ==============================================================================
def test_08_flood_waterlogging_and_drainage_support():
    """Requirement 8: Elevated waterlogging & drainage congestion support flood prediction."""
    raw_res = {
        "prediction": "flood",
        "confidence": 0.65,
        "probabilities": {"flood": 0.65, "nominal": 0.20}
    }
    d_built = {
        "features": {"river_discharge": 8.0, "rain_24h": 12.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
        "waterlogging_probability": 0.52,
        "drainage_congestion_probability": 0.58
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 8.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 15.0},
        "drought": {"status": "NORMAL"}
    }

    corrob = corroborate_disaster_evidence(raw_res, d_built, history)
    v = corrob["verified_hazard"]
    assert v["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert v["active"] is True
    assert v["hazard"] == "flood"
    assert "waterlogging" in v["reason"].lower() or "drainage" in v["reason"].lower()


# ==============================================================================
# 9. CYCLONE ACTIVE / INACTIVE
# ==============================================================================
def test_09_cyclone_active_vs_inactive():
    """Requirement 9: Active regional cyclone corroborates alert; absence rejects raw ML signal."""
    raw_res = {
        "prediction": "cyclone",
        "confidence": 0.70,
        "probabilities": {"cyclone": 0.70, "nominal": 0.15}
    }

    # Case A: Active GDACS cyclone nearby with severe winds
    d_built_active = {
        "features": {
            "wind_speed_10m": 24.0,
            "wind_gusts_10m": 35.0,
            "surface_pressure": 985.0,
            "rain_24h": 15.0,
            "gdacs_cyclone": 1.0,
        },
        "seismic_metadata": {},
        "gdacs_metadata": {
            "gdacs_cyclone": 1,
            "gdacs_volcano": 0,
            "active_cyclone_details": {"name": "Test Cyclone", "distance_km": 180.0, "severity": "ORANGE"}
        }
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 2.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 60.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_act = corroborate_disaster_evidence(raw_res, d_built_active, history)
    v_act = corrob_act["verified_hazard"]
    assert v_act["status"] == "VERIFIED_ACTIVE_HAZARD"
    assert v_act["active"] is True
    assert v_act["hazard"] == "cyclone"

    # Case B: No active cyclone feed -> rejected to prevent hallucinated tropical storm
    d_built_inactive = {
        "features": {"wind_speed_10m": 8.0, "wind_gusts_10m": 12.0, "surface_pressure": 1012.0, "rain_24h": 0.0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    corrob_inact = corroborate_disaster_evidence(raw_res, d_built_inactive, history)
    v_inact = corrob_inact["verified_hazard"]
    assert v_inact["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v_inact["active"] is False
    assert v_inact["hazard"] is None


# ==============================================================================
# 10. DROUGHT CORROBORATED / UNVERIFIED
# ==============================================================================
def test_10_drought_corroborated_vs_unverified():
    """Requirement 10: Multi-week rainfall deficit corroborates drought; nominal rain rejects it."""
    raw_res = {
        "prediction": "drought",
        "confidence": 0.68,
        "probabilities": {"drought": 0.68, "nominal": 0.20}
    }

    # Case A: Corroborated with severe rainfall deficit
    d_built = {
        "features": {"temp_2m": 38.0, "rain_24h": 0.0, "rain_48h": 0.0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    hist_deficit = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 0.1, "discharge_trend": "RECEDING"},
        "rainfall": {"accumulation_7d_mm": 0.0, "accumulation_30d_mm": 2.0},
        "drought": {"status": "SEVERE_DEFICIT", "rainfall_deficit_pct": -80.0}
    }
    corrob_drought = corroborate_disaster_evidence(raw_res, d_built, hist_deficit)
    v_d = corrob_drought["verified_hazard"]
    assert v_d["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert v_d["active"] is True
    assert v_d["hazard"] == "drought"

    # Case B: Unverified with healthy rainfall
    hist_healthy = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 5.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 45.0, "accumulation_30d_mm": 180.0},
        "drought": {"status": "NORMAL", "rainfall_deficit_pct": 5.0}
    }
    corrob_unver = corroborate_disaster_evidence(raw_res, d_built, hist_healthy)
    v_u = corrob_unver["verified_hazard"]
    assert v_u["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v_u["active"] is False
    assert v_u["hazard"] is None


# ==============================================================================
# 11. HEAT WAVE
# ==============================================================================
def test_11_heat_wave_corroboration():
    """Requirement 11: Heat wave requires extreme thermal conditions (>40°C) or official alert."""
    raw_res = {
        "prediction": "heat_wave",
        "confidence": 0.65,
        "probabilities": {"heat_wave": 0.65, "nominal": 0.20}
    }
    # Severe thermal condition
    d_built_hot = {
        "features": {"temp_2m": 43.5, "wind_speed_10m": 4.0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 0.1, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built_hot, history)
    v = corrob["verified_hazard"]
    assert v["status"] in ["VERIFIED_ACTIVE_HAZARD", "SUPPORTED_HAZARD_SIGNAL"]
    assert v["active"] is True
    assert v["hazard"] == "heat_wave"

    # Mild temperature (28°C) rejects heat wave
    d_built_mild = {
        "features": {"temp_2m": 28.0, "wind_speed_10m": 4.0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    corrob_mild = corroborate_disaster_evidence(raw_res, d_built_mild, history)
    assert corrob_mild["verified_hazard"]["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert corrob_mild["verified_hazard"]["active"] is False


# ==============================================================================
# 12. THUNDERSTORM
# ==============================================================================
def test_12_thunderstorm_corroboration():
    """Requirement 12: Thunderstorm corroborated by WMO convective code or gale wind gusts."""
    raw_res = {
        "prediction": "thunderstorm",
        "confidence": 0.72,
        "probabilities": {"thunderstorm": 0.72, "nominal": 0.15}
    }
    d_built = {
        "features": {
            "weather_code": 95,       # Thunderstorm slight or moderate
            "wind_gusts_10m": 22.0,
            "precip_1h": 12.0
        },
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 1.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 12.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history)
    v = corrob["verified_hazard"]
    assert v["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert v["active"] is True
    assert v["hazard"] == "thunderstorm"


# ==============================================================================
# 13. MULTIPLE SIMULTANEOUS HAZARDS
# ==============================================================================
def test_13_multiple_simultaneous_hazards():
    """Requirement 13: Platform returns verified_hazards list supporting concurrent risks."""
    raw_res = {
        "prediction": "thunderstorm",
        "confidence": 0.60,
        "probabilities": {"thunderstorm": 0.60, "flood": 0.25}
    }
    d_built = {
        "features": {"weather_code": 95, "wind_gusts_10m": 24.0, "river_discharge": 18.0, "rain_24h": 35.0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 18.0, "discharge_trend": "RISING"},
        "rainfall": {"accumulation_7d_mm": 50.0},
        "drought": {"status": "NORMAL"}
    }
    official_alerts = [
        {
            "hazard": "flood",
            "severity": "ORANGE",
            "source": "CWC",
            "title": "Severe Riverine Flood Advisory",
            "location": "Local District",
            "issued_at": "2026-09-29T08:00:00Z"
        }
    ]

    corrob = corroborate_disaster_evidence(raw_res, d_built, history, official_alerts=official_alerts)
    assert "verified_hazards" in corrob
    vh_list = corrob["verified_hazards"]
    assert len(vh_list) >= 2

    hazards_present = {h["hazard"] for h in vh_list}
    assert "flood" in hazards_present
    assert "thunderstorm" in hazards_present


# ==============================================================================
# 14. PROVIDER TIMEOUT
# ==============================================================================
def test_14_provider_timeout_graceful_handling():
    """Requirement 14: Provider timeout falls back safely without unhandled exception."""
    provider_cache.clear()
    with patch("backend.data_sources.open_meteo.get_json", side_effect=requests.exceptions.Timeout("Connection timed out")):
        built = build_disaster_features(16.3067, 80.4365)
        assert built is not None
        assert "features" in built
        assert len(built["features"]) == 20
        # Data quality must reflect fallback or incompleteness
        assert built["model_feature_quality"] in ["PARTIAL_DATA", "FALLBACK_USED", "DATA_INCOMPLETE"]
        status_val = built["provider_status"].get("weather") or built["provider_status"].get("open_meteo_weather", "")
        if isinstance(status_val, dict):
            status_val = status_val.get("status", "")
        assert any(term in str(status_val).lower() for term in ["error", "unavailable", "failed", "fallback", "part"])


# ==============================================================================
# 15. HISTORICAL DATA UNAVAILABLE
# ==============================================================================
def test_15_historical_data_unavailable():
    """Requirement 15: Failure of historical feed produces graceful fallback without crashing."""
    raw_res = {
        "prediction": "flood",
        "confidence": 0.70,
        "probabilities": {"flood": 0.70, "nominal": 0.20}
    }
    d_built = {
        "features": {"river_discharge": 25.0, "rain_24h": 40.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    # Empty or unavailable historical context
    history_empty = {
        "earthquake": {"status": "UNAVAILABLE", "events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"status": "UNAVAILABLE", "peak_discharge_7d": 25.0, "discharge_trend": "UNKNOWN"},
        "rainfall": {"status": "UNAVAILABLE", "accumulation_7d_mm": 0.0},
        "drought": {"status": "UNAVAILABLE"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_empty)
    assert corrob is not None
    assert "verified_hazard" in corrob
    assert corrob["verified_hazard"]["active"] is True
    assert corrob["verified_hazard"]["hazard"] == "flood"


# ==============================================================================
# 16. OFFICIAL ALERT WITH DIFFERENT RAW MODEL CLASS
# ==============================================================================
def test_16_official_alert_with_different_raw_model_class():
    """Requirement 16: Raw ML prediction is preserved even when official alert dictates verified hazard."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.54,
        "probabilities": {"earthquake": 0.54, "nominal": 0.30, "flood": 0.08}
    }
    d_built = {
        "features": {"river_discharge": 4.5, "rain_24h": 10.0, "max_quake_mag": 4.1, "quake_depth": 10.0, "tsunami_flag": 0},
        "seismic_metadata": {"event_distance_km": 350.0, "event_age_hours": 200.0, "max_quake_mag": 4.1},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 1, "events_90d": 2},
        "flood": {"peak_discharge_7d": 4.5, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 15.0},
        "drought": {"status": "NORMAL"}
    }
    official_alerts = [
        {
            "hazard": "flood",
            "severity": "RED",
            "source": "State Disaster Management Authority",
            "title": "Severe Flash Flood Warning",
            "location": "Target District",
            "issued_at": "2026-09-29T11:00:00Z"
        }
    ]

    corrob = corroborate_disaster_evidence(raw_res, d_built, history, official_alerts=official_alerts)

    # 1. Raw model signal preserved
    assert corrob["raw_model"]["prediction"] == "earthquake"
    assert corrob["raw_model"]["confidence"] == 0.54

    # 2. Official alert captured
    assert len(corrob["official_live_evidence"]["official_alerts"]) == 1
    assert corrob["official_live_evidence"]["official_alerts"][0]["hazard"] == "flood"

    # 3. Verified operational hazard elevated to flood
    v = corrob["verified_hazard"]
    assert v["hazard"] == "flood"
    assert v["severity"] in ["CRITICAL", "RED"]
    assert v["active"] is True
    assert v["status"] == "VERIFIED_ACTIVE_HAZARD"


# ==============================================================================
# 17. PATNA REGRESSION (NO FALSE EARTHQUAKE)
# ==============================================================================
def test_17_patna_regression_no_false_earthquake():
    """Requirement 17: Live Patna request must not show active earthquake alarm."""
    res = predict_disaster_live(25.5941, 85.1376, state="Bihar", district="Patna")
    assert res["success"] is True

    # Raw model must be visible
    assert "raw_model" in res
    assert "prediction" in res["raw_model"]

    # Verified hazard must NOT be an active earthquake!
    v = res["verified_hazard"]
    if v.get("hazard") == "earthquake":
        assert v.get("active") is False, "Patna must not report active earthquake hazard"
    assert v["status"] in ["UNCORROBORATED_MODEL_SIGNAL", "NO_ACTIVE_HAZARD", "SUPPORTED_HAZARD_SIGNAL", "VERIFIED_ACTIVE_HAZARD"]


# ==============================================================================
# 18. MULTI-CITY REGRESSION (12 INDIAN CITIES)
# ==============================================================================
def test_18_multi_city_regression():
    """Requirement 18: Live pipeline executes across 12 diverse Indian cities."""
    benchmark_cities = [
        ("Patna", "Bihar", 25.5941, 85.1376),
        ("Guntur", "Andhra Pradesh", 16.3067, 80.4365),
        ("Mumbai", "Maharashtra", 19.0760, 72.8777),
        ("Delhi", "Delhi", 28.6139, 77.2090),
        ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
        ("Kochi", "Kerala", 9.9312, 76.2673),
        ("Guwahati", "Assam", 26.1445, 91.7362),
        ("Hyderabad", "Telangana", 17.3850, 78.4867),
        ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
        ("Dehradun", "Uttarakhand", 30.3165, 78.0322),
        ("Jaipur", "Rajasthan", 26.9124, 75.7873),
        ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185),
    ]

    for name, state, lat, lon in benchmark_cities:
        res = predict_disaster_live(lat, lon, state=state, district=name)
        assert res["success"] is True, f"Failed pipeline for {name}"
        assert "raw_model" in res
        assert "verified_hazard" in res
        assert "model_feature_quality" in res
        assert len(res["raw_model"]["probabilities"]) == 18


# ==============================================================================
# 19. CACHE ISOLATION
# ==============================================================================
def test_19_cache_isolation():
    """Requirement 19: Cache keys isolate coordinates to avoid cross-city contamination."""
    provider_cache.clear()

    key_patna = ("test_weather", round(25.5941, 2), round(85.1376, 2))
    key_guntur = ("test_weather", round(16.3067, 2), round(80.4365, 2))

    provider_cache.set(key_patna, {"temp": 32.0, "city": "Patna"})
    provider_cache.set(key_guntur, {"temp": 28.0, "city": "Guntur"})

    val_patna = provider_cache.get(key_patna)
    val_guntur = provider_cache.get(key_guntur)

    assert val_patna["city"] == "Patna"
    assert val_guntur["city"] == "Guntur"
    assert val_patna["temp"] != val_guntur["temp"]


# ==============================================================================
# 20. NO REGRESSION IN OTHER PLATFORM MODULES
# ==============================================================================
def test_20_no_regression_in_other_platform_modules():
    """Requirement 20: Unified prediction endpoint executes Weather, Disaster, Agriculture, Waterlogging without regression."""
    app = create_app()
    client = app.test_client()

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
    assert json_data["weather"]["success"] is True
    assert json_data["disaster"]["success"] is True
    assert json_data["agriculture"]["success"] is True
    assert json_data["waterlogging"]["success"] is True
    assert "summary" in json_data
