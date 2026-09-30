import pytest
import math
from unittest.mock import patch
import requests

from backend.data_sources.open_meteo import flood, _extract_valid_discharge
from backend.services.disaster_features import build_disaster_features, EXTERNAL_HAZARD_NAMES
from backend.services.model_service import registry
from app import create_app


def test_1_successful_river_discharge_extraction():
    """Test 1: Successful river-discharge extraction from realistic GloFAS response."""
    mock_response = {
        "daily": {
            "time": ["2026-09-27", "2026-09-28"],
            "river_discharge": [0.80, 0.47],
        },
        "daily_units": {"river_discharge": "m³/s"}
    }
    with patch("backend.data_sources.open_meteo.get_json", return_value=mock_response):
        discharge = flood(16.3067, 80.4365)
        assert discharge is not None
        assert isinstance(discharge, float)
        assert abs(discharge - 0.47) < 1e-4

        # Verify detailed metadata mode
        details = flood(16.3067, 80.4365, return_details=True)
        assert details["status"] == "available"
        assert details["unit"] == "m³/s"
        assert details["river_discharge"] == 0.47


def test_2_empty_hydrology_response():
    """Test 2: Empty hydrology response returns None without fabricating fake data."""
    mock_null_response = {
        "daily": {
            "time": ["2026-09-27", "2026-09-28"],
            "river_discharge": [None, None],
        }
    }
    with patch("backend.data_sources.open_meteo.get_json", return_value=mock_null_response):
        discharge = flood(0.0, 0.0)
        assert discharge is None

    # Empty dictionary
    with patch("backend.data_sources.open_meteo.get_json", return_value={}):
        discharge = flood(0.0, 0.0)
        assert discharge is None


def test_3_api_timeout():
    """Test 3: API timeout is caught gracefully and returns None without crashing."""
    from backend.utils.cache import provider_cache
    provider_cache.clear()
    with patch("backend.data_sources.open_meteo.get_json", side_effect=requests.exceptions.Timeout("Read timed out")):
        discharge = flood(16.3067, 80.4365)
        assert discharge is None

        details = flood(16.3067, 80.4365, return_details=True)
        assert details["status"] == "unavailable"
        assert details["river_discharge"] is None
        assert "timed out" in details["reason"].lower()


def test_4_invalid_numeric_values():
    """Test 4: Invalid numeric values (NaN, infinity, non-numeric strings) are rejected."""
    # NaN and inf
    val, _ = _extract_valid_discharge({
        "time": ["2026-09-28"],
        "river_discharge": [float("nan")]
    })
    assert val is None

    val, _ = _extract_valid_discharge({
        "time": ["2026-09-28"],
        "river_discharge": [float("inf")]
    })
    assert val is None

    # Non-numeric string
    val, _ = _extract_valid_discharge({
        "time": ["2026-09-28"],
        "river_discharge": ["not_a_number"]
    })
    assert val is None

    # Mixed list with one valid prior observation
    val, _ = _extract_valid_discharge({
        "time": ["2026-09-27", "2026-09-28"],
        "river_discharge": [1.25, None]
    })
    assert val == 1.25


def test_5_complete_20_feature_disaster_vector():
    """Test 5: Feature builder produces exactly 20 valid features in expected order."""
    expected_20 = [
        "temp_2m", "precip_1h", "rain_24h", "rain_48h", "wind_speed_10m",
        "wind_gusts_10m", "surface_pressure", "weather_code", "us_aqi", "pm2_5",
        "river_discharge", "elevation", "max_quake_mag", "quake_depth", "tsunami_flag",
        "gdacs_cyclone", "gdacs_volcano", "wind_pressure_shear",
        "precipitation_saturation_rate", "thermal_particulate_index"
    ]

    mock_precomputed = {
        "forecast": {
            "current": {
                "temperature_2m": 28.5,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
                "wind_gusts_10m": 12.0,
                "surface_pressure": 1008.0,
                "weather_code": 1,
            },
            "hourly": {
                "rain": [0.0] * 48
            }
        },
        "air_quality": {
            "current": {"us_aqi": 65.0, "pm2_5": 25.0}
        },
        "elevation": 42.0,
        "flood": 0.55,
        "usgs": {"max_quake_mag": 0.0, "quake_depth": 0.0, "tsunami_flag": 0},
        "gdacs": {"gdacs_cyclone": 0, "gdacs_volcano": 0}
    }

    built = build_disaster_features(16.3067, 80.4365, precomputed_sources=mock_precomputed)
    features = built["features"]

    assert len(features) == 20
    assert built["missing_external_features"] == []
    assert built["diagnostics"]["missing_features"] == []
    assert built["diagnostics"]["invalid_features"] == []

    for name in expected_20:
        assert name in features
        assert features[name] is not None
        assert isinstance(features[name], (int, float))
        assert not (math.isnan(features[name]) or math.isinf(features[name]))


def test_6_disaster_inference_success():
    """Test 6: Disaster XGBoost model predicts successfully with all 20 features."""
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
    assert len(res["probabilities"]) == 18
    assert len(res["used_features"]) == 20


def test_7_unified_endpoint_4_of_4_success():
    """Test 7: Unified endpoint returns 4/4 model success flags."""
    app = create_app()
    client = app.test_client()

    payload = {
        "lat": 16.3067,
        "lon": 80.4365,
        "crop": "Rice",
        "crop_stage": "Flowering"
    }

    res = client.post("/api/predict/all/live", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert data["success"] is True
    assert data["weather"]["success"] is True
    assert data["disaster"]["success"] is True
    assert data["agriculture"]["success"] is True
    assert data["waterlogging"]["success"] is True

    # Confirm disaster output
    d_res = data["disaster"]["data"]["result"]
    assert d_res["prediction"] is not None
    assert d_res["confidence"] > 0

    # Confirm river discharge in input features
    d_feat = data["disaster"]["data"]["input"]["features"]
    assert d_feat["river_discharge"] is not None
    assert isinstance(d_feat["river_discharge"], (int, float))


# ==============================================================================
# DISASTER HYBRID PIPELINE TESTS (ITEMS 1-19)
# ==============================================================================
from backend.services.disaster_service import (
    corroborate_disaster_evidence,
    predict_disaster_live,
    SEISMIC_POLICY,
)
from backend.services.disaster_history_service import get_disaster_history


def test_8_no_earthquake_event():
    """Test 8: Zero earthquake activity results in clean nominal verified status."""
    raw_res = {
        "prediction": "nominal",
        "confidence": 0.88,
        "probabilities": {"nominal": 0.88, "earthquake": 0.02}
    }
    d_built = {
        "features": {"max_quake_mag": 0.0, "quake_depth": 0.0, "tsunami_flag": 0, "river_discharge": 0.5},
        "seismic_metadata": {"event_distance_km": None, "event_age_hours": None, "max_quake_mag": 0.0},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_data = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0, "nearest_event": None},
        "flood": {"peak_discharge_7d": 0.5, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_data)
    v = corrob["verified_hazard"]
    assert v["status"] == "NO_ACTIVE_HAZARD"
    assert v["active"] is False
    assert v["hazard"] is None


def test_9_distant_earthquake_uncorroborated():
    """Test 9: Distant earthquake (>250km, >7 days old) does NOT become active alert."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.71,
        "probabilities": {"earthquake": 0.71, "nominal": 0.15}
    }
    d_built = {
        "features": {"max_quake_mag": 4.9, "quake_depth": 10.0, "tsunami_flag": 0, "river_discharge": 5.0},
        "seismic_metadata": {"event_distance_km": 418.0, "event_age_hours": 490.0, "max_quake_mag": 4.9},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_data = {
        "earthquake": {
            "events_24h": 0, "events_7d": 0, "events_30d": 2, "events_90d": 6,
            "nearest_event": {"distance_km": 418.0, "age_days": 20.4, "mag": 4.9},
        },
        "flood": {"peak_discharge_7d": 12.0, "discharge_trend": "FALLING"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_data)
    v = corrob["verified_hazard"]
    # Must preserve raw model prediction
    assert corrob["raw_model"]["prediction"] == "earthquake"
    assert corrob["raw_model"]["confidence"] == 0.71
    # Verified hazard must NOT be active
    assert v["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v["active"] is False
    assert v["hazard"] is None
    assert "no recent locally relevant seismic event" in v["reason"].lower()


def test_10_old_earthquake_uncorroborated():
    """Test 10: Local but old earthquake (>7 days old) is historical context only."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.65,
        "probabilities": {"earthquake": 0.65}
    }
    d_built = {
        "features": {"max_quake_mag": 4.2, "quake_depth": 15.0, "tsunami_flag": 0, "river_discharge": 2.0},
        "seismic_metadata": {"event_distance_km": 80.0, "event_age_hours": 360.0, "max_quake_mag": 4.2},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_data = {
        "earthquake": {
            "events_24h": 0, "events_7d": 0, "events_30d": 1, "events_90d": 3,
            "nearest_event": {"distance_km": 80.0, "age_days": 15.0, "mag": 4.2},
        },
        "flood": {"peak_discharge_7d": 2.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_data)
    v = corrob["verified_hazard"]
    assert v["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert v["active"] is False


def test_11_active_local_earthquake_verified():
    """Test 11: Real active local earthquake (M5.2, <100km, <6h) verifies active hazard."""
    raw_res = {
        "prediction": "earthquake",
        "confidence": 0.85,
        "probabilities": {"earthquake": 0.85}
    }
    d_built = {
        "features": {"max_quake_mag": 5.2, "quake_depth": 12.0, "tsunami_flag": 0, "river_discharge": 3.0},
        "seismic_metadata": {"event_distance_km": 35.0, "event_age_hours": 1.5, "max_quake_mag": 5.2},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_data = {
        "earthquake": {
            "events_24h": 1, "events_7d": 1, "events_30d": 2, "events_90d": 5,
            "nearest_event": {"distance_km": 35.0, "age_days": 0.06, "mag": 5.2},
        },
        "flood": {"peak_discharge_7d": 3.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_data)
    v = corrob["verified_hazard"]
    assert v["status"] == "VERIFIED_ACTIVE_HAZARD"
    assert v["active"] is True
    assert v["hazard"] == "earthquake"
    assert v["severity"] in ["CRITICAL", "HIGH"]


def test_12_historical_earthquake_does_not_create_active_alert():
    """Test 12: Frequent past earthquakes (90d) do NOT create current alert if model says nominal."""
    raw_res = {
        "prediction": "nominal",
        "confidence": 0.90,
        "probabilities": {"nominal": 0.90, "earthquake": 0.05}
    }
    d_built = {
        "features": {"max_quake_mag": 0.0, "quake_depth": 0.0, "tsunami_flag": 0, "river_discharge": 1.0},
        "seismic_metadata": {"event_distance_km": None, "event_age_hours": None, "max_quake_mag": 0.0},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_data = {
        "earthquake": {
            "events_24h": 0, "events_7d": 0, "events_30d": 8, "events_90d": 25,
            "nearest_event": {"distance_km": 150.0, "age_days": 22.0, "mag": 3.8},
        },
        "flood": {"peak_discharge_7d": 1.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built, history_data)
    v = corrob["verified_hazard"]
    assert v["status"] == "NO_ACTIVE_HAZARD"
    assert v["active"] is False
    assert v["hazard"] is None


def test_13_live_flood_support_vs_uncorroborated():
    """Test 13: Flood corroborated with elevated river flow vs uncorroborated with dry river."""
    # Case A: Elevated river flow + rising trend
    raw_res = {"prediction": "flood", "confidence": 0.78, "probabilities": {"flood": 0.78}}
    d_built_high = {
        "features": {"river_discharge": 350.0, "rain_24h": 45.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_high = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 380.0, "discharge_trend": "RISING"},
        "rainfall": {"accumulation_7d_mm": 80.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_a = corroborate_disaster_evidence(raw_res, d_built_high, history_high)
    assert corrob_a["verified_hazard"]["status"] == "VERIFIED_ACTIVE_HAZARD"
    assert corrob_a["verified_hazard"]["active"] is True
    assert corrob_a["verified_hazard"]["hazard"] == "flood"

    # Case B: Very low river flow, zero rainfall
    d_built_low = {
        "features": {"river_discharge": 0.2, "rain_24h": 0.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }
    history_low = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 0.3, "discharge_trend": "FALLING"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_b = corroborate_disaster_evidence(raw_res, d_built_low, history_low)
    assert corrob_b["verified_hazard"]["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert corrob_b["verified_hazard"]["active"] is False


def test_14_drought_corroboration_with_and_without_rainfall_deficit():
    """Test 14: Drought signal verified with deficit vs rejected when recent rain is ample."""
    raw_res = {"prediction": "drought", "confidence": 0.62, "probabilities": {"drought": 0.62}}
    d_built = {
        "features": {"temp_2m": 38.0, "rain_24h": 0.0, "river_discharge": 0.1, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
    }

    # Case A: Deficit supported
    hist_deficit = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 0.1, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 0.0},
        "drought": {"status": "RAINFALL_DEFICIT", "deficit": True}
    }
    corrob_a = corroborate_disaster_evidence(raw_res, d_built, hist_deficit)
    assert corrob_a["verified_hazard"]["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert corrob_a["verified_hazard"]["active"] is True
    assert corrob_a["verified_hazard"]["hazard"] == "drought"

    # Case B: Recent rain is 75mm (no deficit)
    hist_wet = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 2.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 75.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_b = corroborate_disaster_evidence(raw_res, d_built, hist_wet)
    assert corrob_b["verified_hazard"]["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert corrob_b["verified_hazard"]["active"] is False


def test_15_gdacs_cyclone_active_vs_inactive():
    """Test 15: Cyclone corroborated by active GDACS feed vs uncorroborated when absent."""
    raw_res = {"prediction": "cyclone", "confidence": 0.81, "probabilities": {"cyclone": 0.81}}

    # Case A: Active GDACS cyclone
    d_built_active = {
        "features": {"wind_speed_10m": 65.0, "wind_gusts_10m": 95.0, "surface_pressure": 985.0, "gdacs_cyclone": 1, "river_discharge": 10.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 1, "event_type": "TC", "alert_level": "Red", "distance_km": 180.0},
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 10.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 50.0},
        "drought": {"status": "NORMAL"}
    }
    corrob_active = corroborate_disaster_evidence(raw_res, d_built_active, history)
    assert corrob_active["verified_hazard"]["status"] == "VERIFIED_ACTIVE_HAZARD"
    assert corrob_active["verified_hazard"]["active"] is True
    assert corrob_active["verified_hazard"]["hazard"] == "cyclone"

    # Case B: GDACS has no cyclone alerts
    d_built_inactive = {
        "features": {"wind_speed_10m": 12.0, "wind_gusts_10m": 18.0, "surface_pressure": 1012.0, "gdacs_cyclone": 0, "river_discharge": 1.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0},
    }
    corrob_inactive = corroborate_disaster_evidence(raw_res, d_built_inactive, history)
    assert corrob_inactive["verified_hazard"]["status"] == "UNCORROBORATED_MODEL_SIGNAL"
    assert corrob_inactive["verified_hazard"]["active"] is False


def test_16_cache_isolation_by_coordinate():
    """Test 16: Different coordinates do NOT share or overwrite historical cache."""
    hist_patna = get_disaster_history(25.5941, 85.1376)
    hist_hyd = get_disaster_history(17.3850, 78.4867)

    assert hist_patna is not None
    assert hist_hyd is not None
    assert hist_patna["location"]["lat"] == 25.5941
    assert hist_hyd["location"]["lat"] == 17.3850


def test_17_patna_regression_not_active_earthquake():
    """Test 17 (Patna Regression): Patna live run preserves raw ML signal but marks verified hazard inactive."""
    res = predict_disaster_live(25.5941, 85.1376, overrides={"us_aqi": 30.0, "pm2_5": 10.0})
    assert res["success"] is True

    raw = res["raw_model"]
    verified = res["verified_hazard"]

    # Raw model must be preserved
    assert "prediction" in raw
    assert "confidence" in raw
    assert "probabilities" in raw
    assert len(raw["probabilities"]) == 18

    # Patna verified hazard MUST NOT be an active earthquake!
    assert verified["active"] is False
    assert verified["hazard"] is None or verified["hazard"] != "earthquake"
    assert verified["status"] in ["UNCORROBORATED_MODEL_SIGNAL", "NO_ACTIVE_HAZARD"]
    assert "Uncorroborated" in verified["alert_title"] or "No active" in verified["alert_title"]


def test_18_disaster_history_endpoint():
    """Test 18: GET /api/disaster/history returns valid multi-window context."""
    app = create_app()
    client = app.test_client()

    res = client.get("/api/disaster/history?lat=25.5941&lon=85.1376")
    assert res.status_code == 200
    data = res.get_json()

    assert data["success"] is True
    assert "earthquake" in data["data"]
    assert "flood" in data["data"]
    assert "rainfall" in data["data"]
    eq = data["data"]["earthquake"]
    assert "events_24h" in eq
    assert "events_7d" in eq
    assert "events_30d" in eq
    assert "events_90d" in eq


def test_19_exact_model_contract_and_probability_mapping():
    """Test 19: Validates 20-feature order, RobustScaler, LabelEncoder classes, and raw fields exposure."""
    slot = registry.slots["disaster"]
    slot.ensure_loaded()
    assert slot.obj is not None, "Disaster bundle failed to load"

    bundle = slot.obj
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    encoder = bundle.get("label_encoder")

    assert model is not None
    assert scaler is not None
    assert encoder is not None

    meta = slot.metadata
    meta_features = meta.get("features", [])
    assert len(meta_features) == 20, "Contract must specify exactly 20 features"

    # Validate classes mapping
    expected_classes = [
        "air_pollution", "cold_wave", "cyclone", "drought", "earthquake",
        "extreme_temperature", "flood", "heat_wave", "heavy_rainfall",
        "high_pressure", "landslide", "low_pressure", "nominal", "storm_surge",
        "strong_winds", "thunderstorm", "tsunami", "volcano"
    ]
    assert list(encoder.classes_) == expected_classes
    assert meta.get("classes") == expected_classes

    # Dummy 20-feature vector
    dummy_input = {f: 1.0 for f in meta_features}
    res = registry.predict("disaster", dummy_input)

    assert "prediction" in res
    assert res["prediction"] in expected_classes
    assert "probabilities" in res
    assert len(res["probabilities"]) == len(expected_classes)
    for c in expected_classes:
        assert c in res["probabilities"]
        assert 0.0 <= res["probabilities"][c] <= 1.0

    # Ensure probabilities sum to ~1.0
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-3


def test_20_flood_corroboration_with_waterlogging_and_drainage():
    """Test 20: Flood prediction supported by elevated waterlogging & drainage congestion."""
    raw_res = {"prediction": "flood", "confidence": 0.72, "probabilities": {"flood": 0.72}}
    d_built_supported = {
        "features": {"river_discharge": 15.0, "rain_24h": 18.0, "max_quake_mag": 0.0, "tsunami_flag": 0},
        "seismic_metadata": {},
        "gdacs_metadata": {"gdacs_cyclone": 0, "gdacs_volcano": 0},
        "waterlogging_probability": 0.42,
        "drainage_congestion_probability": 0.45,
    }
    history = {
        "earthquake": {"events_24h": 0, "events_7d": 0, "events_30d": 0, "events_90d": 0},
        "flood": {"peak_discharge_7d": 15.0, "discharge_trend": "STEADY"},
        "rainfall": {"accumulation_7d_mm": 20.0},
        "drought": {"status": "NORMAL"}
    }
    corrob = corroborate_disaster_evidence(raw_res, d_built_supported, history)
    v = corrob["verified_hazard"]
    assert v["status"] == "SUPPORTED_HAZARD_SIGNAL"
    assert v["active"] is True
    assert v["hazard"] == "flood"
    assert "waterlogging" in v["reason"].lower() or "drainage" in v["reason"].lower() or "rainfall" in v["reason"].lower()


def test_21_disaster_history_schema_and_aliases():
    """Test 21: Disaster history endpoint provides recent_discharge and peak_discharge aliases."""
    hist = get_disaster_history(25.5941, 85.1376)
    assert hist is not None
    assert "flood" in hist
    f = hist["flood"]
    assert "recent_discharge" in f
    assert "peak_discharge" in f
    assert "recent_discharge_series" in f
    assert "peak_discharge_7d_m3s" in f
    assert isinstance(f["recent_discharge"], list)


def test_22_multi_city_pipeline_validation():
    """Test 22: Tests 8 benchmark Indian cities end-to-end through the hybrid disaster pipeline."""
    cities = [
        ("Patna", 25.5941, 85.1376),
        ("Guntur", 16.3067, 80.4365),
        ("Hyderabad", 17.3850, 78.4867),
        ("Visakhapatnam", 17.6868, 83.2185),
        ("Delhi", 28.6139, 77.2090),
        ("Mumbai", 19.0760, 72.8777),
        ("Guwahati", 26.1445, 91.7362),
        ("Srinagar", 34.0837, 74.7973),
    ]

    for name, lat, lon in cities:
        res = predict_disaster_live(lat, lon)
        assert res["success"] is True, f"Failed for {name}"
        assert "raw_model" in res, f"raw_model missing for {name}"
        assert "live_evidence" in res, f"live_evidence missing for {name}"
        assert "historical_context" in res, f"historical_context missing for {name}"
        assert "verified_hazard" in res, f"verified_hazard missing for {name}"
        assert "raw_model_prediction" in res, f"raw_model_prediction missing for {name}"
        assert "raw_model_confidence" in res, f"raw_model_confidence missing for {name}"
        assert "raw_model_probabilities" in res, f"raw_model_probabilities missing for {name}"

        # In Patna, verified hazard must NOT be an active earthquake!
        if name == "Patna":
            v = res["verified_hazard"]
            assert v.get("hazard") != "earthquake", "Patna must not report active earthquake"
            if v.get("active"):
                assert v.get("hazard") != "earthquake"


