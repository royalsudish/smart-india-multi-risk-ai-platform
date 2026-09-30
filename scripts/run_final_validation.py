import sys
import os
import time
import json
import requests

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:5000"

LOCATIONS = [
    {"region": "North India", "query": "Delhi", "crop": "Wheat", "stage": "Vegetative"},
    {"region": "South India", "query": "Bengaluru", "crop": "Rice", "stage": "Flowering"},
    {"region": "South India", "query": "Guntur", "crop": "Cotton", "stage": "Boll_Formation"},
    {"region": "East India", "query": "Patna", "crop": "Rice", "stage": "Flowering"},
    {"region": "East India", "query": "Kolkata", "crop": "Rice", "stage": "Transplanting"},
    {"region": "West India", "query": "Mumbai", "crop": "Cotton", "stage": "Square_Formation"},
    {"region": "Central India", "query": "Bhopal", "crop": "Soybean", "stage": "Pod_Formation"},
    {"region": "Northeast India", "query": "Guwahati", "crop": "Rice", "stage": "Tillering"},
    {"region": "Island Territory", "query": "Port Blair", "crop": "Rice", "stage": "Vegetative"},
    {"region": "Rural India", "query": "Ralegan Siddhi", "crop": "Wheat", "stage": "Vegetative"},
    {"region": "PIN-Code Test", "query": "522002", "crop": "Rice", "stage": "Flowering"},
]

def run_all():
    print("=" * 100)
    print("FINAL INDIA-WIDE VALIDATION SUITE")
    print("=" * 100)

    location_results = []

    # 1. Location tests
    for loc in LOCATIONS:
        reg = loc["region"]
        q = loc["query"]
        crop = loc["crop"]
        stage = loc["stage"]

        # Geocode
        t0 = time.time()
        g_res = requests.get(f"{BASE_URL}/api/sources/geocode", params={"q": q, "limit": 5})
        t_geo = (time.time() - t0) * 1000
        g_data = g_res.json()
        item = g_data.get("data", [])[0]
        exact_lat = float(item["lat"])
        exact_lon = float(item["lon"])
        addr = item.get("address", {})
        state = addr.get("state", "")
        dist = addr.get("state_district") or addr.get("county") or addr.get("district") or addr.get("city", "")

        # Prediction request
        payload = {
            "lat": exact_lat,
            "lon": exact_lon,
            "crop": crop,
            "crop_stage": stage
        }
        t0_p = time.time()
        p_res = requests.post(f"{BASE_URL}/api/predict/all/live", json=payload)
        t_pred = (time.time() - t0_p) * 1000
        pdata = p_res.json()

        w = pdata.get("weather", {})
        d = pdata.get("disaster", {})
        a = pdata.get("agriculture", {})
        wl = pdata.get("waterlogging", {})
        sumry = pdata.get("summary", {})

        # Disaster details
        d_data = d.get("data", {})
        d_input = d_data.get("input", {})
        d_feats = d_input.get("features", {})
        seismic = d_input.get("seismic_metadata", {})
        d_result = d_data.get("result", {})

        # Agriculture details
        a_data = a.get("data", {})
        a_result = a_data.get("result", {})
        a_preds = a_result.get("prediction", {})
        a_probs = a_result.get("probabilities", {})

        # Waterlogging details
        wl_data = wl.get("data", {})
        wl_result = wl_data.get("result", {})
        wl_input = wl_data.get("input", {})
        wl_rain = wl_input.get("rainfall_summary_mm", {})
        wl_terrain = wl_input.get("terrain", {})

        entry = {
            "region": reg,
            "query": q,
            "display_name": item.get("display_name", ""),
            "state": state,
            "district": dist,
            "lat": exact_lat,
            "lon": exact_lon,
            "weather_success": w.get("success", False),
            "disaster_success": d.get("success", False),
            "agriculture_success": a.get("success", False),
            "waterlogging_success": wl.get("success", False),
            "disaster_class": d_result.get("predicted_class"),
            "disaster_confidence": d_result.get("confidence"),
            "river_discharge": d_feats.get("river_discharge_m3_s"),
            "max_quake_mag": d_feats.get("max_quake_mag"),
            "quake_depth": d_feats.get("quake_depth_km"),
            "tsunami_flag": d_feats.get("tsunami_flag"),
            "gdacs_cyclone": d_feats.get("gdacs_cyclone"),
            "gdacs_volcano": d_feats.get("gdacs_volcano"),
            "seismic_category": seismic.get("category", "NO_RELEVANT_EVENT"),
            "event_distance_km": seismic.get("distance_km"),
            "event_age_days": seismic.get("age_days"),
            "event_magnitude": seismic.get("magnitude"),
            "waterlogging_risk": wl_result.get("risk_level", "N/A"),
            "waterlogging_prob": wl_result.get("positive_probability"),
            "waterlogging_warning": wl_result.get("warning"),
            "waterlogging_threshold": wl_result.get("warning_threshold"),
            "rainfall_1h": wl_rain.get("rainfall_1h"),
            "rainfall_24h": wl_rain.get("rainfall_24h"),
            "rainfall_3d": wl_rain.get("rainfall_3d"),
            "rainfall_7d": wl_rain.get("rainfall_7d"),
            "elevation": wl_terrain.get("elevation_m"),
            "slope": wl_terrain.get("slope_deg"),
            "agriculture_risk": a_preds.get("crop_weather_risk", "N/A"),
            "irrigation": a_preds.get("irrigation", "N/A"),
            "heat_stress": a_preds.get("heat_stress", "N/A"),
            "agri_waterlogging": a_preds.get("waterlogging_risk", "N/A"),
            "spraying_suitability": a_preds.get("spraying_suitability", "N/A"),
            "overall_risk": sumry.get("overall_risk", "N/A"),
            "trigger_reason": sumry.get("trigger_reason", "N/A"),
            "rule_branch": sumry.get("rule_branch", "N/A"),
            "total_time_ms": pdata.get("total_execution_time_ms", t_pred),
            "weather_features": (w.get("data", {}).get("input", {}).get("features") or {}),
            "agriculture_predictions": a_preds,
            "agriculture_probabilities": a_probs,
        }
        location_results.append(entry)

    # 2. Cold Cache vs Warm Cache Test (using fresh coordinates)
    print("\n--- Running Cold vs Warm Cache Test ---")
    test_lat = 15.8281
    test_lon = 78.0373

    # Cold run (first request to this location)
    t0_c = time.time()
    res_cold = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": test_lat, "lon": test_lon, "crop": "Cotton", "crop_stage": "Boll_Formation"})
    cold_data = res_cold.json()

    # Warm run (immediate second request to the exact same coordinates)
    t0_w = time.time()
    res_warm = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": test_lat, "lon": test_lon, "crop": "Cotton", "crop_stage": "Boll_Formation"})
    warm_data = res_warm.json()

    cache_comparison = {
        "cold": {
            "total_execution_time_ms": cold_data.get("total_execution_time_ms"),
            "provider_execution_times": cold_data.get("provider_execution_times", {}),
            "provider_cache_status": cold_data.get("provider_cache_status", {}),
        },
        "warm": {
            "total_execution_time_ms": warm_data.get("total_execution_time_ms"),
            "provider_execution_times": warm_data.get("provider_execution_times", {}),
            "provider_cache_status": warm_data.get("provider_cache_status", {}),
        }
    }

    # 3. Failure tests
    print("\n--- Running Search Failure Tests ---")
    fail_res_empty = requests.get(f"{BASE_URL}/api/sources/geocode?q=")
    fail_res_unknown = requests.get(f"{BASE_URL}/api/sources/geocode?q=xyznonexistentplace9876543210")
    
    failure_tests = {
        "empty_query": {
            "status_code": fail_res_empty.status_code,
            "response": fail_res_empty.json()
        },
        "unknown_place": {
            "status_code": fail_res_unknown.status_code,
            "data_count": len(fail_res_unknown.json().get("data", [])),
            "response": fail_res_unknown.json()
        }
    }

    # Save full report
    final_output = {
        "locations": location_results,
        "cache_comparison": cache_comparison,
        "failure_tests": failure_tests,
    }

    with open("C:/Users/ksudi/.gemini/antigravity-ide/brain/75e0f7c0-8dae-44b0-a9dd-6ffce6d1dd2a/scratch/final_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\nValidation completed successfully! Report saved to final_validation_report.json.")

if __name__ == "__main__":
    run_all()
