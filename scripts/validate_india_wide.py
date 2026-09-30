import sys
import time
import json
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:5000"

LOCATION_TEST_CASES = [
    {"region": "North India", "query": "Delhi", "crop": "Wheat", "stage": "Vegetative"},
    {"region": "South India", "query": "Bengaluru", "crop": "Rice", "stage": "Flowering"},
    {"region": "South India", "query": "Guntur", "crop": "Cotton", "stage": "Boll_Formation"},
    {"region": "East India", "query": "Patna", "crop": "Rice", "stage": "Flowering"},
    {"region": "East India", "query": "Kolkata", "crop": "Rice", "stage": "Transplanting"},
    {"region": "West India", "query": "Mumbai", "crop": "Cotton", "stage": "Square_Formation"},
    {"region": "Central India", "query": "Bhopal", "crop": "Soybean", "stage": "Pod_Formation"},
    {"region": "Northeast India", "query": "Guwahati", "crop": "Rice", "stage": "Tillering"},
    {"region": "Island Territory", "query": "Port Blair", "crop": "Rice", "stage": "Vegetative"},
    {"region": "Rural India", "query": "Punsari, Gujarat", "crop": "Cotton", "stage": "Vegetative"},
    {"region": "PIN-Code Test", "query": "522002", "crop": "Rice", "stage": "Flowering"},
]

def run_suite():
    print("=" * 80)
    print("STARTING COMPREHENSIVE INDIA-WIDE VALIDATION SUITE")
    print("=" * 80)

    results = []

    for item in LOCATION_TEST_CASES:
        region = item["region"]
        query = item["query"]
        crop = item["crop"]
        stage = item["stage"]

        print(f"\n---> Testing: {query} ({region})")

        # 1. Geocode lookup
        t0_geo = time.time()
        geo_res = requests.get(f"{BASE_URL}/api/sources/geocode", params={"q": query, "limit": 5})
        t_geo = (time.time() - t0_geo) * 1000

        if geo_res.status_code != 200:
            print(f"FAILED: Geocode HTTP {geo_res.status_code}")
            continue

        geo_data = geo_res.json()
        places = geo_data.get("data", [])
        if not places:
            print(f"FAILED: No geocode results returned for {query}")
            continue

        first = places[0]
        exact_lat = float(first["lat"])
        exact_lon = float(first["lon"])
        display_name = first.get("display_name", "")
        address = first.get("address", {})
        country = address.get("country", "")
        state = address.get("state", "")
        district = address.get("state_district") or address.get("county") or address.get("district", "")
        locality = address.get("city") or address.get("town") or address.get("village") or address.get("suburb", "")

        print(f"   Resolved: {display_name}")
        print(f"   Coords: lat={exact_lat}, lon={exact_lon}")
        print(f"   State: {state} | District/Locality: {district or locality} | Country: {country}")

        # 2. Unified Prediction Request
        payload = {
            "lat": exact_lat,
            "lon": exact_lon,
            "crop": crop,
            "crop_stage": stage
        }

        t0_pred = time.time()
        pred_res = requests.post(f"{BASE_URL}/api/predict/all/live", json=payload)
        t_pred = (time.time() - t0_pred) * 1000

        if pred_res.status_code != 200:
            print(f"FAILED: Prediction HTTP {pred_res.status_code}: {pred_res.text}")
            continue

        pdata = pred_res.json()

        # Extract Module Status
        weather = pdata.get("weather", {})
        disaster = pdata.get("disaster", {})
        agri = pdata.get("agriculture", {})
        waterlog = pdata.get("waterlogging", {})
        summary = pdata.get("summary", {})

        w_success = weather.get("success", False)
        d_success = disaster.get("success", False)
        a_success = agri.get("success", False)
        wl_success = waterlog.get("success", False)

        # Disaster fields
        d_feats = disaster.get("features", {})
        river_discharge = d_feats.get("river_discharge_m3_s")
        max_quake_mag = d_feats.get("max_quake_mag")
        quake_depth = d_feats.get("quake_depth_km")
        tsunami_flag = d_feats.get("tsunami_flag")
        gdacs_cyclone = d_feats.get("gdacs_cyclone")
        gdacs_volcano = d_feats.get("gdacs_volcano")

        # Seismic metadata
        seismic_meta = disaster.get("seismic_metadata", {})
        seismic_cat = seismic_meta.get("category", "N/A")
        eq_dist = seismic_meta.get("distance_km")
        eq_age = seismic_meta.get("age_days")
        eq_mag = seismic_meta.get("magnitude")

        # GDACS active hazard
        gdacs_active = bool(gdacs_cyclone or gdacs_volcano)

        # Risks
        wl_risk = waterlog.get("risk_level", "N/A")
        wl_prob = waterlog.get("positive_probability", 0.0)
        wl_warn = waterlog.get("warning", False)
        ag_risk = agri.get("predictions", {}).get("crop_weather_risk", "N/A")
        overall_risk = summary.get("overall_risk", "N/A")
        trigger_reason = summary.get("trigger_reason", "N/A")
        total_time = pdata.get("total_execution_time_ms", t_pred)

        res_entry = {
            "region": region,
            "query": query,
            "display_name": display_name,
            "state": state,
            "district": district or locality,
            "country": country,
            "lat": exact_lat,
            "lon": exact_lon,
            "w_success": w_success,
            "d_success": d_success,
            "a_success": a_success,
            "wl_success": wl_success,
            "river_discharge": river_discharge,
            "max_quake_mag": max_quake_mag,
            "quake_depth": quake_depth,
            "tsunami_flag": tsunami_flag,
            "gdacs_cyclone": gdacs_cyclone,
            "gdacs_volcano": gdacs_volcano,
            "seismic_category": seismic_cat,
            "eq_dist": eq_dist,
            "eq_age": eq_age,
            "eq_mag": eq_mag,
            "gdacs_active": gdacs_active,
            "wl_risk": wl_risk,
            "wl_prob": wl_prob,
            "wl_warn": wl_warn,
            "ag_risk": ag_risk,
            "overall_risk": overall_risk,
            "trigger_reason": trigger_reason,
            "total_time_ms": total_time,
            "geocode_time_ms": t_geo
        }
        results.append(res_entry)

        print(f"   Success: W={w_success}, D={d_success}, A={a_success}, WL={wl_success}")
        print(f"   Overall Risk: {overall_risk} | Trigger: {trigger_reason}")
        print(f"   Seismic Cat: {seismic_cat} | GDACS Active: {gdacs_active} | Total ms: {total_time:.1f}")

    # Save results to scratch artifact
    with open("C:/Users/ksudi/.gemini/antigravity-ide/brain/75e0f7c0-8dae-44b0-a9dd-6ffce6d1dd2a/scratch/india_wide_full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL INDIA-WIDE LOCATION TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
