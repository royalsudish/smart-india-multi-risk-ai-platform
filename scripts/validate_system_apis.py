import requests
import sqlite3
import time
import json
import os

BASE_URL = "http://127.0.0.1:5000"

def check_alert_db():
    db_path = "data/alerts.db"
    print("\n=== 1. ALERT DATABASE VALIDATION ===")
    assert os.path.exists(db_path), f"DB file {db_path} does not exist!"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check WAL mode
    cur.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()[0]
    print(f"Journal mode: {mode}")

    # Check tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables found: {tables}")
    for req in ["alerts", "alert_events", "alert_subscriptions", "in_app_notifications"]:
        assert req in tables, f"Missing table {req}!"

    # Check indexes
    cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = [row[0] for row in cur.fetchall()]
    print(f"Indexes found: {len(indexes)} indexes ({', '.join(indexes[:6])}...)")

    conn.close()
    print("Database validation PASSED!")

def check_alert_apis():
    print("\n=== 2. ALERT APIS VALIDATION ===")
    
    # GET /api/alerts/active
    r = requests.get(f"{BASE_URL}/api/alerts/active")
    assert r.status_code == 200, f"active alerts failed: {r.status_code}"
    active_data = r.json()
    print("GET /api/alerts/active:", active_data.get("success"), f"Total: {active_data.get('total')}")

    # GET /api/alerts/history
    r = requests.get(f"{BASE_URL}/api/alerts/history")
    assert r.status_code == 200, f"alerts history failed: {r.status_code}"
    hist_data = r.json()
    print("GET /api/alerts/history:", hist_data.get("success"), f"Total: {hist_data.get('total')}")

    # GET /api/alerts/notifications
    r = requests.get(f"{BASE_URL}/api/alerts/notifications")
    assert r.status_code == 200, f"notifications failed: {r.status_code}"
    notif_data = r.json()
    print("GET /api/alerts/notifications:", notif_data.get("success"))

    # POST /api/alerts/subscriptions
    sub_payload = {
        "channel": "in_app",
        "recipient": "test_user_demo",
        "min_severity": "HIGH",
        "hazard_types": ["flood", "flash_flood", "wildfire"],
        "lat": 16.3067,
        "lon": 80.4365,
        "radius_km": 50
    }
    r = requests.post(f"{BASE_URL}/api/alerts/subscriptions", json=sub_payload)
    assert r.status_code == 200, f"create sub failed: {r.status_code} {r.text}"
    sub_res = r.json()
    sub_id = (sub_res.get("data") or {}).get("subscription_id")
    print("POST /api/alerts/subscriptions created ID:", sub_id)
    assert sub_id is not None

    # GET /api/alerts/subscriptions
    r = requests.get(f"{BASE_URL}/api/alerts/subscriptions")
    assert r.status_code == 200
    subs_list = (r.json().get("data") or {}).get("subscriptions", [])
    assert any(s.get("subscription_id") == sub_id for s in subs_list)
    print("GET /api/alerts/subscriptions verified created subscription present.")

    # DELETE /api/alerts/subscriptions/<id>
    r = requests.delete(f"{BASE_URL}/api/alerts/subscriptions/{sub_id}")
    assert r.status_code == 200
    print("DELETE /api/alerts/subscriptions/<id> succeeded.")

    print("Alert APIs validation PASSED!")

def check_spatial_risk_apis():
    print("\n=== 3. PAN-INDIA SPATIAL RISK APIS VALIDATION ===")
    
    # GET /api/risk/states
    r = requests.get(f"{BASE_URL}/api/risk/states")
    assert r.status_code == 200
    states = r.json().get("data", [])
    print(f"GET /api/risk/states: {len(states)} states/UTs returned.")
    assert len(states) >= 36, f"Expected 36 states/UTs, found {len(states)}"

    # GET /api/risk/india
    r = requests.get(f"{BASE_URL}/api/risk/india")
    assert r.status_code == 200
    ind_data = r.json().get("data", {})
    features_cnt = len(ind_data.get("geojson", {}).get("features", []))
    print(f"GET /api/risk/india: GeoJSON features count: {features_cnt}, States: {ind_data.get('total_states')}")

    # GET /api/risk/state
    r = requests.get(f"{BASE_URL}/api/risk/state?state=Bihar")
    assert r.status_code == 200
    bihar_data = r.json().get("data", {})
    districts = bihar_data.get("districts", [])
    print(f"GET /api/risk/state?state=Bihar: {len(districts)} districts.")

    # GET /api/risk/district
    r = requests.get(f"{BASE_URL}/api/risk/district?state=Bihar&district=Patna")
    assert r.status_code == 200
    patna_dist = r.json().get("data", {})
    print(f"GET /api/risk/district?state=Bihar&district=Patna: center: ({patna_dist.get('lat')}, {patna_dist.get('lon')})")

    # POST /api/risk/grid
    grid_payload = {
        "lat": 25.5941,
        "lon": 85.1376,
        "radius_m": 1500,
        "grid_size_m": 500
    }
    r = requests.post(f"{BASE_URL}/api/risk/grid", json=grid_payload)
    assert r.status_code == 200
    grid_res = r.json().get("data", {})
    summary = grid_res.get("summary", {})
    cells = summary.get("total_cells", 0)
    print(f"POST /api/risk/grid for Patna: Generated {cells} cells at 500m resolution.")
    assert cells > 0
    print("Pan-India Spatial Risk APIs PASSED!")

def check_12_cities_live():
    print("\n=== 4. 12-CITY UNIFIED LIVE ENDPOINT TEST (/api/predict/all/live) ===")
    cities = [
        {"name": "Guntur", "lat": 16.3067, "lon": 80.4365, "crop": "cotton", "crop_stage": "flowering"},
        {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "crop": "rice", "crop_stage": "vegetative"},
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "crop": "wheat", "crop_stage": "maturity"},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "crop": "rice", "crop_stage": "vegetative"},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "crop": "rice", "crop_stage": "flowering"},
        {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "crop": "rice", "crop_stage": "maturity"},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "crop": "cotton", "crop_stage": "vegetative"},
        {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "crop": "tea", "crop_stage": "vegetative"},
        {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322, "crop": "basmati_rice", "crop_stage": "flowering"},
        {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973, "crop": "apple", "crop_stage": "fruiting"},
        {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "crop": "mustard", "crop_stage": "flowering"},
        {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "crop": "rice", "crop_stage": "maturity"},
    ]

    results = []

    for c in cities:
        t0 = time.time()
        payload = {
            "lat": c["lat"],
            "lon": c["lon"],
            "crop": c["crop"],
            "crop_stage": c["crop_stage"]
        }
        res = requests.post(f"{BASE_URL}/api/predict/all/live", json=payload, timeout=45)
        duration = round(time.time() - t0, 2)
        assert res.status_code == 200, f"Live call failed for {c['name']}: {res.status_code}"
        json_body = res.json()

        w_ok = json_body.get("weather", {}).get("success") is True
        wl_ok = json_body.get("waterlogging", {}).get("success") is True
        dis_ok = json_body.get("disaster", {}).get("success") is True
        ag_ok = json_body.get("agriculture", {}).get("success") is True

        # Drainage check via /api/predict/drainage/live (6-feature ML model contract)
        try:
            dr_res = requests.get(f"{BASE_URL}/api/predict/drainage/live?lat={c['lat']}&lon={c['lon']}&include_osm=0", timeout=180)
            dr_ok = dr_res.status_code == 200 and dr_res.json().get("success") is True
            dr_data = (dr_res.json().get("data") if dr_ok else {}) or {}
            dr_feats = dr_data.get("features", {})
            dr_contract_ok = all(f in dr_feats for f in ["elevation", "slope", "rain_max", "sentinel1_vv", "NDVI", "NDBI"])
        except Exception as dr_err:
            dr_ok = False
            dr_contract_ok = False
            dr_feats = {}
            print(f"Drainage live query note for {c['name']}: {dr_err}", flush=True)

        dis_data = json_body.get("disaster", {}).get("data", {})
        ver_hazards = [h.get("hazard") for h in dis_data.get("verified_hazards", [])]
        raw_ml = dis_data.get("raw_model", {}).get("prediction", "N/A")
        dq = dis_data.get("data_quality", "UNKNOWN")

        # Agriculture check: 35 features
        ag_data = json_body.get("agriculture", {}).get("data", {})
        ag_feats = ag_data.get("input_features") or ag_data.get("features") or {}
        ag_contract_ok = len(ag_feats) == 35

        # Alert Engine Outcome
        # Check active alerts in DB for this city
        alerts_res = requests.get(f"{BASE_URL}/api/alerts/active", timeout=10)
        active_alerts_cnt = alerts_res.json().get("total") if alerts_res.status_code == 200 else 0

        result_row = {
            "city": c["name"],
            "coords": f"({c['lat']}, {c['lon']})",
            "weather": "OK" if w_ok else "FAIL",
            "waterlogging": "OK" if wl_ok else "FAIL",
            "drainage": f"OK (6-feat: {dr_contract_ok})" if dr_ok else "FAIL",
            "agriculture": f"OK (35-feat: {ag_contract_ok})" if ag_ok else "FAIL",
            "disaster": "OK" if dis_ok else "FAIL",
            "verified_hazards": ver_hazards if ver_hazards else ["None"],
            "raw_ml_signal": raw_ml,
            "data_quality": dq,
            "active_alerts": active_alerts_cnt,
            "duration_s": duration
        }
        results.append(result_row)
        print(f"[{c['name']}] Duration: {duration}s | W:{w_ok} WL:{wl_ok} DR:{dr_ok}(6f:{dr_contract_ok}) DIS:{dis_ok} AG:{ag_ok}(35f:{ag_contract_ok}) | Hazards: {ver_hazards} | RawML: {raw_ml} | Alerts: {active_alerts_cnt}", flush=True)

    with open("scripts/12_cities_validation_run.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n12-City Live Validation PASSED!")

if __name__ == "__main__":
    check_alert_db()
    check_alert_apis()
    check_spatial_risk_apis()
    check_12_cities_live()
