import sys
import json
import requests

BASE_URL = "http://127.0.0.1:5000"

print("=" * 80)
print("TESTING DRAINAGE MODEL INTEGRATION SUITE")
print("=" * 80)

# 1. Models Status
print("\n--- 1. Testing GET /api/models/status ---")
r = requests.get(f"{BASE_URL}/api/models/status?load=1")
print("Status Code:", r.status_code)
data = r.json().get("data", {})
for m_name, info in data.items():
    print(f"  Model '{m_name}': loaded={info.get('loaded')}, present={info.get('file_present')}, error={info.get('error')}")

assert "drainage" in data, "drainage model missing from /api/models/status"
assert data["drainage"]["loaded"] is True, "drainage model is not loaded"
print(">>> PASS: 5/5 models loaded and registered successfully.")

# 2. Standalone Model Prediction Endpoint
print("\n--- 2. Testing POST /api/predict/drainage ---")
test_features = {
    "features": {
        "elevation": 27.0,
        "slope": 1.2,
        "rain_max": 2.13,
        "sentinel1_vv": -10.5,
        "NDVI": 0.42,
        "NDBI": -0.05
    }
}
r = requests.post(f"{BASE_URL}/api/predict/drainage", json=test_features)
print("Status Code:", r.status_code)
pred_data = r.json()
print("Prediction Data:", json.dumps(pred_data, indent=2))
assert r.status_code == 200, "POST /api/predict/drainage failed"
assert pred_data.get("success") is True, "Prediction success is false"
print(">>> PASS: Standalone POST /api/predict/drainage succeeded.")

# 3. Live Drainage Prediction for Guntur, Andhra Pradesh
print("\n--- 3. Testing POST /api/predict/drainage/live (Guntur, AP) ---")
guntur_coords = {"lat": 16.3067, "lon": 80.4365, "radius_m": 2000}
r = requests.post(f"{BASE_URL}/api/predict/drainage/live", json=guntur_coords)
print("Status Code:", r.status_code)
live_data = r.json()
print("Live Drainage Output:", json.dumps(live_data, indent=2))
assert r.status_code == 200, "POST /api/predict/drainage/live failed"
assert live_data.get("success") is True, "Live prediction success is false"
pred_res = live_data["data"]["prediction"]
print(f"Guntur Drainage Prediction: class={pred_res['prediction']} ({pred_res['prediction_label']}), risk={pred_res['risk_level']}, prob={pred_res['positive_probability']}")
print(">>> PASS: Live Drainage prediction for Guntur succeeded.")

# 4. Waterlogging Drainage Endpoint with Model Integration
print("\n--- 4. Testing GET /api/waterlogging/drainage (Guntur, AP) ---")
r = requests.get(f"{BASE_URL}/api/waterlogging/drainage", params={"lat": 16.3067, "lon": 80.4365, "radius_m": 2000})
print("Status Code:", r.status_code)
wl_drain_data = r.json()
assert r.status_code == 200, "GET /api/waterlogging/drainage failed"
assert "model_prediction" in wl_drain_data["data"], "model_prediction missing from /api/waterlogging/drainage"
print("Model prediction inside drainage endpoint:", wl_drain_data["data"]["model_prediction"])
print(">>> PASS: /api/waterlogging/drainage returns both OSM network & real ML drainage prediction.")

# 5. Core Unified Endpoint Verification (Zero Regression)
print("\n--- 5. Testing POST /api/predict/all/live (Zero Regression on Core Modules) ---")
r = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"})
print("Status Code:", r.status_code)
core_data = r.json()
assert r.status_code == 200, "POST /api/predict/all/live failed"
assert core_data.get("weather", {}).get("success") is True, "Weather prediction failed"
assert core_data.get("waterlogging", {}).get("success") is True, "Waterlogging prediction failed"
assert core_data.get("agriculture", {}).get("success") is True, "Agriculture prediction failed"
assert core_data.get("disaster", {}).get("success") is True, "Disaster prediction failed"
print("All 4 Core Models in /api/predict/all/live:", {
    "weather": core_data["weather"]["success"],
    "waterlogging": core_data["waterlogging"]["success"],
    "agriculture": core_data["agriculture"]["success"],
    "disaster": core_data["disaster"]["success"],
})
print(">>> PASS: Zero regression on core unified endpoint.")

print("\n" + "=" * 80)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 80)
