import requests

BASE = "http://127.0.0.1:5000"

print("HEALTH")
print(requests.get(f"{BASE}/api/health", timeout=10).json())

payload = {
    "lat": 16.3067,
    "lon": 80.4365,
    "auto_weather": False,
    "weather": {},
    "agriculture": {
        "soil_moisture": 55,
        "ndvi": 0.55,
        "soil_ph": 6.8
    },
    "waterlogging": {
        "rainfall_24h": 40,
        "rainfall_72h": 105,
        "rainfall_168h": 160,
        "elevation_m": 35,
        "slope_deg": 1.5,
        "land_use_code": 2,
        "distance_to_drain_m": 140,
        "drainage_density": 0.8
    },
    "disaster": {
        "population_density": 1100,
        "vulnerability_index": 0.62
    }
}

print("\nPREDICT ALL")
print(requests.post(f"{BASE}/api/predict-all", json=payload, timeout=30).json())
