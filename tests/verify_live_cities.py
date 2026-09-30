import requests
import json

cities = [
    ("Patna", 25.5941, 85.1376, "Bihar"),
    ("Guntur", 16.3067, 80.4365, "Andhra Pradesh"),
    ("Mumbai", 19.0760, 72.8777, "Maharashtra"),
    ("Delhi", 28.6139, 77.2090, "Delhi"),
    ("Guwahati", 26.1445, 91.7362, "Assam"),
    ("Srinagar", 34.0837, 74.7973, "Jammu and Kashmir"),
    ("Jaipur", 26.9124, 75.7873, "Rajasthan"),
    ("Visakhapatnam", 17.6868, 83.2185, "Andhra Pradesh"),
]

print("=" * 100)
print(f"{'CITY':14} | {'RAW ML PREDICTION':26} | {'VERIFIED OPERATIONAL HAZARD':38} | {'DATA QUALITY':12}")
print("=" * 100)

for name, lat, lon, state in cities:
    r = requests.post(
        "http://127.0.0.1:5000/api/predict/disaster/live",
        json={"lat": lat, "lon": lon, "state": state, "district": name},
        timeout=25
    )
    assert r.status_code == 200, f"{name} failed: {r.status_code}"
    data = r.json().get("data", r.json())
    raw = data.get("raw_model", {})
    v = data.get("verified_hazard", {})
    q = data.get("model_feature_quality")
    raw_str = f"{raw.get('prediction')} ({raw.get('confidence', 0):.1%})"
    v_haz = v.get("hazard") or "None"
    v_str = f"{v_haz} [{v.get('status')}] (act={v.get('active')})"
    print(f"{name:14} | {raw_str:26} | {v_str:38} | {q:12}")

print("=" * 100)
