import json
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.disaster_service import predict_disaster_live

CITIES = [
    {"name": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376},
    {"name": "Guntur", "state": "Andhra Pradesh", "lat": 16.3067, "lon": 80.4365},
    {"name": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kochi", "state": "Kerala", "lat": 9.9312, "lon": 76.2673},
    {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362},
    {"name": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    {"name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    {"name": "Srinagar", "state": "Jammu and Kashmir", "lat": 34.0837, "lon": 74.7973},
    {"name": "Dehradun", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185},
]

def main():
    results = {}
    print(f"=== TESTING 12 PAN-INDIA CITIES LIVE DISASTER OPERATIONAL ENGINE ===")
    for c in CITIES:
        print(f"\n--- Testing {c['name']} ({c['state']}) [{c['lat']}, {c['lon']}] ---")
        try:
            res = predict_disaster_live(c['lat'], c['lon'], state=c['state'], district=c['name'])
            raw = res.get("raw_model", {})
            v_hazards = res.get("verified_hazards", [])
            data_q = res.get("data_quality", {})
            
            top_hazard_str = " · ".join([f"{h['hazard'].replace('_', ' ').title()} {h['severity']}" for h in v_hazards]) if v_hazards else "No Active Disaster Threat"
            
            results[c['name']] = {
                "city": c['name'],
                "state": c['state'],
                "coordinates": [c['lat'], c['lon']],
                "verified_hazards": v_hazards,
                "operational_alert": top_hazard_str,
                "raw_model": {
                    "prediction": raw.get("prediction"),
                    "confidence": raw.get("confidence"),
                },
                "data_quality": data_q.get("status"),
                "missing_providers": data_q.get("missing_providers", []),
                "historical_context": {
                    "earthquake_24h": res.get("historical_context", {}).get("earthquake", {}).get("events_24h", 0),
                    "earthquake_30d": res.get("historical_context", {}).get("earthquake", {}).get("events_30d", 0),
                    "rainfall_7d_mm": res.get("historical_context", {}).get("rainfall", {}).get("rain_7d_mm", 0.0),
                    "drought_deficit_pct": res.get("historical_context", {}).get("drought", {}).get("rainfall_deficit_pct", 0.0),
                }
            }
            print(f"  Operational Alert: {top_hazard_str}")
            print(f"  Verified Hazards Count: {len(v_hazards)}")
            print(f"  Raw ML Signal: {raw.get('prediction')} ({raw.get('confidence'):.1%})")
            print(f"  Data Quality: {data_q.get('status')} (Quality Score: {data_q.get('quality_score', 1.0):.2f})")
        except Exception as e:
            print(f"  ERROR for {c['name']}: {e}")
            results[c['name']] = {"error": str(e)}

    # Save to json file
    out_path = os.path.join(os.path.dirname(__file__), "12_cities_disaster_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

if __name__ == "__main__":
    main()
