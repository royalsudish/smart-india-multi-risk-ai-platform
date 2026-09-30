"""
Pan-India Validation Script across Wildfire, Avalanche, and Flash Flood zones.
Coordinates:
1. Wildfire Zones:
   - Central India forest zone (Kanha/Mandla: 22.33, 80.61)
   - Western Ghats (Wayanad: 11.68, 76.13)
   - Northeast (Karbi Anglong: 26.15, 93.35)
   - Himalayan foothills (Jim Corbett / Nainital: 29.53, 79.13)
2. Avalanche Zones:
   - Jammu & Kashmir (Kupwara/Gulmarg: 34.05, 74.38)
   - Ladakh (Khardung La / Leh: 34.28, 77.60)
   - Himachal Pradesh (Rohtang / Lahaul: 32.37, 77.25)
   - Uttarakhand (Chamoli / Joshimath: 30.55, 79.56)
3. Flash Flood Zones:
   - Mumbai (19.0760, 72.8777)
   - Chennai (13.0827, 80.2707)
   - Hyderabad (17.3850, 78.4867)
   - Guwahati (26.1445, 91.7362)
   - Dehradun (30.3165, 78.0322)
   - Kochi (9.9312, 76.2673)
4. Nominal Baseline / No-Hazard:
   - Jaipur (26.9124, 75.7873)
"""

import json
from backend.services.disaster_service import predict_disaster_live

TARGETS = [
    # Wildfire Focus
    {"name": "Central India Forest (Mandla)", "lat": 22.33, "lon": 80.61, "category": "Wildfire"},
    {"name": "Western Ghats (Wayanad)", "lat": 11.68, "lon": 76.13, "category": "Wildfire"},
    {"name": "Northeast Forest (Karbi Anglong)", "lat": 26.15, "lon": 93.35, "category": "Wildfire"},
    {"name": "Himalayan Foothills (Nainital)", "lat": 29.53, "lon": 79.13, "category": "Wildfire"},

    # Avalanche Focus
    {"name": "Jammu & Kashmir (Gulmarg/Baramulla)", "lat": 34.05, "lon": 74.38, "category": "Avalanche"},
    {"name": "Ladakh (Khardung / Leh)", "lat": 34.28, "lon": 77.60, "category": "Avalanche"},
    {"name": "Himachal Pradesh (Lahaul)", "lat": 32.37, "lon": 77.25, "category": "Avalanche"},
    {"name": "Uttarakhand (Joshimath / Chamoli)", "lat": 30.55, "lon": 79.56, "category": "Avalanche"},

    # Flash Flood Focus
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "category": "Flash Flood"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "category": "Flash Flood"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "category": "Flash Flood"},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "category": "Flash Flood"},
    {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322, "category": "Flash Flood"},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "category": "Flash Flood"},

    # Normal / Baseline
    {"name": "Jaipur (Nominal Baseline)", "lat": 26.9124, "lon": 75.7873, "category": "Baseline"},
]

results = []
print(f"Beginning Pan-India Multi-Zone Validation ({len(TARGETS)} locations)...")

for t in TARGETS:
    name = t["name"]
    lat = t["lat"]
    lon = t["lon"]
    cat = t["category"]
    try:
        res = predict_disaster_live(lat, lon)
        assert res["success"] is True, f"Inference failed for {name}"
        
        v_hazards = res.get("verified_hazards", [])
        h_status = res.get("hazard_status", {})
        dq = res.get("data_quality", {})
        raw_ml = res.get("raw_model", {})

        wf_stat = h_status.get("wildfire", {})
        av_stat = h_status.get("avalanche", {})
        ff_stat = h_status.get("flash_flood", {})

        summary = {
            "name": name,
            "category": cat,
            "lat": lat,
            "lon": lon,
            "data_quality_status": dq.get("status"),
            "raw_ml_prediction": raw_ml.get("prediction"),
            "raw_ml_confidence": raw_ml.get("confidence"),
            "verified_active_hazards": [f"{h['hazard']}:{h['severity']}" for h in v_hazards],
            "wildfire_active": wf_stat.get("active", False),
            "wildfire_severity": wf_stat.get("severity"),
            "wildfire_detections_10km": wf_stat.get("evidence", {}).get("detections_10km", 0),
            "avalanche_active": av_stat.get("active", False),
            "avalanche_severity": av_stat.get("severity"),
            "avalanche_elevation_m": av_stat.get("evidence", {}).get("elevation_m"),
            "flash_flood_active": ff_stat.get("active", False),
            "flash_flood_severity": ff_stat.get("severity"),
            "flash_flood_rain_1h": ff_stat.get("evidence", {}).get("rain_1h_mm"),
        }
        results.append(summary)
        print(f"[{cat}] {name} -> DQ: {summary['data_quality_status']} | Verified Hazards: {summary['verified_active_hazards'] or 'None'} | ML: {summary['raw_ml_prediction']}")
    except Exception as e:
        print(f"ERROR on {name}: {e}")
        results.append({"name": name, "error": str(e)})

with open("scripts/pan_india_validation_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nPan-India Multi-Zone Validation complete. Saved to scripts/pan_india_validation_summary.json")
