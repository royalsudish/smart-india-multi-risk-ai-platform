import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from backend.services.disaster_service import predict_disaster_live

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

results = {}
for name, lat, lon in cities:
    res = predict_disaster_live(lat, lon)
    results[name] = {
        "city": name,
        "lat": lat,
        "lon": lon,
        "raw_model_prediction": res.get("raw_model_prediction"),
        "raw_model_confidence": res.get("raw_model_confidence"),
        "live_evidence": res.get("live_evidence"),
        "historical_context": res.get("historical_context"),
        "verified_hazard": res.get("verified_hazard"),
    }

print("=== MULTI-CITY VALIDATION REPORT ===")
for name, r in results.items():
    print(f"\n==================================================")
    print(f"CITY: {name} (Lat: {r['lat']}, Lon: {r['lon']})")
    print(f"==================================================")
    print(f"RAW MODEL:")
    print(f"  Prediction: {r['raw_model_prediction']}")
    print(f"  Confidence: {r['raw_model_confidence']:.1%}")
    v = r["verified_hazard"]
    print(f"VERIFIED HAZARD:")
    print(f"  Status:     {v['status']}")
    print(f"  Active:     {v['active']}")
    print(f"  Hazard:     {v['hazard']}")
    print(f"  Severity:   {v['severity']}")
    print(f"  Alert:      {v.get('alert_title')}")
    print(f"  Reason:     {v['reason']}")
    le = r["live_evidence"]
    print(f"LIVE EVIDENCE:")
    print(f"  Seismic Status:   {le.get('seismic_status')}")
    print(f"  Nearest Quake:    {le.get('nearest_quake_km')} km (Age: {le.get('nearest_quake_age_days')} days)")
    print(f"  River Discharge:  {le.get('river_discharge_m3s')} m3/s")
    print(f"  Rainfall 24h:     {le.get('rain_24h_mm')} mm")
    print(f"  Active Cyclone:   {le.get('active_gdacs_cyclone')}")
    print(f"  Active Volcano:   {le.get('active_gdacs_volcano')}")
    hc = r["historical_context"]
    print(f"HISTORICAL CONTEXT:")
    print(f"  Quake Counts:     24h={hc.get('quake_count_24h')}, 7d={hc.get('quake_count_7d')}, 30d={hc.get('quake_count_30d')}, 90d={hc.get('quake_count_90d')}")
    print(f"  Largest Quake 90d: M{hc.get('largest_quake_mag_90d')}")
    print(f"  7d Peak Discharge: {hc.get('peak_discharge_7d_m3s')} m3/s (Trend: {hc.get('discharge_trend')})")
    print(f"  7d Rain:          {hc.get('recent_rain_7d_mm')} mm")
    print(f"  Drought Indicated:{hc.get('is_drought_indicated')}")
