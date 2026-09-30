import sys
from pathlib import Path
import json

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app

app = create_app()
client = app.test_client()

REGIONAL_LOCATIONS = [
    {"region": "North", "name": "Patna (Bihar)", "lat": 25.5941, "lon": 85.1376, "crop": "Rice", "crop_stage": "Flowering"},
    {"region": "North", "name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "crop": "Wheat", "crop_stage": "Vegetative"},
    {"region": "South", "name": "Guntur (Andhra Pradesh)", "lat": 16.3067, "lon": 80.4365, "crop": "Cotton", "crop_stage": "Boll_Formation"},
    {"region": "East", "name": "Kolkata (West Bengal)", "lat": 22.5726, "lon": 88.3639, "crop": "Rice", "crop_stage": "Transplanting"},
    {"region": "West", "name": "Pune (Maharashtra)", "lat": 18.5204, "lon": 73.8567, "crop": "Soybean", "crop_stage": "Pod_Formation"},
    {"region": "Central India", "name": "Bhopal (Madhya Pradesh)", "lat": 23.2599, "lon": 77.4126, "crop": "Soybean", "crop_stage": "Flowering"},
    {"region": "Northeast", "name": "Guwahati (Assam)", "lat": 26.1445, "lon": 91.7362, "crop": "Rice", "crop_stage": "Grain_Filling"},
    {"region": "Island Territory", "name": "Port Blair (Andaman & Nicobar)", "lat": 11.6234, "lon": 92.7265, "crop": "Rice", "crop_stage": "Vegetative"},
    {"region": "Rural Location", "name": "Yavatmal (Vidarbha, Maharashtra)", "lat": 20.3888, "lon": 78.1204, "crop": "Cotton", "crop_stage": "Square_Formation"},
]

print("=" * 100)
print("COMPREHENSIVE INDIA-WIDE OVERALL-RISK SYNTHESIS REGRESSION AUDIT")
print("=" * 100)

audit_records = []

for loc in REGIONAL_LOCATIONS:
    payload = {
        "lat": loc["lat"],
        "lon": loc["lon"],
        "crop": loc["crop"],
        "crop_stage": loc["crop_stage"],
    }
    resp = client.post("/api/predict/all/live", json=payload)
    assert resp.status_code == 200, f"HTTP error {resp.status_code} for {loc['name']}"
    data = resp.get_json()
    assert data.get("success") is True

    summary = data.get("summary", {})
    disaster_res = data.get("disaster", {})
    d_data = disaster_res.get("data", {})
    d_input = d_data.get("input", {})
    d_feat = d_input.get("features", {})
    d_res = d_data.get("result", {})
    seismic_meta = d_input.get("seismic_metadata", {})

    d_class = d_res.get("prediction")
    d_conf = d_res.get("confidence")

    record = {
        "region": loc["region"],
        "name": loc["name"],
        "coordinates": f"{loc['lat']:.4f}N, {loc['lon']:.4f}E",
        "crop_config": f"{loc['crop']} ({loc['crop_stage']})",
        "usgs_selected_event": seismic_meta.get("event_title") or "None within 500km/30d",
        "magnitude": seismic_meta.get("max_quake_mag"),
        "distance_km": seismic_meta.get("event_distance_km"),
        "age_days": seismic_meta.get("event_age_days"),
        "seismic_evidence_category": summary.get("disaster_highlights", {}).get("seismic_evidence_category"),
        "seismic_evidence_reason": summary.get("disaster_highlights", {}).get("seismic_evidence_reason"),
        "disaster_predicted_class": d_class,
        "disaster_confidence": round(d_conf, 4) if d_conf is not None else None,
        "live_hazard_signals": {
            "river_discharge_m3s": d_feat.get("river_discharge"),
            "gdacs_cyclone": d_feat.get("gdacs_cyclone"),
            "gdacs_volcano": d_feat.get("gdacs_volcano"),
            "tsunami_flag": d_feat.get("tsunami_flag"),
        },
        "overall_risk": summary.get("overall_risk"),
        "rule_branch_fired": summary.get("rule_branch"),
        "trigger_reason": summary.get("trigger_reason"),
        "overall_advisory": summary.get("overall_advisory"),
    }
    audit_records.append(record)

    print(f"\n[{loc['region']}] {loc['name']} ({loc['lat']:.4f}N, {loc['lon']:.4f}E)")
    print(f"  USGS Event:        {record['usgs_selected_event']}")
    print(f"  Event Magnitude:   {record['magnitude']}, Distance: {record['distance_km']} km, Age: {record['age_days']} days")
    print(f"  Seismic Category:  {record['seismic_evidence_category']}")
    print(f"  Disaster Model:    Class: {d_class} (Confidence: {round(d_conf, 4) if d_conf else 'N/A'})")
    print(f"  Live Hazards:      Discharge: {d_feat.get('river_discharge')} m3/s, Cyclone: {d_feat.get('gdacs_cyclone')}, Volcano: {d_feat.get('gdacs_volcano')}, Tsunami: {d_feat.get('tsunami_flag')}")
    print(f"  -> OVERALL RISK:   {record['overall_risk']}")
    print(f"  -> RULE BRANCH:    {record['rule_branch_fired']}")
    print(f"  -> TRIGGER REASON: {record['trigger_reason']}")
    print(f"  -> ADVISORY:       {record['overall_advisory']}")

with open("C:/Users/ksudi/.gemini/antigravity-ide/brain/75e0f7c0-8dae-44b0-a9dd-6ffce6d1dd2a/scratch/india_wide_audit_report.json", "w") as f:
    json.dump(audit_records, f, indent=2)

print("\n" + "=" * 100)
print("AUDIT COMPLETED WITH 100% COVERAGE ACROSS ALL 9 INDIAN REGIONS")
print("=" * 100)
