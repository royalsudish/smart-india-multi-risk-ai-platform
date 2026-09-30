"""
Multi-City Validation Script for Disaster Intelligence Pipeline
Tests 8 target cities:
Patna, Guntur, Hyderabad, Visakhapatnam, Delhi, Mumbai, Guwahati, Srinagar
Reports:
- Raw Model Prediction & Confidence
- Live Evidence (USGS nearest dist/age, GDACS, River discharge)
- Historical Context (24h/7d/30d/90d quakes, 7d peak discharge, trend)
- Verified Operational Hazard (status, hazard, active flag, alert title, reason)
"""
import sys
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.services.disaster_service import predict_disaster_live

CITIES = [
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Guntur", "lat": 16.3067, "lon": 80.4365},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362},
    {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973},
]

def run_multi_city_validation():
    print("=" * 80)
    print("SMART INDIA MULTI-RISK PLATFORM - DISASTER PIPELINE MULTI-CITY VALIDATION")
    print("=" * 80)

    results = []

    for city in CITIES:
        name = city["name"]
        lat = city["lat"]
        lon = city["lon"]
        print(f"\nEvaluating {name} ({lat}°N, {lon}°E)...")

        res = predict_disaster_live(lat, lon)
        raw = res.get("raw_model", {})
        live = res.get("live_evidence", {})
        hist = res.get("historical_context", {})
        verified = res.get("verified_hazard", {})

        eq_hist = hist.get("earthquake", {})
        f_hist = hist.get("flood", {})
        r_hist = hist.get("rainfall", {})

        city_report = {
            "city": name,
            "coordinates": {"lat": lat, "lon": lon},
            "raw_model": {
                "prediction": raw.get("prediction"),
                "confidence": raw.get("confidence"),
                "top_probabilities": dict(sorted(raw.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:3])
            },
            "live_evidence": {
                "nearest_quake_km": live.get("nearest_quake_km"),
                "nearest_quake_age_days": live.get("nearest_quake_age_days"),
                "max_quake_mag": live.get("max_quake_mag"),
                "river_discharge": live.get("river_discharge"),
                "gdacs_cyclone": live.get("gdacs_cyclone"),
                "gdacs_volcano": live.get("gdacs_volcano"),
                "tsunami_flag": live.get("tsunami_flag"),
            },
            "historical_context": {
                "quakes_24h": eq_hist.get("events_24h"),
                "quakes_7d": eq_hist.get("events_7d"),
                "quakes_30d": eq_hist.get("events_30d"),
                "quakes_90d": eq_hist.get("events_90d"),
                "peak_discharge_7d": f_hist.get("peak_discharge_7d"),
                "discharge_trend": f_hist.get("discharge_trend"),
                "rain_7d_mm": r_hist.get("accumulation_7d_mm"),
            },
            "verified_hazard": {
                "status": verified.get("status"),
                "hazard": verified.get("hazard"),
                "active": verified.get("active"),
                "severity": verified.get("severity"),
                "alert_title": verified.get("alert_title"),
                "reason": verified.get("reason"),
            }
        }
        results.append(city_report)

        print(f"  [RAW ML]       Prediction: {raw.get('prediction')} ({raw.get('confidence', 0):.1%})")
        print(f"  [LIVE]         Nearest Quake: {live.get('nearest_quake_km')} km ({live.get('nearest_quake_age_days')}d old, M{live.get('max_quake_mag')}) | Discharge: {live.get('river_discharge')} m3/s")
        print(f"  [HISTORY]      Quakes: 24h={eq_hist.get('events_24h')}, 7d={eq_hist.get('events_7d')}, 30d={eq_hist.get('events_30d')}, 90d={eq_hist.get('events_90d')} | 7d Peak Flow: {f_hist.get('peak_discharge_7d')} m3/s ({f_hist.get('discharge_trend')})")
        print(f"  [VERIFIED]     Status: {verified.get('status')} | Hazard: {verified.get('hazard')} | Active: {verified.get('active')} | Severity: {verified.get('severity')}")
        print(f"  [ALERT TITLE]  \"{verified.get('alert_title')}\"")
        print(f"  [REASON]       {verified.get('reason')}")

    out_file = os.path.join(BASE_DIR, "scripts", "multi_city_validation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved multi-city validation results to {out_file}")

if __name__ == "__main__":
    run_multi_city_validation()
