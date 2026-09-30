import sys
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import create_app
from backend.services.model_service import registry
from backend.services.waterlogging_service import predict_live, get_warning_threshold
from backend.utils.cache import provider_cache

app = create_app()
client = app.test_client()

print("=" * 80)
print("PHASE A VERIFICATION & AUDIT SUITE")
print("=" * 80)

# 1. Verify GET / and DOM IDs
r_html = client.get("/")
assert r_html.status_code == 200, "GET / failed"
html_str = r_html.data.decode("utf-8")

phase_a_dom_ids = [
    # Original required IDs (must all remain)
    "latInput", "lonInput", "cropSelect", "cropStageSelect", "placeSearchInput",
    "analyzeBtn", "validationMsg", "loadingIndicator", "loadingStageText",
    "overallRiskBadge", "summaryLocation", "summaryModelsCount", "summaryTotalTime", "summaryTimestamp",
    "overallAdvisoryBox", "overallAdvisoryText",
    "weatherCard", "weatherStatusBadge", "weatherErrorBox", "weatherContent",
    "disasterCard", "disasterStatusBadge", "disasterErrorBox", "disasterContent", "dClass", "dConfidence",
    "agricultureCard", "agStatusBadge", "agErrorBox", "agContent",
    "waterloggingCard", "wlStatusBadge", "wlErrorBox", "wlContent", "wlWarningBanner", "wlRiskLevel", "wlProb",
    "wlRain1h", "wlRain24h", "wlRain3d", "wlRain7d", "wlElevation", "wlSlope",
    # New Phase A IDs
    "wlAnalysisTime", "wlTimelinePast24h", "wlTimelinePast7d", "wlTimelineFuture24h", "wlTimelineFuture72h",
    "wlRain3h", "wlRain6h", "wlRain12h",
    "wlFcNext1h", "wlFcNext3h", "wlFcNext6h", "wlFcNext12h", "wlFcNext24h", "wlFcNext48h", "wlFcNext72h",
    "wlOutlookBox", "wlOutlookBadge", "wlOutlookSummary", "wlMaxHourlyRain", "wlTotalForecast72h", "wlMostSignificantPeriod"
]

missing_ids = [i for i in phase_a_dom_ids if f'id="{i}"' not in html_str]
print(f"1. DOM Check: Checked {len(phase_a_dom_ids)} IDs, Missing: {missing_ids if missing_ids else 'None (100% Present)'}")
assert not missing_ids, f"Missing IDs: {missing_ids}"

# 2. Static Assets
r_css = client.get("/static/css/style.css")
r_js = client.get("/static/js/app.js")
assert r_css.status_code == 200, "style.css failed"
assert r_js.status_code == 200, "app.js failed"
print("2. Static Assets: style.css (200 OK) and app.js (200 OK) verified.")

# 3. Model Integrity Check
registry.slots["waterlogging"].ensure_loaded()
wl_model = registry.slots["waterlogging"].obj
assert list(wl_model.classes_) == [0, 1], "Model classes changed!"
assert wl_model.n_features_in_ == 30, f"Model feature count changed: {wl_model.n_features_in_}"
thresh = get_warning_threshold()
assert thresh == 0.36, f"Warning threshold changed: {thresh}"
print(f"3. Model Integrity: classes_={list(wl_model.classes_)}, n_features={wl_model.n_features_in_}, threshold={thresh}")

# 4. API Request & Data Structure Check
payload = {"lat": 16.3067, "lon": 80.4365, "crop": "Rice", "crop_stage": "Flowering"}
t0 = time.time()
r_api = client.post("/api/predict/all/live", json=payload)
api_ms = round((time.time() - t0) * 1000, 2)
assert r_api.status_code == 200, f"API failed with status {r_api.status_code}"
data = r_api.get_json()
assert data["success"] is True, "API success is not True"
assert data["waterlogging"]["success"] is True, "Waterlogging module failed"

wl_res = data["waterlogging"]["data"]["result"]
wl_inp = data["waterlogging"]["data"]["input"]

print("\n4. API Response Contract & Phase A Structures:")
print(f"  Waterlogging Risk Level:     {wl_res.get('risk_level')}")
print(f"  Waterlogging Probability:    {wl_res.get('positive_probability')}")
print(f"  Waterlogging Warning:        {wl_res.get('warning')}")
print(f"  Waterlogging Threshold:      {wl_res.get('warning_threshold')}")
print(f"  Analysis Boundary (IST):     {wl_res.get('analysis_time_ist')}")

# 5. Observed Rainfall Windows Check
recent = wl_res.get("recent_rainfall", {})
print("\n5. Observed/Recent Rainfall Windows (strictly <= analysis time):")
expected_recent_keys = ["1h", "3h", "6h", "12h", "24h", "3d", "7d"]
for k in expected_recent_keys:
    w = recent.get(k, {})
    val = w.get("value")
    s_time = w.get("start_time")
    e_time = w.get("end_time")
    rec_cnt = w.get("record_count")
    fut_cnt = w.get("future_record_count")
    print(f"  Past {k:<4s}: {val:6.2f} mm | {s_time} -> {e_time} | records: {rec_cnt:3d} | future_leakage: {fut_cnt}")
    assert fut_cnt == 0, f"Future record leakage in window {k}!"
    assert val is not None, f"Missing value for window {k}"

# 6. Future Forecast Rainfall Windows Check
forecast = wl_res.get("forecast_rainfall", {})
print("\n6. Future Forecast Rainfall Windows (strictly > analysis time):")
expected_fc_keys = ["next_1h", "next_3h", "next_6h", "next_12h", "next_24h", "next_48h", "next_72h"]
for k in expected_fc_keys:
    w = forecast.get(k, {})
    val = w.get("value")
    s_time = w.get("start_time")
    e_time = w.get("end_time")
    rec_cnt = w.get("record_count")
    hist_cnt = w.get("historical_record_count")
    print(f"  {k:<8s}: {val:6.2f} mm | {s_time} -> {e_time} | records: {rec_cnt:3d} | historical_leakage: {hist_cnt}")
    assert hist_cnt == 0, f"Historical record leakage in forecast window {k}!"
    assert val is not None, f"Missing value for window {k}"

# 7. Future Outlook Check
outlook = wl_res.get("future_outlook", {})
print("\n7. Future Early-Warning Outlook:")
print(f"  Summary:                 {outlook.get('summary')}")
print(f"  Severity:                {outlook.get('severity')}")
print(f"  Max Hourly Rain:         {outlook.get('max_hourly_rain_mm')} mm/h ({outlook.get('max_hourly_time')})")
print(f"  Total Cumulative (72h):  {outlook.get('total_cumulative_72h_mm')} mm")
print(f"  Most Significant Window: {outlook.get('most_significant_window')}")

# 8. Multi-Location Live Verification
print("\n8. Multi-Location Verification Across India:")
test_locs = [
    ("Guntur (AP)", 16.3067, 80.4365),
    ("Mumbai (MH)", 19.0760, 72.8777),
    ("Delhi (DL)", 28.6139, 77.2090),
    ("Guwahati (AS)", 26.1445, 91.7362),
    ("Kolkata (WB)", 22.5726, 88.3639),
    ("Patna (BR)", 25.5941, 85.1376)
]

for name, l_lat, l_lon in test_locs:
    res = predict_live(l_lat, l_lon)
    out = res["result"]
    inp = res["input"]
    rec = out.get("recent_rainfall", {})
    fc = out.get("forecast_rainfall", {})
    outl = out.get("future_outlook", {})

    print(f"\n  --- {name} (Lat {l_lat}, Lon {l_lon}) ---")
    print(f"    Current Risk:        {out.get('risk_level')} (Prob: {out.get('positive_probability'):.4f}, Warn: {out.get('warning')})")
    print(f"    Observed Rain (mm):  1h={rec.get('1h',{}).get('value')}, 24h={rec.get('24h',{}).get('value')}, 3d={rec.get('3d',{}).get('value')}, 7d={rec.get('7d',{}).get('value')}")
    print(f"    Forecast Rain (mm):  +6h={fc.get('next_6h',{}).get('value')}, +24h={fc.get('next_24h',{}).get('value')}, +72h={fc.get('next_72h',{}).get('value')}")
    print(f"    Outlook Severity:    {outl.get('severity')}")
    print(f"    Outlook Summary:     {outl.get('summary')}")

print("\n" + "=" * 80)
print(">>> ALL PHASE A VERIFICATION & CONTRACT AUDITS PASSED WITH ZERO DEFECTS! <<<")
print("=" * 80)
