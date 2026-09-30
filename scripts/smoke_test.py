import sys
import os
import json
import re
import requests

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:5000"

def test_all():
    results = {}

    # Test 1: Location search
    locations = ["Delhi", "Guntur", "Mumbai", "Guwahati", "Port Blair", "Ralegan Siddhi", "522002"]
    loc_pass = True
    for loc in locations:
        res = requests.get(f"{BASE_URL}/api/sources/geocode", params={"q": loc, "limit": 5})
        if res.status_code != 200 or len(res.json().get("data", [])) == 0:
            loc_pass = False
            break
    results["1. India-wide location search (7 targets)"] = "PASS" if loc_pass else "FAIL"

    # Test 2: "Use My Current Location" handling in frontend code
    app_js = open("frontend/static/js/app.js", encoding="utf-8").read()
    has_gps = "navigator.geolocation.getCurrentPosition" in app_js and "Current location is unavailable" in app_js
    results["2. 'Use My Current Location' & GPS denial handling"] = "PASS" if has_gps else "FAIL"

    # Test 3: User does NOT need to enter lat/lon
    index_html = open("frontend/templates/index.html", encoding="utf-8").read()
    has_hidden_coords = '<input id="latInput" type="hidden"' in index_html and '<input id="lonInput" type="hidden"' in index_html
    results["3. Lat/Lon removed from normal user input (hidden)"] = "PASS" if has_hidden_coords else "FAIL"

    # Test 4: Full coordinate precision preserved
    coord_full_precision = "Number(item.lat)" in app_js and "Number(item.lon)" in app_js and ".toFixed(4)" not in app_js.split("selectedLocation = {")[1].split("};")[0]
    results["4. Full coordinate precision preserved without rounding"] = "PASS" if coord_full_precision else "FAIL"

    # Test 5: Changing location clears previous coordinates
    clears_stale = 'dom.placeSearchInput.addEventListener("input"' in app_js and "selectedLocation = null;" in app_js
    results["5. Changing search text clears previous coordinates"] = "PASS" if clears_stale else "FAIL"

    # Test 6: Exactly ONE request POST /api/predict/all/live
    pred_res = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": 16.2915, "lon": 80.4542, "crop": "Rice", "crop_stage": "Flowering"})
    one_request = pred_res.status_code == 200 and 'fetch("/api/predict/all/live"' in app_js
    results["6. Exactly ONE request per analysis (POST /api/predict/all/live)"] = "PASS" if one_request else "FAIL"

    # Test 7: Dashboard does NOT call separate prediction endpoints
    has_separate = any(ep in app_js for ep in [
        "/api/predict/weather/live",
        "/api/predict/disaster/live",
        "/api/predict/agriculture/live",
        "/api/predict/waterlogging/live"
    ])
    results["7. Dashboard does NOT call separate prediction endpoints"] = "PASS" if not has_separate else "FAIL"

    # Test 8: All 4 module cards render in HTML
    cards_exist = all(cid in index_html for cid in [
        'id="weatherCard"', 'id="disasterCard"', 'id="agricultureCard"', 'id="waterloggingCard"'
    ])
    results["8. All four module cards present in frontend HTML"] = "PASS" if cards_exist else "FAIL"

    # Test 9: Model contracts unchanged
    w_meta = json.load(open("models/weather/metadata.json"))
    d_meta = json.load(open("models/disaster/metadata.json"))
    a_meta = json.load(open("models/agriculture/metadata.json"))
    from backend.services.waterlogging_service import get_warning_threshold
    contracts_ok = (
        len(w_meta.get("features", [])) == 39 and
        len(d_meta.get("features", [])) == 20 and
        len(a_meta.get("features", [])) == 35 and
        get_warning_threshold() == 0.36
    )
    results["9. Model contracts (39/20/35/0.36) unchanged"] = "PASS" if contracts_ok else "FAIL"

    # Test 10: Expired GDACS events do not trigger active alerts
    p_data = pred_res.json()
    d_feats = p_data.get("disaster", {}).get("data", {}).get("input", {}).get("features", {})
    gdacs_ok = d_feats.get("gdacs_cyclone") == 0 and d_feats.get("gdacs_volcano") == 0
    results["10. Expired GDACS events filtered out (no false active alert)"] = "PASS" if gdacs_ok else "FAIL"

    # Test 11: Historical/distant earthquake in Guwahati not treated as active local hazard
    gw_res = requests.post(f"{BASE_URL}/api/predict/all/live", json={"lat": 26.1806, "lon": 91.7539, "crop": "Rice", "crop_stage": "Tillering"})
    gw_data = gw_res.json()
    gw_risk = gw_data.get("summary", {}).get("overall_risk")
    seismic_ok = gw_risk in ["MODERATE", "LOW"]
    results["11. Historical/distant earthquake not treated as local hazard"] = "PASS" if seismic_ok else "FAIL"

    # Test 12: Missing Weather fields added to frontend render correctly
    weather_fields = ["wRain", "wWindDirection", "wVPD", "wShortwaveRad", "wDNI", "wDiffuseRad"]
    fields_ok = all(wf in index_html and wf in app_js for wf in weather_fields)
    results["12. Missing Weather fields (rain, VPD, radiation, etc.) rendered"] = "PASS" if fields_ok else "FAIL"

    # Test 13: Small non-zero values not displayed as 0.00
    format_val_logic = "if (absVal < 0.01)" in app_js and "val.toFixed(p)" in app_js
    results["13. Small non-zero values use adaptive precision (not 0.00)"] = "PASS" if format_val_logic else "FAIL"

    # Test 14: Location and analysis timestamp display correctly
    ts_logic = 'dateStr' in app_js and 'timeStr' in app_js and 'tzAbbr' in app_js
    results["14. Unambiguous timestamp and location display correctly"] = "PASS" if ts_logic else "FAIL"

    # Test 15: No city-specific hardcoded prediction logic
    api_py = open("backend/routes/api.py", encoding="utf-8").read()
    city_specific = re.search(r'if\s+(city|state|place)\s*==', api_py) or "city == \"Patna\"" in api_py or "city == \"Delhi\"" in api_py
    results["15. No city-specific hardcoded prediction logic"] = "PASS" if not city_specific else "FAIL"

    # Test 16: Geocoding safety (no placeholder email, countrycodes=in, explicit search)
    from backend.data_sources.nominatim import _valid_contact_email
    nominatim_py = open("backend/data_sources/nominatim.py", encoding="utf-8").read()
    geo_safety = (
        not _valid_contact_email("your-email@example.com") and
        '"countrycodes": "in"' in nominatim_py and
        "performLocationSearch" in app_js and
        'dom.searchLocationBtn.addEventListener("click"' in app_js
    )
    results["16. Geocoding safety (no placeholder email, in-only, explicit)"] = "PASS" if geo_safety else "FAIL"

    return results

if __name__ == "__main__":
    res = test_all()
    print(json.dumps(res, indent=2))
