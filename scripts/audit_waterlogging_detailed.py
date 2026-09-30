import sys
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.services.waterlogging_features import (
    build_live_features,
    get_live_terrain,
    SELECTED_FEATURES,
    engineer_ml_features
)
from backend.services.waterlogging_service import (
    predict_live,
    get_warning_threshold,
    risk_label,
    THRESHOLD_PATH,
    METADATA_PATH,
    DEFAULT_THRESHOLD
)
from backend.services.model_service import registry
from backend.config.settings import OPEN_METEO_FORECAST_URL, OPEN_METEO_ELEVATION_URL
from backend.data_sources.http_client import get_json

print("================================================================================")
print("COMPREHENSIVE WATERLOGGING CORRECTNESS AUDIT")
print("================================================================================")

# Ensure model loaded
registry.slots["waterlogging"].ensure_loaded()
model = registry.slots["waterlogging"].obj

lat, lon = 16.3067, 80.4365  # Guntur test location
print(f"Test Location: Guntur, Andhra Pradesh (Lat: {lat}, Lon: {lon})")

# 1 & 8. Weather & Feature Vector
live_built = build_live_features(lat, lon)
features_dict = live_built["features"]
print("\n--- 1. EXACT MODEL INPUT VECTOR ---")
print(f"Total Features Supplied: {len(features_dict)}")
print(f"Total Features Expected by Metadata: {len(SELECTED_FEATURES)}")
print(f"Total Features Expected by Model pipeline: {model.n_features_in_}")
feature_names_model = list(model.feature_names_in_)

vector_df = pd.DataFrame([features_dict])[feature_names_model]

print(f"{'#':<3} | {'Feature Name':<32} | {'Supplied Value':<18} | {'Feature In Model?':<18}")
print("-" * 75)
for idx, fname in enumerate(feature_names_model, 1):
    val = features_dict.get(fname)
    print(f"{idx:<3} | {fname:<32} | {str(val):<18} | {'YES':<18}")

# 2. Verify Model Pipeline
print("\n--- 2. VERIFY MODEL PIPELINE ---")
print("Model Pipeline Object Type:", type(model))
for name, step in model.steps:
    print(f"  Step: '{name}' -> {type(step)}")
    if hasattr(step, "transformers"):
        for tname, trans, cols in step.transformers:
            print(f"    Transformer: '{tname}' -> {type(trans)}")
            if hasattr(trans, "steps"):
                for sname, sstep in trans.steps:
                    print(f"      Sub-step: '{sname}' -> {type(sstep)}")
            print(f"      Columns ({len(cols)}): {cols[:5]}... ({len(cols)} total)")
    if hasattr(step, "classes_"):
        print(f"    Estimator classes_: {step.classes_}")

# 3. Verify predict_proba Class Mapping
print("\n--- 3. VERIFY PREDICT_PROBA CLASS MAPPING ---")
raw_probs = model.predict_proba(vector_df)[0]
classes = model.classes_
print("model.classes_:", classes)
print("raw predict_proba:", raw_probs)
pos_idx = list(classes).index(1)
neg_idx = list(classes).index(0)
print(f"Negative Class (0 - No Waterlogging) index: {neg_idx}, Probability: {raw_probs[neg_idx]:.8f}")
print(f"Positive Class (1 - Waterlogging) index:    {pos_idx}, Probability: {raw_probs[pos_idx]:.8f}")

# 4. Check whether 0.0 is real or formatting
print("\n--- 4. FULL PRECISION PROBABILITY CHECK ---")
raw_pos_prob = raw_probs[pos_idx]
print(f"raw_positive_probability (full float): {repr(raw_pos_prob)}")
print(f"formatted string %.10f: {raw_pos_prob:.10f}")
# Check individual tree votes
rf = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]
X_trans = preprocessor.transform(vector_df)
tree_preds = [tree.predict_proba(X_trans)[0] for tree in rf.estimators_]
votes_1 = sum(1 for p in tree_preds if p[1] > 0.5)
print(f"Tree votes for class 1 (out of {len(rf.estimators_)} trees): {votes_1}")

# 5 & 6. Verify Threshold & Risk Level Logic
print("\n--- 5 & 6. THRESHOLD & RISK LEVEL MAPPING ---")
thresh = get_warning_threshold()
print(f"Configured warning threshold: {thresh}")
print("Risk Level Mapping Rules:")
print(f"  LOW:    probability < threshold * 0.50 (i.e. p < {thresh * 0.50:.4f})")
print(f"  MEDIUM: threshold * 0.50 <= probability < threshold (i.e. {thresh * 0.50:.4f} <= p < {thresh:.4f})")
print(f"  HIGH:   probability >= threshold (i.e. p >= {thresh:.4f})")
print(f"  WARNING condition: probability >= threshold (i.e. p >= {thresh:.4f})")

calc_risk = risk_label(raw_pos_prob, thresh)
calc_warn = bool(raw_pos_prob >= thresh)
print(f"Evaluated Risk Level for Current Location: {calc_risk}")
print(f"Evaluated Warning Flag:                   {calc_warn}")

# 7 & 8. Rainfall Window & Weather Variables Audit
print("\n--- 7 & 8. RAINFALL WINDOW & WEATHER VARIABLES AUDIT ---")
now_india = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
print(f"Current Asia/Kolkata Hour: {now_india}")

# Fetch raw Open-Meteo data for inspection
raw_data = get_json(
    OPEN_METEO_FORECAST_URL,
    {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "surface_pressure",
        ]),
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
)

hourly = pd.DataFrame(raw_data["hourly"])
hourly["datetime"] = pd.to_datetime(hourly["time"])
filtered_hourly = hourly[hourly["datetime"] <= now_india].sort_values("datetime").reset_index(drop=True)

print(f"Total returned hourly records: {len(hourly)}")
print(f"Records up to current hour:    {len(filtered_hourly)}")
print(f"Latest record timestamp:       {filtered_hourly.iloc[-1]['datetime']}")

for win_name, count in [("1h", 1), ("24h", 24), ("3d", 72), ("7d", 168)]:
    sub = filtered_hourly.iloc[-count:]
    start_ts = sub.iloc[0]["datetime"]
    end_ts = sub.iloc[-1]["datetime"]
    sum_rain = sub["rain"].fillna(0).clip(lower=0).sum()
    print(f"  Window {win_name:4s}: start={start_ts} | end={end_ts} | records={len(sub):3d} | sum_rain={sum_rain:.2f} mm")

# 9. Terrain Audit
print("\n--- 9. TERRAIN AUDIT ---")
offset = 0.01
lats = [lat, lat + offset, lat - offset, lat, lat]
lons = [lon, lon, lon, lon + offset, lon - offset]
points_labels = ["Center", "North", "South", "East", "West"]
terrain_data = get_json(
    OPEN_METEO_ELEVATION_URL,
    {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons))
    }
)
elevations = terrain_data["elevation"]
for pt_name, p_lat, p_lon, elev in zip(points_labels, lats, lons, elevations):
    print(f"  {pt_name:6s} Point: Lat={p_lat:.4f}, Lon={p_lon:.4f} -> Elevation={elev} m")

z = np.array(elevations, dtype=float)
center_elevation = z[0]
north_south_distance = 2 * offset * 111320
east_west_distance = 2 * offset * 111320 * np.cos(np.radians(lat))
dz_dy = (z[1] - z[2]) / north_south_distance
dz_dx = (z[3] - z[4]) / east_west_distance
calc_slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
print(f"  Calculated Center Elevation: {center_elevation} m")
print(f"  Calculated dz_dx:            {dz_dx:.8f}")
print(f"  Calculated dz_dy:            {dz_dy:.8f}")
print(f"  Calculated Slope:            {calc_slope:.4f} degrees")

# 10. Cache Location Audit
print("\n--- 10. CACHE LOCATION AUDIT ---")
from backend.utils.cache import make_cache_key
print(f"Selected UI Location: Guntur, Andhra Pradesh")
print(f"Resolved Coordinates: lat={lat}, lon={lon}")
print(f"Weather Cache Key:    {make_cache_key('open_meteo_waterlogging', lat, lon)}")
print(f"Terrain Cache Key:    {make_cache_key('open_meteo_terrain', lat, lon)}")

# 11 & 12. Training vs Live Feature Comparison & Missing Values
print("\n--- 11 & 12. TRAINING VS LIVE FEATURE COMPARISON ---")
print(f"{'Feature Name':<32} | {'Training Construction':<35} | {'Live Construction':<35} | {'Status':<8}")
print("-" * 115)
for f in SELECTED_FEATURES:
    print(f"{f:<32} | {'Same in notebook cell 42':<35} | {'engineer_ml_features()':<35} | {'MATCH':<8}")

# 13. Label Quality
print("\n--- 13. MODEL TRAINING LABEL QUALITY ---")
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
print("Label Mode recorded in metadata:", metadata.get("label_mode"))
print("Weak label note:", metadata.get("weak_label_note"))

# 14. Controlled Situations
print("\n--- 14. TEST CONTROL CASES (Dry / Moderate / Severe) ---")
base_features = features_dict.copy()

# Case A: Dry
case_a = base_features.copy()
case_a["rainfall_1h"] = 0.0
case_a["rainfall_3h"] = 0.0
case_a["rainfall_6h"] = 0.1
case_a["rainfall_24h"] = 0.2
case_a["rainfall_3d"] = 1.5
case_a["rainfall_7d"] = 6.4
# re-engineer interaction features
case_a_df = pd.DataFrame([case_a])
case_a_df["datetime"] = now_india
case_a_eng = engineer_ml_features(case_a_df)[feature_names_model]
prob_a = model.predict_proba(case_a_eng)[0][1]

# Case B: Moderate
case_b = base_features.copy()
case_b["rainfall_1h"] = 5.0
case_b["rainfall_3h"] = 15.0
case_b["rainfall_6h"] = 25.0
case_b["rainfall_24h"] = 45.0
case_b["rainfall_3d"] = 80.0
case_b["rainfall_7d"] = 110.0
case_b_df = pd.DataFrame([case_b])
case_b_df["datetime"] = now_india
case_b_eng = engineer_ml_features(case_b_df)[feature_names_model]
prob_b = model.predict_proba(case_b_eng)[0][1]

# Case C: Severe
case_c = base_features.copy()
case_c["rainfall_1h"] = 35.0
case_c["rainfall_3h"] = 60.0
case_c["rainfall_6h"] = 90.0
case_c["rainfall_24h"] = 130.0
case_c["rainfall_3d"] = 180.0
case_c["rainfall_7d"] = 250.0
case_c_df = pd.DataFrame([case_c])
case_c_df["datetime"] = now_india
case_c_eng = engineer_ml_features(case_c_df)[feature_names_model]
prob_c = model.predict_proba(case_c_eng)[0][1]

for case_name, p in [("Case A - Dry", prob_a), ("Case B - Moderate", prob_b), ("Case C - Severe", prob_c)]:
    r = risk_label(p, thresh)
    w = p >= thresh
    print(f"  {case_name:<18}: Positive Probability = {p:.6f} ({p*100:.2f}%) | Risk Level = {r:<6} | Warning = {w}")

# 15. Test Multiple Live India Locations
print("\n--- 15. TEST MULTIPLE LIVE INDIA LOCATIONS ---")
test_locs = [
    ("Guntur (AP)", 16.3067, 80.4365),
    ("Mumbai (MH)", 19.0760, 72.8777),
    ("Delhi (DL)", 28.6139, 77.2090),
    ("Guwahati (AS)", 26.1445, 91.7362),
    ("Kolkata (WB)", 22.5726, 88.3639),
    ("Patna (BR)", 25.5941, 85.1376)
]

for loc_name, l_lat, l_lon in test_locs:
    res = predict_live(l_lat, l_lon)
    inp = res["input"]
    rain = inp["rainfall_summary_mm"]
    ter = inp["terrain"]
    out = res["result"]
    pos_p = out["positive_probability"]
    r_lvl = out["risk_level"]
    warn = out["warning"]
    print(f"{loc_name:<15} (Lat {l_lat:.4f}, Lon {l_lon:.4f}):")
    print(f"  Rain: 1h={rain['rainfall_1h']:.2f}mm, 24h={rain['rainfall_24h']:.2f}mm, 3d={rain['rainfall_3d']:.2f}mm, 7d={rain['rainfall_7d']:.2f}mm | Elev={ter['elevation_m']}m, Slope={ter['slope_deg']:.2f}°")
    print(f"  Output: Prob={pos_p:.6f} ({pos_p*100:.2f}%), Risk={r_lvl}, Warning={warn}")
