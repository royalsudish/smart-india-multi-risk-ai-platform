import json

with open("scripts/12_cities_validation_run.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

print("| City | Coords | Weather | Waterlogging | Drainage (6f) | Ag (35f) | Disaster | Verified Hazards | Raw ML Signal | Data Quality | Duration |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    vh = ", ".join(r["verified_hazards"])
    dq = r["data_quality"].get("status") if isinstance(r["data_quality"], dict) else r["data_quality"]
    c_name = r["city"]
    coords = r["coords"]
    w = r["weather"]
    wl = r["waterlogging"]
    dr = r["drainage"]
    ag = r["agriculture"]
    dis = r["disaster"]
    raw_ml = r["raw_ml_signal"]
    dur = f"{r['duration_s']}s"
    print(f"| {c_name} | {coords} | {w} | {wl} | {dr} | {ag} | {dis} | {vh} | {raw_ml} | {dq} | {dur} |")
