import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import math
import numpy as np
from backend.config.settings import OPEN_METEO_ELEVATION_URL
from backend.data_sources.http_client import get_json
from backend.services.waterlogging_features import build_terrain_manifest, get_live_terrain, build_live_features
from backend.services.waterlogging_service import predict_live, get_warning_threshold
from backend.utils.cache import make_cache_key

lat, lon = 16.3067, 80.4365
manifest = build_terrain_manifest(lat, lon)
lats = ",".join(str(p["lat"]) for p in manifest)
lons = ",".join(str(p["lon"]) for p in manifest)

data = get_json(OPEN_METEO_ELEVATION_URL, {"latitude": lats, "longitude": lons})
elevations = data.get("elevation", [])

print("=== 13-POINT LIVE OPEN-METEO RESPONSE FOR GUNTUR (16.3067, 80.4365) ===")
for i, (p, e) in enumerate(zip(manifest, elevations)):
    print(f"[{i:2d}] {p['id']:15s} | {p['role']:18s} | lat: {p['lat']:9.6f} | lon: {p['lon']:9.6f} | elevation: {e} m")

# Model 5-point extraction
elev_map = {p["id"]: e for p, e in zip(manifest, elevations)}
z_center = elev_map["model_center"]
z_north = elev_map["model_north"]
z_south = elev_map["model_south"]
z_east = elev_map["model_east"]
z_west = elev_map["model_west"]

offset = 0.01
north_south_distance = 2 * offset * 111320
east_west_distance = 2 * offset * 111320 * np.cos(np.radians(lat))
dz_dy = (z_north - z_south) / north_south_distance
dz_dx = (z_east - z_west) / east_west_distance
model_slope = float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))

print("\n=== MODEL CRITICAL VALUES ===")
print(f"model_center: {z_center} m")
print(f"model_north:  {z_north} m")
print(f"model_south:  {z_south} m")
print(f"model_east:   {z_east} m")
print(f"model_west:   {z_west} m")
print(f"model_slope:  {model_slope:.6f}° (rounded: {model_slope:.2f}°)")

# Contextual 8 points
context_ids = ["context_n", "context_ne", "context_e", "context_se", "context_s", "context_sw", "context_w", "context_nw"]
context_elevs = [elev_map[cid] for cid in context_ids]
context_mean = float(np.mean(context_elevs))
rel_elev = float(z_center - context_mean)
all_9 = [z_center] + context_elevs
local_min = float(min(all_9))
local_max = float(max(all_9))
local_relief = float(local_max - local_min)
model_gradient = float(np.tan(np.radians(model_slope)) * 1000.0)

# Downslope
azimuth = float(np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0)

print("\n=== ENHANCED CONTEXTUAL VALUES ===")
for cid in context_ids:
    print(f"{cid:12s}: {elev_map[cid]} m")
print(f"context_mean:       {context_mean:.2f} m")
print(f"relative_elevation: {rel_elev:+.2f} m")
print(f"local_relief:       {local_relief:.2f} m (range: {local_min}m - {local_max}m)")
print(f"model_gradient:     {model_gradient:.2f} m/km")
print(f"downslope_azimuth:  {azimuth:.1f}°")
