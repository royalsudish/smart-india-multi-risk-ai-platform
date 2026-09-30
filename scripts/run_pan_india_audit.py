import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
from datetime import datetime, timezone
from backend.data_sources.nasa_gpm import get_grid_indices, discover_latest_granules, aggregate_gpm_rainfall

cities = [
    ('Guntur', 16.3067, 80.4365),
    ('Mumbai', 19.0760, 72.8777),
    ('Patna', 25.5941, 85.1376),
    ('Guwahati', 26.1445, 91.7362),
    ('Bengaluru', 12.9716, 77.5946),
    ('Dehradun', 30.3165, 78.0322),
    ('Jaisalmer', 26.9157, 70.9083)
]

print("1. DISCOVERING LATEST GRANULES FROM NASA CMR (COLD FETCH)...")
t0 = time.perf_counter()
disc = discover_latest_granules()
disc_time = (time.perf_counter() - t0) * 1000
print(f"Discovery latency: {disc_time:.2f} ms")
print("Latest HH Granule:", disc['half_hourly']['granule_id'])
print("Start Time UTC:", disc['half_hourly']['start_time_utc'])
print("Start Time IST:", disc['half_hourly']['start_time_ist'])
print("Data Age Hours:", disc['half_hourly']['data_age_hours'])
print("Latest Daily Granule:", disc['daily']['granules'][0]['granule_id'] if disc['daily']['granules'] else None)
print("Latest Daily Date:", disc['daily']['latest_completed_daily_date'])
print("Daily Granules Count:", disc['daily']['days_count'])

print("\n" + "="*110)
print(f"{'City':<11} | {'Coords (Lat, Lon)':<18} | {'Nearest Cell':<16} | {'Dist (km)':<9} | {'1h / 24h':<13} | {'Data Age':<10} | {'Grid Size':<9} | {'Status':<14} | {'Latency'}")
print("="*110)

for name, lat, lon in cities:
    t_c = time.perf_counter()
    res = aggregate_gpm_rainfall(lat, lon, delta_deg=0.5)
    lat_ms = (time.perf_counter() - t_c) * 1000
    pt = res['point_extraction']
    obs = res['observation_metadata']
    grid = res['grid']
    status = res['status']
    cache_st = res.get('cache_status', 'unknown')
    
    h1 = pt['windows']['1h']['value_mm']
    h24 = pt['windows']['24h']['value_mm']
    h1_str = f"{h1}mm" if h1 is not None else "null"
    h24_str = f"{h24}mm" if h24 is not None else "null"
    rain_str = f"{h1_str} / {h24_str}"
    
    c_lat = pt['cell_center_lat']
    c_lon = pt['cell_center_lon']
    dist = pt['distance_to_cell_center_km']
    grid_sz = f"{grid['grid_rows']}x{grid['grid_columns']}"
    age = f"{obs['data_age_hours']}h" if obs['data_age_hours'] is not None else "N/A"
    
    print(f"{name:<11} | {lat:.4f}, {lon:.4f} | {c_lat:.2f}, {c_lon:.2f} | {dist:<9.3f} | {rain_str:<13} | {age:<10} | {grid_sz:<9} | {status:<14} | {lat_ms:.2f} ms ({cache_st})")

print("\n" + "="*110)
print("SECOND RUN: TESTING AGGREGATE CACHE HITS")
print("="*110)
for name, lat, lon in cities:
    t_c = time.perf_counter()
    res = aggregate_gpm_rainfall(lat, lon, delta_deg=0.5)
    lat_ms = (time.perf_counter() - t_c) * 1000
    cache_st = res.get('cache_status', 'unknown')
    print(f"{name:<11} | Latency: {lat_ms:.3f} ms | Cache Status: {cache_st}")
