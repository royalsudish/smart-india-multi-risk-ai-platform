import time
import requests
import rasterio
import numpy as np
from datetime import datetime, timezone, timedelta
from rasterio.windows import Window
from rasterio.warp import transform
from backend.data_sources.sentinel1_sar import sign_planetary_computer_asset

CITIES = {
    "Guntur": (16.3067, 80.4365),
    "Visakhapatnam": (17.6868, 83.2185),
    "Hyderabad": (17.3850, 78.4867),
}

def check_cities():
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=90)
    time_range = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    for name, (lat, lon) in CITIES.items():
        print(f"\n--- Checking {name} ({lat}, {lon}) ---", flush=True)
        payload = {
            "collections": ["sentinel-1-rtc"],
            "intersects": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
            "limit": 5,
            "datetime": time_range,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }
        r = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", json=payload, timeout=10)
        if r.status_code != 200:
            print(f"STAC search error: {r.status_code}", flush=True)
            continue
        features = r.json().get("features", [])
        print(f"Found {len(features)} scenes.", flush=True)
        if not features:
            continue
        scene = features[0]
        scene_id = scene.get("id")
        props = scene.get("properties", {})
        scene_date = props.get("datetime") or props.get("start_datetime")
        orbit = props.get("sat:orbit_state") or props.get("sat:orbit_direction") or "unknown"
        print(f"Latest Scene ID: {scene_id}", flush=True)
        print(f"Date: {scene_date}, Orbit: {orbit}", flush=True)

        vv_asset = scene.get("assets", {}).get("vv")
        if not vv_asset:
            print("No VV asset!", flush=True)
            continue
        raw_href = vv_asset["href"]
        signed_url, _, _ = sign_planetary_computer_asset(raw_href)

        t0 = time.time()
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
            VSI_CACHE="TRUE",
        ):
            with rasterio.open(signed_url) as src:
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                inv_trans = ~src.transform
                col, row = inv_trans * (xs[0], ys[0])
                c_int, r_int = int(round(col)), int(round(row))
                win = Window(max(0, c_int - 2), max(0, r_int - 2), 5, 5)
                arr = src.read(1, window=win)
                val = arr[min(2, arr.shape[0]-1), min(2, arr.shape[1]-1)]
                val_linear = float(val)
                val_db = 10.0 * np.log10(val_linear) if val_linear > 0 else -10.5
                print(f"Pixel read in {time.time()-t0:.2f}s! Linear: {val_linear:.5f}, VV dB: {val_db:.2f} dB", flush=True)

if __name__ == "__main__":
    check_cities()
