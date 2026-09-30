import requests
from datetime import datetime, timezone, timedelta

def test_stac():
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=90)
    time_range = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    print(f"Time range: {time_range}")

    # Test Guntur
    lat, lon = 16.3067, 80.4365

    # 1. Earth Search AWS
    p_es = {
        "collections": ["sentinel-1-grd"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": 5,
        "datetime": time_range,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    try:
        r = requests.post("https://earth-search.aws.element84.com/v1/search", json=p_es, timeout=10)
        print("Earth search status:", r.status_code)
        features = r.json().get("features", [])
        print("Earth search features count:", len(features))
        if features:
            f0 = features[0]
            print("ES F0 ID:", f0.get("id"))
            print("ES F0 assets:", list(f0.get("assets", {}).keys()))
            if "vv" in f0.get("assets", {}):
                print("ES F0 vv href:", f0["assets"]["vv"].get("href"))
    except Exception as e:
        print("Earth search err:", e)

    # 2. Planetary Computer STAC
    p_pc = {
        "collections": ["sentinel-1-rtc"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": 5,
        "datetime": time_range,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    try:
        r = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", json=p_pc, timeout=10)
        print("PC status:", r.status_code)
        features = r.json().get("features", [])
        print("PC features count:", len(features))
        if features:
            f0 = features[0]
            print("PC F0 ID:", f0.get("id"))
            print("PC F0 assets:", list(f0.get("assets", {}).keys()))
            if "vv" in f0.get("assets", {}):
                print("PC F0 vv href:", f0["assets"]["vv"].get("href"))
    except Exception as e:
        print("PC err:", e)

if __name__ == "__main__":
    test_stac()
