import time
import json
from app import create_app

def test_routes():
    app = create_app()
    client = app.test_client()

    print("1. Testing GET /api/waterlogging/sar/sentinel1_vv...", flush=True)
    t0 = time.time()
    r1 = client.get("/api/waterlogging/sar/sentinel1_vv?lat=16.3067&lon=80.4365")
    print(f"Status: {r1.status_code} in {time.time()-t0:.2f}s", flush=True)
    data1 = r1.get_json()
    print("Response data:", json.dumps(data1, indent=2), flush=True)
    d1 = data1.get("data", data1)
    assert r1.status_code == 200
    assert d1["status"] == "available"
    assert d1["provenance"]["sentinel1_vv"]["quality"] == "LIVE"

    print("\n2. Testing POST /api/predict/drainage/live (with precomputed S2 to test drainage route)...", flush=True)
    payload = {
        "lat": 16.3067,
        "lon": 80.4365,
        "include_osm": False,
        "precomputed_sources": {
            "sentinel2": {
                "status": "available",
                "provenance": {
                    "NDVI": {"value": 0.45, "quality": "LIVE"},
                    "NDBI": {"value": -0.08, "quality": "LIVE"},
                }
            }
        }
    }
    t0 = time.time()
    r2 = client.post("/api/predict/drainage/live", json=payload)
    print(f"Drainage POST Status: {r2.status_code} in {time.time()-t0:.2f}s", flush=True)
    data2 = r2.get_json()
    assert r2.status_code == 200
    d = data2.get("data", {})
    print("Features:", json.dumps(d.get("features"), indent=2), flush=True)
    print("Data quality:", json.dumps(d.get("data_quality"), indent=2), flush=True)
    print("Prediction:", json.dumps(d.get("prediction"), indent=2), flush=True)
    assert d.get("data_quality", {}).get("status") == "FULL_LIVE"

    print("\n3. Testing GET /api/waterlogging/drainage...", flush=True)
    t0 = time.time()
    r3 = client.get("/api/waterlogging/drainage?lat=16.3067&lon=80.4365&radius_m=500")
    print(f"Drainage GET Status: {r3.status_code} in {time.time()-t0:.2f}s", flush=True)
    assert r3.status_code in (200, 400)

    print("\nALL ROUTE TESTS COMPLETED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    test_routes()
