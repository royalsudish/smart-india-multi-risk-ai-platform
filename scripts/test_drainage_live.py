import json
from backend.services.drainage_service import build_live_drainage_features, predict_live

def test_drainage():
    # Pass mock or precomputed sentinel2 to see sentinel1 live behavior clearly
    s2_mock = {
        "status": "available",
        "scene_id": "S2B_MSIL2A_20260920",
        "scene_date": "2026-09-20T05:10:00Z",
        "ndvi": 0.45,
        "ndbi": -0.08,
        "provenance": {
            "NDVI": {
                "value": 0.45,
                "unit": "index (-1 to 1)",
                "source": "Sentinel-2 L2A (10m)",
                "quality": "LIVE",
                "scene_id": "S2B_MSIL2A_20260920",
                "scene_date": "2026-09-20T05:10:00Z",
            },
            "NDBI": {
                "value": -0.08,
                "unit": "index (-1 to 1)",
                "source": "Sentinel-2 L2A (10m)",
                "quality": "LIVE",
                "scene_id": "S2B_MSIL2A_20260920",
                "scene_date": "2026-09-20T05:10:00Z",
            }
        }
    }

    res = predict_live(
        lat=16.3067,
        lon=80.4365,
        precomputed_sources={"sentinel2": s2_mock},
        include_osm=False
    )
    print("--- DRAINAGE PREDICTION WITH LIVE S1 & LIVE S2 ---", flush=True)
    print("Features:", json.dumps(res["features"], indent=2), flush=True)
    print("Data Quality:", json.dumps(res["data_quality"], indent=2), flush=True)
    print("Prediction:", json.dumps(res["prediction"], indent=2), flush=True)
    print("S1 Metadata:", json.dumps(res.get("sentinel1_metadata"), indent=2), flush=True)

if __name__ == "__main__":
    test_drainage()
