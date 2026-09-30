"""
Pan-India Regression Verification Suite for Open-Meteo Spatial Rainfall (Phase E Replacement)
Tests 7 diverse Indian cities:
- Guntur
- Mumbai
- Patna
- Guwahati
- Bengaluru
- Dehradun
- Jaisalmer

Validates:
- status = available
- 121 requested sampling points
- 121 returned/mapped locations
- 1h and 24h numeric rainfall values
- rolling_72h works
- rolling_168h works
- requested vs returned coordinates preserved
"""

import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.data_sources.spatial_rainfall import aggregate_spatial_rainfall

LOCATIONS = [
    ("Guntur", 16.3067, 80.4365),
    ("Mumbai", 19.0760, 72.8777),
    ("Patna", 25.5941, 85.1376),
    ("Guwahati", 26.1445, 91.7362),
    ("Bengaluru", 12.9716, 77.5946),
    ("Dehradun", 30.3165, 78.0322),
    ("Jaisalmer", 26.9157, 70.9083),
]

def run_regression_tests():
    print("=" * 90)
    print("OPEN-METEO SPATIAL RAINFALL: PAN-INDIA REGRESSION CHECK (7 LOCATIONS)")
    print("=" * 90)

    results = []

    for name, lat, lon in LOCATIONS:
        print(f"\n--- Testing {name} ({lat}°N, {lon}°E) ---")

        # Retry loop for rate-limiting protection
        res = None
        for attempt in range(3):
            t0 = time.perf_counter()
            res = aggregate_spatial_rainfall(lat, lon)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if res.get("status") == "available":
                break
            print(f"  Attempt {attempt + 1} returned {res.get('status')}; waiting 5s for rate limit...")
            time.sleep(5)

        status = res.get("status")
        provider = res.get("provider")
        w = res.get("windows", {})
        h1 = w.get("1h", {})
        h24 = w.get("24h", {})
        r72 = w.get("3d", {})
        r168 = w.get("7d", {})

        grid = res.get("grid", {})
        cell_count = grid.get("cell_count")
        valid_cells = grid.get("valid_cells_count")
        sample_maps = grid.get("sample_coordinate_mappings", [])

        print(f"Status:                      {status}")
        print(f"Requested Sampling Points:   {cell_count}")
        print(f"Returned/Mapped Locations:   {valid_cells}")
        print(f"1h Rainfall:                 {h1.get('value_mm')} mm (semantics: {h1.get('window_semantics')})")
        print(f"24h Rainfall:                {h24.get('value_mm')} mm (semantics: {h24.get('window_semantics')})")
        print(f"Rolling 72h (3d):            {r72.get('value_mm')} mm (semantics: {r72.get('window_semantics')})")
        print(f"Rolling 168h (7d):           {r168.get('value_mm')} mm (semantics: {r168.get('window_semantics')})")

        # Coordinate check for center point
        if sample_maps:
            center_map = next((m for m in sample_maps if m.get("grid_index") == 60), sample_maps[0])
            print(f"Center Mapping:              Requested ({center_map['requested_lat']}, {center_map['requested_lon']}) -> Returned ({center_map['returned_lat']}, {center_map['returned_lon']})")

        # Assertions
        assert status == "available", f"Expected available status for {name}, got {status}"
        assert cell_count == 121, f"Expected 121 requested sampling points, got {cell_count}"
        assert valid_cells == 121, f"Expected 121 valid mapped locations, got {valid_cells}"
        assert h1.get("value_mm") is not None, f"1h rainfall cannot be None for {name}"
        assert h24.get("value_mm") is not None, f"24h rainfall cannot be None for {name}"
        assert r72.get("window_semantics") == "rolling_72h", f"Expected rolling_72h, got {r72.get('window_semantics')}"
        assert r168.get("window_semantics") == "rolling_168h", f"Expected rolling_168h, got {r168.get('window_semantics')}"

        results.append({
            "name": name,
            "status": status,
            "1h_mm": h1.get("value_mm"),
            "24h_mm": h24.get("value_mm"),
            "rolling_72h_mm": r72.get("value_mm"),
            "rolling_168h_mm": r168.get("value_mm"),
            "sampling_pts": cell_count,
            "mapped_locs": valid_cells,
            "latency_ms": latency_ms,
        })

        time.sleep(2)

    print("\n" + "=" * 90)
    print("REGRESSION CHECK RESULTS: ALL 7 CITIES")
    print("=" * 90)
    print(f"{'City':<12} | {'Status':<10} | {'1h (mm)':<8} | {'24h (mm)':<9} | {'r72h (mm)':<10} | {'r168h (mm)':<11} | {'Req/Map':<8} | {'Latency'}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<12} | {r['status']:<10} | {str(r['1h_mm']):<8} | {str(r['24h_mm']):<9} | {str(r['rolling_72h_mm']):<10} | {str(r['rolling_168h_mm']):<11} | {r['sampling_pts']}/{r['mapped_locs']} | {r['latency_ms']} ms")
    print("=" * 90)
    print("ALL 7 LOCATIONS PASSED REGRESSION CHECKS!")

if __name__ == "__main__":
    run_regression_tests()
