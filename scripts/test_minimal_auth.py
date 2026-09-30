import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import requests
from dotenv import load_dotenv
load_dotenv(override=True)

from backend.config.settings import NASA_GPM_ENABLED, NASA_EARTHDATA_TOKEN
from backend.data_sources.nasa_gpm import (
    check_auth_configuration, discover_latest_granules, get_grid_indices,
    fetch_opendap_array
)

# 1. Confirm NASA_GPM_ENABLED
print("NASA_GPM_ENABLED =", NASA_GPM_ENABLED)

# 2. Confirm auth_method
auth_info = check_auth_configuration()
print("auth_method =", auth_info.get("auth_method"))

# 3. Earthdata token validation
urs_url = "https://urs.earthdata.nasa.gov/api/users/me"
headers = {"Authorization": f"Bearer {NASA_EARTHDATA_TOKEN.strip()}"}
r_urs = requests.get(urs_url, headers=headers, timeout=10)
print("token validation HTTP status =", r_urs.status_code)

# 4. Discover latest GPM_3IMERGHHE_07 granule
disc = discover_latest_granules()
latest_hh = disc["half_hourly"]
granule_id = latest_hh.get("granule_id")
granule_filename = granule_id.split(":")[-1] if granule_id else None

# 5. One authenticated OPeNDAP precipitation subset for lat=16.3067, lon=80.4365
grid_info = get_grid_indices(16.3067, 80.4365, delta_deg=0.5)
pt = grid_info["point"]
bbox = grid_info["bbox"]

res = fetch_opendap_array(granule_id, bbox, is_daily=False)
http_status = res.get("http_status")
content_type = res.get("content_type")

# Extract nearest cell precipitation value
grid = res.get("grid", [])
cell_val = None
if grid:
    c_r = pt["cell_lat_idx"] - bbox["lat_min_idx"]
    c_c = pt["cell_lon_idx"] - bbox["lon_min_idx"]
    if c_c < len(grid) and c_r < len(grid[c_c]):
        cell_val = grid[c_c][c_r]
    elif c_r < len(grid) and c_c < len(grid[c_r]):
        cell_val = grid[c_r][c_c]

print("OPeNDAP HTTP status =", http_status)
print("content type =", content_type)
print("granule filename =", granule_filename)
print("actual precipitation value =", cell_val)
print("units = mm/hr")
