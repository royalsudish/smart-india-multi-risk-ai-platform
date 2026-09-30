import sys
import time
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from backend.data_sources.sentinel1_sar import sign_planetary_computer_asset

def test_pc_cog():
    lat, lon = 16.3067, 80.4365
    pc_href = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2026/9/22/IW/DV/S1D_IW_GRDH_1SDV_20260922T003011_20260922T003036_004683_008C06_BC81/measurement/iw-vv.rtc.tiff"
    print("Signing PC asset...", flush=True)
    signed_url, exp, _ = sign_planetary_computer_asset(pc_href)
    print(f"Signed URL ready. Exp: {exp}", flush=True)

    t0 = time.time()
    print("Opening PC RTC COG with rasterio...", flush=True)
    with rasterio.open(signed_url) as src:
        t_open = time.time() - t0
        print(f"Opened in {t_open:.2f}s! CRS: {src.crs}, shape: {src.width}x{src.height}, block_shapes: {src.block_shapes}", flush=True)
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        inv_trans = ~src.transform
        col, row = inv_trans * (xs[0], ys[0])
        print(f"Target col: {col:.2f}, row: {row:.2f}", flush=True)
        c_int, r_int = int(round(col)), int(round(row))
        win = Window(c_int - 2, r_int - 2, 5, 5)
        t1 = time.time()
        arr = src.read(1, window=win)
        t_read = time.time() - t1
        print(f"Window read in {t_read:.2f}s! Values:\n{arr}", flush=True)

if __name__ == "__main__":
    test_pc_cog()
