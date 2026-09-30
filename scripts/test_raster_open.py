import time
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from rasterio.transform import from_gcps
from backend.data_sources.sentinel1_sar import sign_planetary_computer_asset

def test_rasterio():
    lat, lon = 16.3067, 80.4365

    # 1. Test PC RTC COG
    pc_href = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2026/9/22/IW/DV/S1D_IW_GRDH_1SDV_20260922T003011_20260922T003036_004683_008C06_BC81/measurement/iw-vv.rtc.tiff"
    signed_url, _, _ = sign_planetary_computer_asset(pc_href)

    t0 = time.time()
    print("Testing Planetary Computer RTC COG open...")
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
        VSI_CACHE="TRUE",
    ):
        with rasterio.open(signed_url) as src:
            print(f"PC open took {time.time()-t0:.2f}s, CRS: {src.crs}, shape: ({src.width}, {src.height})")
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            inv_trans = ~src.transform
            col, row = inv_trans * (xs[0], ys[0])
            print(f"Projected col: {col:.1f}, row: {row:.1f}")
            c_int, r_int = int(round(col)), int(round(row))
            win = Window(c_int - 2, r_int - 2, 5, 5)
            t1 = time.time()
            arr = src.read(1, window=win)
            print(f"PC 5x5 window read took {time.time()-t1:.2f}s: {arr}")

    # 2. Test AWS GRD TIFF
    aws_href = "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2026/9/22/IW/DV/S1D_IW_GRDH_1SDV_20260922T003011_20260922T003036_004683_008C06_1ACE/measurement/iw-vv.tiff"
    t0 = time.time()
    print("\nTesting AWS GRD TIFF open...")
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
        VSI_CACHE="TRUE",
    ):
        try:
            with rasterio.open(aws_href) as src:
                print(f"AWS open took {time.time()-t0:.2f}s, GCPs count: {len(src.gcps[0]) if src.gcps else 0}")
                t1 = time.time()
                # Reading a window from un-tiled GRD TIFF:
                win = Window(100, 100, 5, 5)
                arr = src.read(1, window=win)
                print(f"AWS 5x5 read took {time.time()-t1:.2f}s: {arr}")
        except Exception as e:
            print("AWS error:", e)

if __name__ == "__main__":
    test_rasterio()
