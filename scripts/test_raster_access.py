import requests
import rasterio
from backend.data_sources.sentinel1_sar import sign_planetary_computer_asset

def test_fetch():
    # 1. Test AWS URL
    aws_href = "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2026/9/22/IW/DV/S1D_IW_GRDH_1SDV_20260922T003011_20260922T003036_004683_008C06_1ACE/measurement/iw-vv.tiff"
    print("Testing AWS HEAD request...")
    try:
        r_aws = requests.head(aws_href, timeout=5)
        print("AWS HEAD status:", r_aws.status_code)
    except Exception as e:
        print("AWS HEAD err:", e)

    # 2. Test PC signed URL
    pc_href = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2026/9/22/IW/DV/S1D_IW_GRDH_1SDV_20260922T003011_20260922T003036_004683_008C06_BC81/measurement/iw-vv.rtc.tiff"
    print("\nTesting PC sign...")
    signed_url, exp, src = sign_planetary_computer_asset(pc_href)
    print("PC signed URL:", signed_url[:60] if signed_url else None, "exp:", exp)
    if signed_url:
        print("Testing PC HEAD request...")
        try:
            r_pc = requests.head(signed_url, timeout=5)
            print("PC HEAD status:", r_pc.status_code)
        except Exception as e:
            print("PC HEAD err:", e)

if __name__ == "__main__":
    test_fetch()
