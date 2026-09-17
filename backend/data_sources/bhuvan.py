import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from backend.config.settings import BHUVAN_WMS_URL, REQUEST_TIMEOUT_SECONDS
from backend.data_sources.http_client import session

def get_capabilities_xml():
    s = session()
    r = s.get(BHUVAN_WMS_URL, params={
        "service": "WMS",
        "request": "GetCapabilities",
        "version": "1.1.1",
    }, timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.text

def list_layers(limit=200):
    xml = get_capabilities_xml()
    root = ET.fromstring(xml)
    rows = []
    for layer in root.findall(".//Layer"):
        name = layer.findtext("Name")
        title = layer.findtext("Title")
        if name:
            rows.append({"name": name, "title": title or name})
        if len(rows) >= limit:
            break
    return rows

def getmap_url(layer, bbox, width=800, height=600, srs="EPSG:4326", image_format="image/png"):
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon,min_lat,max_lon,max_lat]")
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.1.1",
        "layers": layer,
        "styles": "",
        "srs": srs,
        "bbox": ",".join(str(x) for x in bbox),
        "width": int(width),
        "height": int(height),
        "format": image_format,
        "transparent": "true",
    }
    return BHUVAN_WMS_URL + "?" + urlencode(params)
