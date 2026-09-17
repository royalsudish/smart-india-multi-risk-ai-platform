import os
from dotenv import load_dotenv

load_dotenv()

APP_USER_AGENT = os.getenv(
    "APP_USER_AGENT",
    "SmartMultiRiskPlatform/1.0 (set APP_CONTACT_EMAIL in .env)"
)
APP_CONTACT_EMAIL = os.getenv("APP_CONTACT_EMAIL", "")

NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL",
    "https://nominatim.openstreetmap.org/search"
)

OVERPASS_ENDPOINTS = [
    x.strip() for x in os.getenv(
        "OVERPASS_ENDPOINTS",
        "https://overpass-api.de/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter"
    ).split(",") if x.strip()
]

BHUVAN_WMS_URL = os.getenv(
    "BHUVAN_WMS_URL",
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)

MOSDAC_MDAPI_ZIP = os.getenv(
    "MOSDAC_MDAPI_ZIP",
    "https://www.mosdac.gov.in/software/mdapi.zip"
)

OPEN_METEO_FORECAST_URL = os.getenv(
    "OPEN_METEO_FORECAST_URL",
    "https://api.open-meteo.com/v1/forecast"
)
OPEN_METEO_ARCHIVE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive"
)
OPEN_METEO_AIR_URL = os.getenv(
    "OPEN_METEO_AIR_URL",
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)
OPEN_METEO_ELEVATION_URL = os.getenv(
    "OPEN_METEO_ELEVATION_URL",
    "https://api.open-meteo.com/v1/elevation"
)
OPEN_METEO_MARINE_URL = os.getenv(
    "OPEN_METEO_MARINE_URL",
    "https://marine-api.open-meteo.com/v1/marine"
)

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
