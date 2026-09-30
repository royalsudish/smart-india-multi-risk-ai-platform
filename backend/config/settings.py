import os
from dotenv import load_dotenv

load_dotenv()

APP_USER_AGENT = os.getenv(
    "APP_USER_AGENT",
    "SmartMultiRiskPlatform/1.0"
)
APP_CONTACT_EMAIL = os.getenv("APP_CONTACT_EMAIL", "")

NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL",
    "https://nominatim.openstreetmap.org/search"
)
NOMINATIM_REVERSE_URL = os.getenv(
    "NOMINATIM_REVERSE_URL",
    "https://nominatim.openstreetmap.org/reverse"
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
OPEN_METEO_FLOOD_URL = os.getenv(
    "OPEN_METEO_FLOOD_URL",
    "https://flood-api.open-meteo.com/v1/flood"
)
USGS_EARTHQUAKE_URL = os.getenv(
    "USGS_EARTHQUAKE_URL",
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
)
GDACS_EVENTS_URL = os.getenv(
    "GDACS_EVENTS_URL",
    "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventtypes=TC,VO"
)

# Open-Meteo Spatial Rainfall Configuration (Phase E Replacement)
OPEN_METEO_SPATIAL_URL = os.getenv(
    "OPEN_METEO_SPATIAL_URL",
    "https://api.open-meteo.com/v1/forecast"
)

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

# NASA FIRMS Active Fire API
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
FIRMS_BASE_URL = os.getenv("FIRMS_BASE_URL", "https://firms.modaps.eosdis.nasa.gov/api/area/csv")

# NDMA SACHET CAP Alerts Feed
SACHET_CAP_URL = os.getenv("SACHET_CAP_URL", "https://sachet.ndma.gov.in/cap_public_website/FetchAllAlertDetails")

# Smart Alert Engine & Database Storage
ALERT_DB_PATH = os.getenv(
    "ALERT_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "alerts.db")
)

# Notification Channels Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "alerts@smart-multi-risk.gov.in")
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", "")



