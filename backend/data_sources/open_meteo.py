from datetime import datetime
from backend.config.settings import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_AIR_URL,
    OPEN_METEO_ELEVATION_URL,
    OPEN_METEO_MARINE_URL,
)
from backend.data_sources.http_client import get_json

WEATHER_BASE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "weather_code",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
]

def forecast(lat, lon, past_days=2, forecast_days=2):
    vars_csv = ",".join(WEATHER_BASE_VARIABLES)
    return get_json(OPEN_METEO_FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "hourly": vars_csv,
        "current": vars_csv,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    })

def historical(lat, lon, start_date, end_date, hourly=None):
    variables = hourly or WEATHER_BASE_VARIABLES
    return get_json(OPEN_METEO_ARCHIVE_URL, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "auto",
    })

def air_quality(lat, lon):
    return get_json(OPEN_METEO_AIR_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone",
        "hourly": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "auto",
        "forecast_days": 2,
    })

def elevation(lat, lon):
    data = get_json(OPEN_METEO_ELEVATION_URL, {
        "latitude": lat,
        "longitude": lon,
    })
    values = data.get("elevation") or []
    return values[0] if values else None

def elevations(points):
    """
    points = [(lat, lon), ...], up to 100 coordinates per Open-Meteo request.
    """
    if not points:
        return []
    data = get_json(OPEN_METEO_ELEVATION_URL, {
        "latitude": ",".join(str(p[0]) for p in points),
        "longitude": ",".join(str(p[1]) for p in points),
    })
    return data.get("elevation") or []

def marine(lat, lon):
    return get_json(OPEN_METEO_MARINE_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "wave_height",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction",
            "sea_level_height_msl"
        ]),
        "hourly": ",".join([
            "wave_height",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction",
            "sea_level_height_msl"
        ]),
        "timezone": "auto",
        "forecast_days": 2,
    })

def rainfall_totals_from_hourly(hourly):
    precip = hourly.get("precipitation") or []
    def total(n):
        vals = precip[-n:] if len(precip) >= n else precip
        return round(sum(float(v or 0.0) for v in vals), 4)
    return {
        "rainfall_24h": total(24),
        "rainfall_48h": total(48),
        "rainfall_72h": total(72),
        "rainfall_168h": total(168),
    }
