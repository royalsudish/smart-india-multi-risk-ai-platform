import math
from datetime import datetime
from backend.config.settings import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_AIR_URL,
    OPEN_METEO_ELEVATION_URL,
    OPEN_METEO_MARINE_URL,
    OPEN_METEO_FLOOD_URL,
)
from backend.data_sources.http_client import get_json
from backend.utils.cache import provider_cache, make_cache_key, TTL_FLOOD

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

def air_quality(lat, lon, timeout=5):
    return get_json(OPEN_METEO_AIR_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone",
        "hourly": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "auto",
        "forecast_days": 2,
    }, timeout=timeout)

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

def _extract_valid_discharge(daily_dict):
    if not daily_dict or not isinstance(daily_dict, dict):
        return None, None
    times = daily_dict.get("time") or []
    discharges = daily_dict.get("river_discharge") or []
    if not discharges:
        return None, None

    today_str = datetime.now().strftime("%Y-%m-%d")
    target_idx = times.index(today_str) if today_str in times else len(times) - 1

    # Search current date first, then recent observed dates backwards, then forward forecast
    search_indices = [target_idx] + [i for i in range(target_idx - 1, -1, -1)] + [i for i in range(target_idx + 1, len(discharges))]
    for idx in search_indices:
        if 0 <= idx < len(discharges):
            val = discharges[idx]
            if val is not None:
                try:
                    fval = float(val)
                    if not (math.isnan(fval) or math.isinf(fval)):
                        date_used = times[idx] if idx < len(times) else None
                        return fval, date_used
                except (ValueError, TypeError):
                    continue
    return None, None

def flood(lat, lon, past_days=1, forecast_days=1, timeout=10, return_details=False):
    """
    Fetches river discharge (m³/s) from Open-Meteo Flood API using GloFAS v4.
    If the exact cell has no river data, searches immediate adjacent grid points
    within 0.05° (~5km) as recommended by Open-Meteo documentation.
    """
    cache_key = make_cache_key("open_meteo_flood", lat, lon, {"p": past_days, "f": forecast_days, "det": return_details})
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    def _ret(res):
        if res is not None and not (isinstance(res, dict) and res.get("status") == "unavailable"):
            provider_cache.set(cache_key, res, ttl=TTL_FLOOD)
        return res

    try:
        data = get_json(OPEN_METEO_FLOOD_URL, {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "auto",
        }, timeout=timeout)
        val, date_used = _extract_valid_discharge(data.get("daily") if isinstance(data, dict) else None)
        if val is not None:
            if return_details:
                return _ret({
                    "river_discharge": val,
                    "status": "available",
                    "unit": "m³/s",
                    "source": "Open-Meteo Flood (GloFAS)",
                    "date": date_used,
                    "method": "exact_coordinate",
                    "lat": lat,
                    "lon": lon,
                })
            return _ret(val)

        # Try adjacent candidate spatial offsets within GloFAS grid spacing (0.05° ~ 5km)
        offsets = [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0.05, 0.05), (-0.05, -0.05)]
        lats = ",".join(str(round(lat + dy, 4)) for dy, dx in offsets)
        lons = ",".join(str(round(lon + dx, 4)) for dy, dx in offsets)
        cand_data = get_json(OPEN_METEO_FLOOD_URL, {
            "latitude": lats,
            "longitude": lons,
            "daily": "river_discharge",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "auto",
        }, timeout=timeout)
        if isinstance(cand_data, list):
            for i, item in enumerate(cand_data):
                cand_val, cand_date = _extract_valid_discharge(item.get("daily") if isinstance(item, dict) else None)
                if cand_val is not None:
                    if return_details:
                        return _ret({
                            "river_discharge": cand_val,
                            "status": "available",
                            "unit": "m³/s",
                            "source": "Open-Meteo Flood (GloFAS)",
                            "date": cand_date,
                            "method": "adjacent_grid_cell",
                            "lat": round(lat + offsets[i][0], 4),
                            "lon": round(lon + offsets[i][1], 4),
                        })
                    return _ret(cand_val)
    except Exception as exc:
        if return_details:
            return {
                "river_discharge": None,
                "status": "unavailable",
                "unit": "m³/s",
                "source": "Open-Meteo Flood (GloFAS)",
                "reason": str(exc),
            }
        return None

    if return_details:
        return _ret({
            "river_discharge": None,
            "status": "unavailable",
            "unit": "m³/s",
            "source": "Open-Meteo Flood (GloFAS)",
            "reason": "No simulated river discharge available within GloFAS basin range",
        })
    return None

