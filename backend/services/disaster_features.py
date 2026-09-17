from backend.data_sources.open_meteo import forecast, air_quality, elevation

AUTO_NAMES = [
    "temp_2m","precip_1h","rain_24h","rain_48h","wind_speed_10m",
    "wind_gusts_10m","surface_pressure","weather_code","us_aqi","pm2_5","elevation"
]

MANUAL_OR_EXTERNAL_NAMES = [
    "river_discharge","max_quake_mag","quake_depth","tsunami_flag",
    "gdacs_cyclone","gdacs_volcano","wind_pressure_shear",
    "precipitation_saturation_rate","thermal_particulate_index"
]

def _last(values, default=None):
    if not values:
        return default
    for v in reversed(values):
        if v is not None:
            return v
    return default

def build_disaster_features(lat, lon, overrides=None):
    overrides = overrides or {}
    w = forecast(lat, lon, past_days=2, forecast_days=1)
    a = air_quality(lat, lon)

    wh = w.get("hourly") or {}
    wc = w.get("current") or {}
    ac = a.get("current") or {}

    rain = wh.get("rain") or wh.get("precipitation") or []
    def sum_tail(n):
        vals = rain[-n:] if len(rain) >= n else rain
        return float(sum(float(x or 0.0) for x in vals))

    features = {
        "temp_2m": wc.get("temperature_2m") or _last(wh.get("temperature_2m")),
        "precip_1h": wc.get("precipitation") or _last(wh.get("precipitation"), 0.0),
        "rain_24h": sum_tail(24),
        "rain_48h": sum_tail(48),
        "wind_speed_10m": wc.get("wind_speed_10m") or _last(wh.get("wind_speed_10m")),
        "wind_gusts_10m": wc.get("wind_gusts_10m") or _last(wh.get("wind_gusts_10m")),
        "surface_pressure": wc.get("surface_pressure") or _last(wh.get("surface_pressure")),
        "weather_code": wc.get("weather_code") or _last(wh.get("weather_code")),
        "us_aqi": ac.get("us_aqi"),
        "pm2_5": ac.get("pm2_5"),
        "elevation": elevation(lat, lon),
    }
    features.update(overrides)

    missing = [k for k in MANUAL_OR_EXTERNAL_NAMES if features.get(k) is None]
    return {
        "features": features,
        "missing_external_features": missing,
        "note": (
            "The uploaded disaster model needs hazard features not covered by the APIs supplied so far. "
            "Send them in overrides or add their source connectors; they are not silently defaulted to zero."
        )
    }
