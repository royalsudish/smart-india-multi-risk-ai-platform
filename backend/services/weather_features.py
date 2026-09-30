from datetime import datetime
from math import sin, cos, pi, radians, sqrt, atan2
from backend.data_sources.open_meteo import forecast

REQUIRED_39 = [
    "temperature_2m","relative_humidity_2m","dew_point_2m","apparent_temperature",
    "precipitation","rain","surface_pressure","et0_fao_evapotranspiration",
    "vapour_pressure_deficit","wind_speed_10m","wind_gusts_10m",
    "soil_temperature_0_to_7cm","soil_moisture_0_to_7cm","shortwave_radiation",
    "direct_normal_irradiance","diffuse_radiation","cloud_cover","latitude","longitude",
    "wind_u_vector","wind_v_vector","wet_bulb_temp","sin_hour","cos_hour","sin_day","cos_day",
    "pressure_tendency_3h","temp_lag_1","humidity_lag_1","radiation_lag_1","temp_lag_2",
    "humidity_lag_2","radiation_lag_2","temp_lag_3","humidity_lag_3","radiation_lag_3",
    "temp_lag_24","humidity_lag_24","radiation_lag_24"
]

def _value(hourly, name, idx):
    values = hourly.get(name) or []
    if not values:
        return None
    idx = max(0, min(idx, len(values)-1))
    return values[idx]

def _wet_bulb_stull(temp_c, rh):
    # Stull (2011) approximation; valid for ordinary near-surface ranges.
    import math
    t = float(temp_c)
    r = max(1e-6, min(100.0, float(rh)))
    return (
        t * math.atan(0.151977 * math.sqrt(r + 8.313659))
        + math.atan(t + r)
        - math.atan(r - 1.676331)
        + 0.00391838 * (r ** 1.5) * math.atan(0.023101 * r)
        - 4.686035
    )

def build_live_weather_features(lat, lon, raw_forecast=None):
    """
    Builds the 39 inputs found in the uploaded WeatherGPT bundle.

    IMPORTANT:
    The engineering formulas below are reconstructed from the feature names and
    saved preprocessor metadata. If you have the original training notebook,
    compare the formulas before treating this as production/scientific output.
    """
    raw = raw_forecast or forecast(lat, lon, past_days=2, forecast_days=2)
    hourly = raw.get("hourly") or {}
    times = hourly.get("time") or []
    current_time = (raw.get("current") or {}).get("time")

    if not times:
        raise RuntimeError("Open-Meteo returned no hourly data")

    if current_time in times:
        idx = times.index(current_time)
    else:
        # Prefer the most recent available row not after current time.
        idx = len(times) - 1
        if current_time:
            candidates = [i for i, t in enumerate(times) if t <= current_time]
            if candidates:
                idx = candidates[-1]

    base_names = [
        "temperature_2m","relative_humidity_2m","dew_point_2m","apparent_temperature",
        "precipitation","rain","surface_pressure","et0_fao_evapotranspiration",
        "vapour_pressure_deficit","wind_speed_10m","wind_gusts_10m",
        "soil_temperature_0_to_7cm","soil_moisture_0_to_7cm","shortwave_radiation",
        "direct_normal_irradiance","diffuse_radiation","cloud_cover"
    ]
    f = {name: _value(hourly, name, idx) for name in base_names}
    f["latitude"] = float(lat)
    f["longitude"] = float(lon)

    wind_dir = _value(hourly, "wind_direction_10m", idx)
    wind_speed = float(f["wind_speed_10m"] or 0.0)
    theta = radians(float(wind_dir or 0.0))
    # Meteorological direction is where wind comes FROM.
    f["wind_u_vector"] = -wind_speed * sin(theta)
    f["wind_v_vector"] = -wind_speed * cos(theta)

    f["wet_bulb_temp"] = _wet_bulb_stull(
        f["temperature_2m"], f["relative_humidity_2m"]
    )

    dt = datetime.fromisoformat(times[idx])
    hour = dt.hour
    day = dt.timetuple().tm_yday
    f["sin_hour"] = sin(2*pi*hour/24.0)
    f["cos_hour"] = cos(2*pi*hour/24.0)
    f["sin_day"] = sin(2*pi*day/365.25)
    f["cos_day"] = cos(2*pi*day/365.25)

    p_now = float(f["surface_pressure"] or 0.0)
    p_3 = float(_value(hourly, "surface_pressure", idx-3) or p_now)
    f["pressure_tendency_3h"] = p_now - p_3

    for lag in (1, 2, 3, 24):
        f[f"temp_lag_{lag}"] = _value(hourly, "temperature_2m", idx-lag)
        f[f"humidity_lag_{lag}"] = _value(hourly, "relative_humidity_2m", idx-lag)
        f[f"radiation_lag_{lag}"] = _value(hourly, "shortwave_radiation", idx-lag)

    missing = [k for k in REQUIRED_39 if f.get(k) is None]
    if missing:
        raise RuntimeError("Missing live weather features: " + ", ".join(missing))

    return {
        "features": f,
        "source": "Open-Meteo forecast API",
        "source_time": times[idx],
        "reconstruction_warning": (
            "Feature formulas should be checked against the original WeatherGPT training notebook."
        )
    }
