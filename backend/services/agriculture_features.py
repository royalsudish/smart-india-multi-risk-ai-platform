import numpy as np
import pandas as pd

from backend.config.settings import OPEN_METEO_FORECAST_URL
from backend.data_sources.http_client import get_json

CORE_NUMERIC_FEATURES = [
    "latitude", "longitude",
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max",
    "surface_pressure_mean", "shortwave_radiation_sum", "dew_point_2m_mean",
    "vapor_pressure_deficit", "et0_fao_evapotranspiration",
    "rainfall_1d", "rainfall_3d_sum", "rainfall_7d_sum", "rainfall_14d_sum",
    "temperature_3d_mean", "temperature_7d_mean", "temperature_3d_max",
    "temperature_7d_max", "temperature_3d_min",
    "humidity_3d_mean", "humidity_7d_mean", "et0_3d_sum", "et0_7d_sum",
    "wind_speed_3d_mean", "consecutive_dry_days", "consecutive_rain_days",
    "doy_sin", "doy_cos", "month"
]


def get_live_weather_metrics(lat: float, lon: float) -> dict:
    """
    Source-faithful implementation of the uploaded notebook's final
    place-to-advisory weather feature builder.
    """
    data = get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "relative_humidity_2m_mean",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration",
            ]),
            "past_days": 14,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        },
    )

    d = data["daily"]
    df = pd.DataFrame(d).ffill().bfill()

    if len(df) < 14:
        raise ValueError("Open-Meteo returned fewer than 14 daily rows.")

    curr = df.iloc[-1]

    r1 = float(curr["precipitation_sum"] or 0.0)
    r3 = float(df["precipitation_sum"].iloc[-3:].sum())
    r7 = float(df["precipitation_sum"].iloc[-7:].sum())
    r14 = float(df["precipitation_sum"].iloc[-14:].sum())

    t_mean = float(curr["temperature_2m_mean"])
    t_max = float(curr["temperature_2m_max"])
    t_min = float(curr["temperature_2m_min"])
    rh = float(curr["relative_humidity_2m_mean"])

    t3_mean = float(df["temperature_2m_mean"].iloc[-3:].mean())
    t7_mean = float(df["temperature_2m_mean"].iloc[-7:].mean())
    t3_max = float(df["temperature_2m_max"].iloc[-3:].max())
    t7_max = float(df["temperature_2m_max"].iloc[-7:].max())
    t3_min = float(df["temperature_2m_min"].iloc[-3:].min())

    h3_mean = float(df["relative_humidity_2m_mean"].iloc[-3:].mean())
    h7_mean = float(df["relative_humidity_2m_mean"].iloc[-7:].mean())

    et0_val = float(curr["et0_fao_evapotranspiration"] or 4.2)
    et0_3d = float(df["et0_fao_evapotranspiration"].iloc[-3:].sum())
    et0_7d = float(df["et0_fao_evapotranspiration"].iloc[-7:].sum())

    wind_max = float(curr["wind_speed_10m_max"] or 8.5)
    wind3_mean = float(df["wind_speed_10m_max"].iloc[-3:].mean())

    p_hist = df["precipitation_sum"].tolist()

    dry_streak = 0
    for p in reversed(p_hist):
        if p < 1.0:
            dry_streak += 1
        else:
            break

    rain_streak = 0
    for p in reversed(p_hist):
        if p >= 2.5:
            rain_streak += 1
        else:
            break

    es = 0.61078 * np.exp(
        (17.27 * t_mean) / (t_mean + 237.3)
    )
    ea = es * (rh / 100.0)
    vpd = max(0.0, es - ea)

    return {
        "temperature_2m_mean": t_mean,
        "temperature_2m_min": t_min,
        "temperature_2m_max": t_max,
        "relative_humidity_2m_mean": rh,
        "precipitation_sum": r1,
        "wind_speed_10m_max": wind_max,

        # These two values intentionally match the notebook's final
        # inference implementation rather than silently changing its contract.
        "surface_pressure_mean": 101.3,
        "dew_point_2m_mean": t_mean - ((100.0 - rh) / 5.0),

        "shortwave_radiation_sum": float(
            curr["shortwave_radiation_sum"] or 19.5
        ),
        "vapor_pressure_deficit": vpd,
        "et0_fao_evapotranspiration": et0_val,
        "rainfall_1d": r1,
        "rainfall_3d_sum": r3,
        "rainfall_7d_sum": r7,
        "rainfall_14d_sum": r14,
        "temperature_3d_mean": t3_mean,
        "temperature_7d_mean": t7_mean,
        "temperature_3d_max": t3_max,
        "temperature_7d_max": t7_max,
        "temperature_3d_min": t3_min,
        "humidity_3d_mean": h3_mean,
        "humidity_7d_mean": h7_mean,
        "et0_3d_sum": et0_3d,
        "et0_7d_sum": et0_7d,
        "wind_speed_3d_mean": wind3_mean,
        "consecutive_dry_days": dry_streak,
        "consecutive_rain_days": rain_streak,
    }


def assemble_agriculture_payload(lat, lon, crop, crop_stage):
    weather = get_live_weather_metrics(float(lat), float(lon))

    payload = dict(weather)
    payload["latitude"] = float(lat)
    payload["longitude"] = float(lon)

    # Exact logic from the notebook's final inference cell.
    if crop in ["Wheat", "Mustard"]:
        agro_season = "RABI"
        month_val = 1
    else:
        agro_season = "KHARIF"
        month_val = 8

    payload["crop"] = crop
    payload["crop_stage"] = crop_stage
    payload["agro_season"] = agro_season
    payload["month"] = month_val

    doy = month_val * 30
    payload["doy_sin"] = float(
        np.sin(2 * np.pi * doy / 365.25)
    )
    payload["doy_cos"] = float(
        np.cos(2 * np.pi * doy / 365.25)
    )

    return payload
