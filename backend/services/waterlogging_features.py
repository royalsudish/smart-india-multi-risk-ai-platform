import numpy as np
import pandas as pd

from backend.config.settings import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_ELEVATION_URL,
)
from backend.data_sources.http_client import get_json

SELECTED_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "latitude",
    "longitude",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "elevation",
    "slope",
    "month",
    "hour",
    "is_monsoon",
    "is_post_monsoon",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "antecedent_rain_24h",
    "antecedent_rain_3d",
    "antecedent_rain_7d",
    "antecedent_rain_index",
    "short_term_rain_ratio",
    "flatness_score",
    "rain_flat_interaction",
    "rain_low_elevation_interaction",
]


def engineer_ml_features(data):
    """
    Same feature formulas used by the uploaded waterlogging notebook.
    """
    out = data.copy()

    out["datetime"] = pd.to_datetime(
        out["datetime"],
        errors="coerce"
    )

    out["month"] = out["datetime"].dt.month
    out["hour"] = out["datetime"].dt.hour
    out["day_of_year"] = out["datetime"].dt.dayofyear

    out["is_monsoon"] = (
        out["month"].isin([6, 7, 8, 9]).astype(int)
    )

    out["is_post_monsoon"] = (
        out["month"].isin([10, 11, 12]).astype(int)
    )

    out["month_sin"] = np.sin(
        2 * np.pi * out["month"] / 12
    )
    out["month_cos"] = np.cos(
        2 * np.pi * out["month"] / 12
    )

    out["hour_sin"] = np.sin(
        2 * np.pi * out["hour"] / 24
    )
    out["hour_cos"] = np.cos(
        2 * np.pi * out["hour"] / 24
    )

    out["antecedent_rain_24h"] = (
        out["rainfall_24h"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_3d"] = (
        out["rainfall_3d"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_7d"] = (
        out["rainfall_7d"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_index"] = (
        0.50 * out["antecedent_rain_24h"]
        + 0.30 * (out["antecedent_rain_3d"] / 3)
        + 0.20 * (out["antecedent_rain_7d"] / 7)
    )

    out["short_term_rain_ratio"] = np.where(
        out["rainfall_24h"] > 0,
        out["rainfall_6h"] / out["rainfall_24h"],
        0.0
    )

    out["short_term_rain_ratio"] = (
        out["short_term_rain_ratio"]
        .clip(lower=0, upper=1)
    )

    out["flatness_score"] = (
        1 / (1 + out["slope"])
    )

    out["rain_flat_interaction"] = (
        out["rainfall_24h"] * out["flatness_score"]
    )

    safe_elevation = out["elevation"].clip(lower=0)

    out["rain_low_elevation_interaction"] = (
        out["rainfall_24h"]
        / (1 + safe_elevation / 100)
    )

    out["rain_intensity_3h"] = (
        out["rainfall_3h"] / 3
    )

    out["rain_intensity_6h"] = (
        out["rainfall_6h"] / 6
    )

    if (
        "distance_to_drain" in out.columns
        and out["distance_to_drain"].notna().any()
    ):
        out["drain_proximity"] = (
            1 / (1 + out["distance_to_drain"] / 1000)
        )
        out["rain_drain_interaction"] = (
            out["rainfall_24h"] * out["drain_proximity"]
        )

    if (
        "built_up_proxy" in out.columns
        and out["built_up_proxy"].notna().any()
    ):
        out["urban_rain_interaction"] = (
            out["rainfall_24h"] * out["built_up_proxy"]
        )

    return out


def get_live_terrain(lat, lon):
    """
    Same 0.01-degree 5-point slope method used by the notebook.
    """
    offset = 0.01

    lats = [lat, lat + offset, lat - offset, lat, lat]
    lons = [lon, lon, lon, lon + offset, lon - offset]

    data = get_json(
        OPEN_METEO_ELEVATION_URL,
        {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons))
        }
    )

    z = np.array(data["elevation"], dtype=float)

    elevation = z[0]

    north_south_distance = 2 * offset * 111320
    east_west_distance = (
        2 * offset * 111320 * np.cos(np.radians(lat))
    )

    dz_dy = (z[1] - z[2]) / north_south_distance
    dz_dx = (z[3] - z[4]) / east_west_distance

    slope = np.degrees(
        np.arctan(
            np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        )
    )

    return float(elevation), float(slope)


def build_live_features(lat, lon, state="LIVE", district="LIVE"):
    """
    Recreates the live inference row from the notebook:
    - last 7 days hourly Open-Meteo data
    - rainfall windows
    - elevation/slope
    - exact notebook engineered features
    """
    data = get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "surface_pressure",
            ]),
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
    )

    hourly = pd.DataFrame(data["hourly"])

    hourly["datetime"] = pd.to_datetime(
        hourly["time"]
    )

    now_india = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .tz_localize(None)
        .floor("h")
    )

    hourly = (
        hourly[
            hourly["datetime"] <= now_india
        ]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(hourly) < 168:
        raise ValueError(
            "Less than 7 days of hourly data available."
        )

    hourly["rainfall_1h"] = (
        hourly["rain"]
        .fillna(0)
        .clip(lower=0)
    )

    hourly["rainfall_3h"] = (
        hourly["rainfall_1h"]
        .rolling(3, min_periods=1)
        .sum()
    )

    hourly["rainfall_6h"] = (
        hourly["rainfall_1h"]
        .rolling(6, min_periods=1)
        .sum()
    )

    hourly["rainfall_24h"] = (
        hourly["rainfall_1h"]
        .rolling(24, min_periods=1)
        .sum()
    )

    hourly["rainfall_3d"] = (
        hourly["rainfall_1h"]
        .rolling(72, min_periods=1)
        .sum()
    )

    hourly["rainfall_7d"] = (
        hourly["rainfall_1h"]
        .rolling(168, min_periods=1)
        .sum()
    )

    latest = hourly.iloc[-1]

    elevation, slope = get_live_terrain(
        lat,
        lon
    )

    live_row = pd.DataFrame([{
        "state": state,
        "district": district,
        "datetime": latest["datetime"],
        "temperature": latest["temperature_2m"],
        "humidity": latest["relative_humidity_2m"],
        "wind_speed": latest["wind_speed_10m"],
        "pressure": latest["surface_pressure"],
        "latitude": lat,
        "longitude": lon,
        "rainfall_1h": latest["rainfall_1h"],
        "rainfall_3h": latest["rainfall_3h"],
        "rainfall_6h": latest["rainfall_6h"],
        "rainfall_24h": latest["rainfall_24h"],
        "rainfall_3d": latest["rainfall_3d"],
        "rainfall_7d": latest["rainfall_7d"],
        "elevation": elevation,
        "slope": slope,
    }])

    live_row = engineer_ml_features(live_row)

    live_features = live_row[
        SELECTED_FEATURES
    ].copy()

    return {
        "features": live_features.iloc[0].to_dict(),
        "source_time": str(latest["datetime"]),
        "state": state,
        "district": district,
        "latitude": float(lat),
        "longitude": float(lon),
        "rainfall_summary_mm": {
            "rainfall_1h": float(latest["rainfall_1h"]),
            "rainfall_24h": float(latest["rainfall_24h"]),
            "rainfall_3d": float(latest["rainfall_3d"]),
            "rainfall_7d": float(latest["rainfall_7d"]),
        },
        "terrain": {
            "elevation_m": elevation,
            "slope_deg": slope,
        },
    }
