"""
Smart Multi-Risk Platform - 20-Feature Disaster Vector Builder & Provenance Engine
Constructs the exact 20 training-compatible features required by the trained XGBoost model
with explicit data quality provenance (LIVE, DERIVED, FALLBACK, DATA_INCOMPLETE).
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from backend.data_sources.open_meteo import forecast, air_quality, elevation, flood
from backend.data_sources.usgs import get_recent_earthquake_features
from backend.data_sources.gdacs import get_cyclone_and_volcano_alerts
from backend.data_sources.india_official import get_official_india_evidence

# Exact 20 feature names preserved in training order
FEATURE_ORDER = [
    "temp_2m",
    "precip_1h",
    "rain_24h",
    "rain_48h",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "weather_code",
    "us_aqi",
    "pm2_5",
    "river_discharge",
    "elevation",
    "max_quake_mag",
    "quake_depth",
    "tsunami_flag",
    "gdacs_cyclone",
    "gdacs_volcano",
    "wind_pressure_shear",
    "precipitation_saturation_rate",
    "thermal_particulate_index",
]

AUTO_NAMES = [
    "temp_2m", "precip_1h", "rain_24h", "rain_48h", "wind_speed_10m",
    "wind_gusts_10m", "surface_pressure", "weather_code", "us_aqi", "pm2_5", "elevation"
]

ENGINEERED_NAMES = [
    "wind_pressure_shear",
    "precipitation_saturation_rate",
    "thermal_particulate_index",
]

EXTERNAL_HAZARD_NAMES = [
    "river_discharge", "max_quake_mag", "quake_depth", "tsunami_flag",
    "gdacs_cyclone", "gdacs_volcano"
]

# Backward compatibility alias
MANUAL_OR_EXTERNAL_NAMES = EXTERNAL_HAZARD_NAMES

# -----------------------------------------------------------------------------
# PHASE 5: TRAINING VS LIVE AUDIT SPECIFICATION
# -----------------------------------------------------------------------------
FEATURE_CONTRACT_AUDIT = {
    "temp_2m": {
        "training": {"source": "ERA5 / Open-Meteo", "unit": "°C", "formula": "Identity", "range": "[-10, 55]", "center": 26.5030, "scale": 8.6467},
        "live": {"source": "Open-Meteo Forecast", "unit": "°C", "formula": "Identity", "range": "[-10, 55]"},
        "transformation": "raw_celsius",
    },
    "precip_1h": {
        "training": {"source": "Hourly Rain / Precipitation", "unit": "mm", "formula": "Identity", "range": "[0, 150]", "center": 1.4353, "scale": 27.7197},
        "live": {"source": "Open-Meteo Current Precipitation", "unit": "mm", "formula": "Identity", "range": "[0, 150]"},
        "transformation": "raw_mm",
    },
    "rain_24h": {
        "training": {"source": "Rolling 24h accumulation", "unit": "mm", "formula": "sum(hourly_rain[-24:])", "range": "[0, 400]", "center": 4.1104, "scale": 71.6772},
        "live": {"source": "Open-Meteo 24h Rolling Sum", "unit": "mm", "formula": "sum_tail(24)", "range": "[0, 400]"},
        "transformation": "rolling_24h_sum",
    },
    "rain_48h": {
        "training": {"source": "Rolling 48h accumulation", "unit": "mm", "formula": "sum(hourly_rain[-48:])", "range": "[0, 600]", "center": 7.6571, "scale": 88.2462},
        "live": {"source": "Open-Meteo 48h Rolling Sum", "unit": "mm", "formula": "sum_tail(48)", "range": "[0, 600]"},
        "transformation": "rolling_48h_sum",
    },
    "wind_speed_10m": {
        "training": {"source": "10m wind velocity", "unit": "km/h", "formula": "Identity", "range": "[0, 180]", "center": 22.6458, "scale": 25.7327},
        "live": {"source": "Open-Meteo Wind Speed 10m", "unit": "km/h", "formula": "Identity", "range": "[0, 180]"},
        "transformation": "raw_kmh",
    },
    "wind_gusts_10m": {
        "training": {"source": "10m wind gusts", "unit": "km/h", "formula": "Identity", "range": "[0, 250]", "center": 30.3848, "scale": 35.6755},
        "live": {"source": "Open-Meteo Wind Gusts 10m", "unit": "km/h", "formula": "Identity", "range": "[0, 250]"},
        "transformation": "raw_kmh",
    },
    "surface_pressure": {
        "training": {"source": "Barometric pressure", "unit": "hPa", "formula": "Identity", "range": "[920, 1060]", "center": 1005.9394, "scale": 14.8016},
        "live": {"source": "Open-Meteo Surface Pressure", "unit": "hPa", "formula": "Identity", "range": "[920, 1060]"},
        "transformation": "raw_hpa",
    },
    "weather_code": {
        "training": {"source": "WMO weather code", "unit": "code", "formula": "Integer WMO (0-99)", "range": "[0, 99]", "center": 2.0, "scale": 64.0},
        "live": {"source": "Open-Meteo Weather Code", "unit": "code", "formula": "Integer WMO", "range": "[0, 99]"},
        "transformation": "wmo_integer_code",
    },
    "us_aqi": {
        "training": {"source": "US EPA Air Quality Index", "unit": "index", "formula": "US AQI (0-500)", "range": "[0, 500]", "center": 65.4245, "scale": 79.5552},
        "live": {"source": "Open-Meteo Air Quality", "unit": "index", "formula": "US AQI", "range": "[0, 500]"},
        "transformation": "epa_aqi_index",
    },
    "pm2_5": {
        "training": {"source": "Fine particulate matter", "unit": "µg/m³", "formula": "Identity", "range": "[0, 500]", "center": 24.6863, "scale": 32.0204},
        "live": {"source": "Open-Meteo Air Quality PM2.5", "unit": "µg/m³", "formula": "Identity", "range": "[0, 500]"},
        "transformation": "raw_ugm3",
    },
    "river_discharge": {
        "training": {"source": "GloFAS v4 River discharge", "unit": "m³/s", "formula": "Latest positive reading or baseline", "range": "[0, 30000]", "center": 93.9934, "scale": 186.2681},
        "live": {"source": "Open-Meteo Flood (GloFAS)", "unit": "m³/s", "formula": "Valid positive float or fallback baseline", "range": "[0, 30000]"},
        "transformation": "positive_discharge_or_baseline",
    },
    "elevation": {
        "training": {"source": "Copernicus 90m DEM", "unit": "m", "formula": "Identity", "range": "[-100, 8848]", "center": 227.3461, "scale": 419.0433},
        "live": {"source": "Open-Meteo Elevation API", "unit": "m", "formula": "Identity", "range": "[-100, 8848]"},
        "transformation": "raw_meters",
    },
    "max_quake_mag": {
        "training": {"source": "USGS Earthquake Catalog", "unit": "Richter", "formula": "Max mag in regional window (or 0.0)", "range": "[0, 10]", "center": 0.0, "scale": 1.0},
        "live": {"source": "USGS / NCS India Seismic Feeds", "unit": "Richter", "formula": "Max mag in 30d/500km window (or 0.0)", "range": "[0, 10]"},
        "transformation": "regional_max_mag",
    },
    "quake_depth": {
        "training": {"source": "USGS Earthquake Catalog", "unit": "km", "formula": "Focal depth of max event (or 10km baseline)", "range": "[0, 700]", "center": 10.0, "scale": 1.0},
        "live": {"source": "USGS / NCS India Seismic Feeds", "unit": "km", "formula": "Focal depth of max event (or 10.0)", "range": "[0, 700]"},
        "transformation": "focal_depth_km",
    },
    "tsunami_flag": {
        "training": {"source": "USGS / GDACS Tsunami Warning", "unit": "binary", "formula": "1 if warning active else 0", "range": "[0, 1]", "center": 0.0, "scale": 1.0},
        "live": {"source": "USGS / GDACS Feeds", "unit": "binary", "formula": "1 if warning active else 0", "range": "[0, 1]"},
        "transformation": "binary_flag",
    },
    "gdacs_cyclone": {
        "training": {"source": "GDACS Tropical Cyclone Alert", "unit": "binary", "formula": "1 if active in region else 0", "range": "[0, 1]", "center": 0.0, "scale": 1.0},
        "live": {"source": "GDACS Global Events Feed", "unit": "binary", "formula": "1 if active in region else 0", "range": "[0, 1]"},
        "transformation": "binary_flag",
    },
    "gdacs_volcano": {
        "training": {"source": "GDACS Volcanic Eruption Alert", "unit": "binary", "formula": "1 if active in region else 0", "range": "[0, 1]", "center": 0.0, "scale": 1.0},
        "live": {"source": "GDACS Global Events Feed", "unit": "binary", "formula": "1 if active in region else 0", "range": "[0, 1]"},
        "transformation": "binary_flag",
    },
    "wind_pressure_shear": {
        "training": {"source": "Derived Ratio", "unit": "ratio", "formula": "wind_speed_10m / (surface_pressure + 1e-5)", "range": "[0, 0.25]", "center": 0.0225, "scale": 0.0258},
        "live": {"source": "Derived from live wind and pressure", "unit": "ratio", "formula": "wind_speed_10m / (surface_pressure + 1e-5)", "range": "[0, 0.25]"},
        "transformation": "wind_to_pressure_shear_ratio",
    },
    "precipitation_saturation_rate": {
        "training": {"source": "Derived Ratio", "unit": "ratio", "formula": "precip_1h / (rain_48h + 1.0)", "range": "[0, 2.0]", "center": 0.1100, "scale": 0.2559},
        "live": {"source": "Derived from 1h precip and 48h rain", "unit": "ratio", "formula": "precip_1h / (rain_48h + 1.0)", "range": "[0, 2.0]"},
        "transformation": "hourly_intensity_over_48h_saturation",
    },
    "thermal_particulate_index": {
        "training": {"source": "Derived Index", "unit": "index", "formula": "(pm2_5 * us_aqi) / (max(1.0, temp_2m) + 10.0)", "range": "[0, 1500]", "center": 43.2204, "scale": 106.3606},
        "live": {"source": "Derived from PM2.5, AQI, and temp", "unit": "index", "formula": "(pm2_5 * us_aqi) / (max(1.0, temp_2m) + 10.0)", "range": "[0, 1500]"},
        "transformation": "thermal_particulate_dispersion_index",
    },
}

def get_feature_contract_audit() -> Dict[str, Any]:
    """Returns the complete feature contract audit specification."""
    return FEATURE_CONTRACT_AUDIT


def _last(values, default=None):
    if not values:
        return default
    for v in reversed(values):
        if v is not None:
            return v
    return default


def build_disaster_features(
    lat: float,
    lon: float,
    overrides: Optional[Dict[str, Any]] = None,
    precomputed_sources: Optional[Dict[str, Any]] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the exact 20-feature input vector for the trained XGBoost disaster model.
    Tracks granular per-feature provenance, data source, timestamp, and transformation.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    overrides = overrides or {}
    precomputed = precomputed_sources or {}

    provider_status: Dict[str, str] = {}
    fallback_sources: List[str] = []
    missing_sources: List[str] = []

    # -------------------------------------------------------------------------
    # 1. Weather & Precipitation (Open-Meteo Forecast)
    # -------------------------------------------------------------------------
    weather_quality = "LIVE"
    if "forecast" in precomputed and precomputed["forecast"] is not None:
        w = precomputed["forecast"]
        provider_status["open_meteo_weather"] = "PRECOMPUTED"
    else:
        try:
            w = forecast(lat, lon, past_days=2, forecast_days=1)
            provider_status["open_meteo_weather"] = "LIVE"
        except Exception as exc:
            w = {}
            weather_quality = "FALLBACK"
            provider_status["open_meteo_weather"] = f"FAILED: {exc}"
            fallback_sources.append("open_meteo_weather")

    wh = (w.get("hourly") or {}) if isinstance(w, dict) else {}
    wc = (w.get("current") or {}) if isinstance(w, dict) else {}

    rain = wh.get("rain") or wh.get("precipitation") or []
    def sum_tail(n):
        vals = rain[-n:] if len(rain) >= n else rain
        return float(sum(float(x or 0.0) for x in vals))

    temp_2m = wc.get("temperature_2m") if wc.get("temperature_2m") is not None else _last(wh.get("temperature_2m"), 26.5)
    precip_1h = wc.get("precipitation") if wc.get("precipitation") is not None else _last(wh.get("precipitation"), 0.0)
    rain_24h = sum_tail(24)
    rain_48h = sum_tail(48)
    wind_speed_10m = wc.get("wind_speed_10m") if wc.get("wind_speed_10m") is not None else _last(wh.get("wind_speed_10m"), 15.0)
    wind_gusts_10m = wc.get("wind_gusts_10m") if wc.get("wind_gusts_10m") is not None else _last(wh.get("wind_gusts_10m"), 25.0)
    surface_pressure = wc.get("surface_pressure") if wc.get("surface_pressure") is not None else _last(wh.get("surface_pressure"), 1010.0)
    weather_code = wc.get("weather_code") if wc.get("weather_code") is not None else _last(wh.get("weather_code"), 0)

    # -------------------------------------------------------------------------
    # 2. Air Quality (Open-Meteo Air Quality)
    # -------------------------------------------------------------------------
    aq_quality = "LIVE"
    if "air_quality" in precomputed and precomputed["air_quality"] is not None:
        a = precomputed["air_quality"]
        provider_status["open_meteo_air_quality"] = "PRECOMPUTED"
    else:
        try:
            a = air_quality(lat, lon)
            provider_status["open_meteo_air_quality"] = "LIVE"
        except Exception as exc:
            a = {}
            aq_quality = "FALLBACK"
            provider_status["open_meteo_air_quality"] = f"FAILED: {exc}"
            fallback_sources.append("open_meteo_air_quality")

    ac = (a.get("current") or {}) if isinstance(a, dict) else {}
    us_aqi = ac.get("us_aqi") if ac.get("us_aqi") is not None else 65.0
    pm2_5 = ac.get("pm2_5") if ac.get("pm2_5") is not None else 25.0

    # -------------------------------------------------------------------------
    # 3. Elevation (DEM / Open-Meteo Elevation)
    # -------------------------------------------------------------------------
    elev_quality = "LIVE"
    elev = None
    if "elevation" in precomputed and precomputed["elevation"] is not None:
        elev = precomputed["elevation"]
        provider_status["elevation_dem"] = "PRECOMPUTED"
    elif "terrain" in precomputed and precomputed["terrain"] is not None:
        t_val = precomputed["terrain"]
        val = None
        if isinstance(t_val, (tuple, list)) and len(t_val) >= 1:
            try:
                val = float(t_val[0])
            except (ValueError, TypeError):
                val = None
        elif isinstance(t_val, dict):
            val = t_val.get("elevation_m")
        if val is not None:
            elev = float(val)
            provider_status["elevation_dem"] = "PRECOMPUTED_TERRAIN"

    if elev is None:
        try:
            elev = elevation(lat, lon)
            provider_status["elevation_dem"] = "LIVE"
        except Exception as exc:
            elev = 150.0  # nominal baseline
            elev_quality = "FALLBACK"
            provider_status["elevation_dem"] = f"FAILED: {exc}"
            fallback_sources.append("elevation_dem")

    # -------------------------------------------------------------------------
    # 4. River Discharge (Open-Meteo Flood / GloFAS v4)
    # -------------------------------------------------------------------------
    flood_quality = "LIVE"
    river_discharge_val = None
    flood_info: Dict[str, Any] = {}

    if "river_discharge" in overrides:
        try:
            raw_v = float(overrides["river_discharge"])
            if not (math.isnan(raw_v) or math.isinf(raw_v)):
                river_discharge_val = raw_v
                flood_info = {"status": "available", "source": "override", "river_discharge": raw_v}
                provider_status["glofas_flood"] = "OVERRIDE"
        except (ValueError, TypeError):
            river_discharge_val = None

    if river_discharge_val is None:
        if "flood" in precomputed and precomputed["flood"] is not None:
            p_flood = precomputed["flood"]
            raw_v = p_flood.get("river_discharge") if isinstance(p_flood, dict) else p_flood
            flood_info = p_flood if isinstance(p_flood, dict) else {"status": "available", "river_discharge": raw_v}
            provider_status["glofas_flood"] = "PRECOMPUTED"
            if raw_v is not None:
                river_discharge_val = float(raw_v)
        else:
            try:
                flood_res = flood(lat, lon, return_details=True)
                if isinstance(flood_res, dict):
                    flood_info = flood_res
                    if flood_res.get("river_discharge") is not None:
                        river_discharge_val = float(flood_res["river_discharge"])
                elif flood_res is not None:
                    river_discharge_val = float(flood_res)
                    flood_info = {"status": "available", "river_discharge": river_discharge_val}
                provider_status["glofas_flood"] = "LIVE"
            except Exception as exc:
                flood_info = {"status": "unavailable", "reason": str(exc)}
                provider_status["glofas_flood"] = f"FAILED: {exc}"

    if river_discharge_val is None:
        river_discharge_val = 0.0  # safe physical default (no active river flow reported)
        flood_quality = "FALLBACK"
        fallback_sources.append("glofas_flood")

    # -------------------------------------------------------------------------
    # 5. Earthquake Features (USGS & NCS India)
    # -------------------------------------------------------------------------
    seismic_quality = "LIVE"
    max_quake_mag_val = None
    quake_depth_val = None
    tsunami_flag_val = None

    if "max_quake_mag" in overrides:
        max_quake_mag_val = float(overrides["max_quake_mag"])
    if "quake_depth" in overrides:
        quake_depth_val = float(overrides["quake_depth"])
    if "tsunami_flag" in overrides:
        tsunami_flag_val = int(overrides["tsunami_flag"])

    if any(v is None for v in (max_quake_mag_val, quake_depth_val, tsunami_flag_val)):
        if "usgs" in precomputed and precomputed["usgs"] is not None:
            eq_data = precomputed["usgs"]
            provider_status["usgs_earthquake"] = "PRECOMPUTED"
        else:
            try:
                eq_data = get_recent_earthquake_features(lat, lon)
                provider_status["usgs_earthquake"] = "LIVE"
            except Exception as exc:
                eq_data = {}
                seismic_quality = "FALLBACK"
                provider_status["usgs_earthquake"] = f"FAILED: {exc}"
                fallback_sources.append("usgs_earthquake")

        if max_quake_mag_val is None:
            max_quake_mag_val = float(eq_data.get("max_quake_mag", 0.0) or 0.0)
        if quake_depth_val is None:
            quake_depth_val = float(eq_data.get("quake_depth", 10.0) or 10.0)
        if tsunami_flag_val is None:
            tsunami_flag_val = int(eq_data.get("tsunami_flag", 0) or 0)
    else:
        eq_data = {}

    # -------------------------------------------------------------------------
    # 6. Cyclone & Volcano Features (GDACS)
    # -------------------------------------------------------------------------
    gdacs_quality = "LIVE"
    gdacs_cyclone_val = None
    gdacs_volcano_val = None

    if "gdacs_cyclone" in overrides:
        gdacs_cyclone_val = int(overrides["gdacs_cyclone"])
    if "gdacs_volcano" in overrides:
        gdacs_volcano_val = int(overrides["gdacs_volcano"])

    if gdacs_cyclone_val is None or gdacs_volcano_val is None:
        if "gdacs" in precomputed and isinstance(precomputed["gdacs"], dict):
            gdacs_data = precomputed["gdacs"]
            provider_status["gdacs_feed"] = "PRECOMPUTED"
        else:
            try:
                gdacs_data = get_cyclone_and_volcano_alerts(lat, lon)
                provider_status["gdacs_feed"] = "LIVE"
            except Exception as exc:
                gdacs_data = {}
                gdacs_quality = "FALLBACK"
                provider_status["gdacs_feed"] = f"FAILED: {exc}"
                fallback_sources.append("gdacs_feed")

        if gdacs_cyclone_val is None:
            gdacs_cyclone_val = int(gdacs_data.get("gdacs_cyclone", 0) or 0)
        if gdacs_volcano_val is None:
            gdacs_volcano_val = int(gdacs_data.get("gdacs_volcano", 0) or 0)
    else:
        gdacs_data = {}

    # -------------------------------------------------------------------------
    # 7. Official Indian Evidence (NCS, CWC, IMD/NDMA)
    # -------------------------------------------------------------------------
    manual_alerts = overrides.get("official_alerts") or precomputed.get("official_alerts")
    official_india = get_official_india_evidence(
        lat=lat,
        lon=lon,
        state=state,
        district=district,
        manual_official_alerts=manual_alerts,
        timeout=2,
    )
    provider_status["official_india_evidence"] = "AVAILABLE"

    # -------------------------------------------------------------------------
    # 8. Derived Engineered Features (Training Formulas)
    # -------------------------------------------------------------------------
    # Formula 1: wind_pressure_shear = wind_speed_10m / (surface_pressure + 1e-5)
    wind_pressure_shear = round(float(wind_speed_10m) / (float(surface_pressure) + 1e-5), 6)

    # Formula 2: precipitation_saturation_rate = precip_1h / (rain_48h + 1.0)
    precipitation_saturation_rate = round(float(precip_1h) / (float(rain_48h) + 1.0), 6)

    # Formula 3: thermal_particulate_index = (pm2_5 * us_aqi) / (max(1.0, temp_2m) + 10.0)
    thermal_particulate_index = round((float(pm2_5) * float(us_aqi)) / (max(1.0, float(temp_2m)) + 10.0), 4)

    # Assemble raw 20 features dictionary
    raw_features: Dict[str, float] = {
        "temp_2m": float(temp_2m),
        "precip_1h": float(precip_1h),
        "rain_24h": float(rain_24h),
        "rain_48h": float(rain_48h),
        "wind_speed_10m": float(wind_speed_10m),
        "wind_gusts_10m": float(wind_gusts_10m),
        "surface_pressure": float(surface_pressure),
        "weather_code": float(weather_code),
        "us_aqi": float(us_aqi),
        "pm2_5": float(pm2_5),
        "river_discharge": float(river_discharge_val),
        "elevation": float(elev),
        "max_quake_mag": float(max_quake_mag_val),
        "quake_depth": float(quake_depth_val),
        "tsunami_flag": float(tsunami_flag_val),
        "gdacs_cyclone": float(gdacs_cyclone_val),
        "gdacs_volcano": float(gdacs_volcano_val),
        "wind_pressure_shear": float(wind_pressure_shear),
        "precipitation_saturation_rate": float(precipitation_saturation_rate),
        "thermal_particulate_index": float(thermal_particulate_index),
    }

    # Apply any explicit overrides
    for k, v in overrides.items():
        if k in raw_features and v is not None:
            raw_features[k] = float(v)

    # -------------------------------------------------------------------------
    # 9. Granular Feature Provenance Construction (Phase 4 Contract)
    # -------------------------------------------------------------------------
    features_provenance: Dict[str, Dict[str, Any]] = {}
    for feat_name in FEATURE_ORDER:
        val = raw_features[feat_name]
        spec = FEATURE_CONTRACT_AUDIT.get(feat_name, {})
        
        # Determine source and quality
        if feat_name in ENGINEERED_NAMES:
            src = "Derived Mathematical Indicator"
            q = "DERIVED"
        elif feat_name in ("temp_2m", "precip_1h", "rain_24h", "rain_48h", "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "weather_code"):
            src = "Open-Meteo Weather"
            q = weather_quality
        elif feat_name in ("us_aqi", "pm2_5"):
            src = "Open-Meteo Air Quality"
            q = aq_quality
        elif feat_name == "elevation":
            src = "Copernicus 90m DEM / Open-Meteo"
            q = elev_quality
        elif feat_name == "river_discharge":
            src = "Open-Meteo Flood (GloFAS v4)"
            q = flood_quality
        elif feat_name in ("max_quake_mag", "quake_depth", "tsunami_flag"):
            src = "USGS Earthquake Catalog / NCS India"
            q = seismic_quality
        elif feat_name in ("gdacs_cyclone", "gdacs_volcano"):
            src = "GDACS Live Hazards Feed"
            q = gdacs_quality
        else:
            src = "System Provider"
            q = "LIVE"

        features_provenance[feat_name] = {
            "value": val,
            "source": src,
            "timestamp": now_iso,
            "quality": q,
            "transformation": spec.get("transformation", "identity"),
            "unit": spec.get("live", {}).get("unit", "--"),
        }

    # Evaluate Overall Data Quality Provenance (Phase 18)
    if len(fallback_sources) == 0:
        overall_quality = "FULL_DATA"
    elif len(fallback_sources) <= 2:
        overall_quality = "PARTIAL_DATA"
    elif len(fallback_sources) <= 4:
        overall_quality = "FALLBACK_USED"
    else:
        overall_quality = "DATA_INCOMPLETE"

    missing = [k for k, v in raw_features.items() if v is None]
    invalid = [k for k, v in raw_features.items() if v is not None and (math.isnan(v) or math.isinf(v))]
    missing_external = [k for k in EXTERNAL_HAZARD_NAMES if raw_features.get(k) is None or k in invalid]
    diagnostics = {
        "expected_feature_count": 20,
        "generated_feature_count": len(raw_features),
        "missing_features": missing,
        "invalid_features": invalid,
        "overall_quality": overall_quality,
        "fallback_count": len(fallback_sources),
    }

    # Convenient provider status aliases
    convenient_provider_status = dict(provider_status)
    convenient_provider_status["weather"] = provider_status.get("open_meteo_weather", {"status": "available"})
    convenient_provider_status["air_quality"] = provider_status.get("open_meteo_air_quality", {"status": "available"})
    convenient_provider_status["flood"] = provider_status.get("glofas_flood", {"status": "available"})
    convenient_provider_status["dem"] = provider_status.get("copernicus_dem", {"status": "available"})
    convenient_provider_status["earthquake"] = provider_status.get("usgs_earthquake", {"status": "available"})
    convenient_provider_status["gdacs"] = provider_status.get("gdacs_hazards", {"status": "available"})

    return {
        "features": raw_features,
        "features_provenance": features_provenance,
        "feature_metadata": features_provenance,
        "missing_external_features": missing_external,
        "missing_features": missing,
        "model_feature_quality": overall_quality,
        "provider_status": convenient_provider_status,
        "fallback_sources": fallback_sources,
        "missing_sources": missing_sources,
        "diagnostics": diagnostics,
        "seismic_metadata": eq_data,
        "gdacs_metadata": gdacs_data,
        "flood_metadata": flood_info,
        "official_india": official_india,
        "official_alerts": official_india.get("official_alerts", []),
        "feature_order": FEATURE_ORDER,
        "note": "Complete 20-feature training-compatible disaster input with provenance."
    }
