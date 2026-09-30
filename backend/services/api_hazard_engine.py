"""
Smart Multi-Risk Platform - API-First Multi-Hazard Operational Detection Engine
=============================================================================
Primary Operational Hazard Engine:
Evaluates multiple real-world hazards independently and in parallel using:
1. Official Indian authoritative bulletins (CWC, NCS, IMD, NDMA SACHET)
2. Live provider telemetry (Open-Meteo Weather, GloFAS River Flow, USGS, GDACS, Air Quality, DEM)
3. Historical and recent multi-window telemetry (24h, 3d, 7d, 30d, 90d)
4. Supporting platform model probabilities (Surface Waterlogging, Drainage Congestion)

Real-world hazards are MULTI-LABEL. Multiple hazards can be active simultaneously:
- verified_hazards: List of active hazards (e.g. Flood + Thunderstorm + High Wind)
- hazard_status: Complete status dictionary across all 12 supported hazard categories

Trained XGBoost model is evaluated separately in parallel as a secondary technical ML signal.
"""

from typing import Dict, Any, List, Optional
import math

SEVERITY_LEVELS = ["NONE", "LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"]
SEVERITY_WEIGHTS = {
    "CRITICAL": 5,
    "SEVERE": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
    "NONE": 0,
}

SUPPORTED_HAZARDS = [
    "flood",
    "flash_flood",
    "thunderstorm",
    "high_wind",
    "heat_wave",
    "cold_wave",
    "earthquake",
    "cyclone",
    "drought",
    "air_pollution",
    "landslide",
    "tsunami",
    "volcano",
    "wildfire",
    "avalanche",
]


# Standard Seismic policy for India / South Asia
SEISMIC_THRESHOLDS = {
    "active_recency_days": 3.0,
    "major_mag": 6.5,
    "major_radius_km": 350.0,
    "strong_mag": 5.5,
    "strong_radius_km": 150.0,
    "moderate_mag": 4.5,
    "moderate_radius_km": 50.0,
    "monitoring_days": 7.0,
    "monitoring_radius_km": 250.0,
}


def _normalize_official_severity(raw_sev: str) -> str:
    s = str(raw_sev or "").strip().upper()
    if s in ("RED", "CRITICAL", "EMERGENCY"):
        return "CRITICAL"
    if s in ("ORANGE", "SEVERE"):
        return "SEVERE"
    if s in ("YELLOW", "HIGH", "WARNING"):
        return "HIGH"
    if s in ("WATCH", "MODERATE", "ADVISORY"):
        return "MODERATE"
    if s in ("GREEN", "LOW"):
        return "LOW"
    return "MODERATE"


# ==============================================================================
# 1. EARTHQUAKE DETECTOR
# ==============================================================================
def detect_earthquake(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates seismic hazard using live USGS, NCS India, event magnitude, distance, age, and depth.
    Never activates purely from model output. Requires verified recent local seismic event.
    """
    eq_hist = history_data.get("earthquake") or {}
    seismic_meta = d_built.get("seismic_metadata") or {}
    nearest_event = eq_hist.get("nearest_event") or {}

    # Distance, age, and magnitude
    dist_km = (
        eq_hist.get("nearest_quake_km")
        or nearest_event.get("distance_km")
        or nearest_event.get("dist_km")
        or seismic_meta.get("event_distance_km")
    )
    if dist_km is not None:
        try:
            dist_km = float(dist_km)
        except (ValueError, TypeError):
            dist_km = None

    age_hours = (
        eq_hist.get("nearest_quake_age_hours")
        or (seismic_meta.get("event_age_hours"))
    )
    age_days = (
        eq_hist.get("nearest_quake_age_days")
        or nearest_event.get("age_days")
        or (seismic_meta.get("event_age_days"))
    )
    if age_days is not None:
        try:
            age_days = float(age_days)
            if age_hours is None:
                age_hours = age_days * 24.0
        except (ValueError, TypeError):
            age_days = None

    if age_hours is not None:
        try:
            age_hours = float(age_hours)
            if age_days is None:
                age_days = age_hours / 24.0
        except (ValueError, TypeError):
            age_hours = None

    mag = float(
        features.get("max_quake_mag")
        or nearest_event.get("mag")
        or nearest_event.get("magnitude")
        or seismic_meta.get("max_quake_mag")
        or 0.0
    )
    depth = float(
        features.get("quake_depth")
        or nearest_event.get("depth")
        or seismic_meta.get("quake_depth")
        or 10.0
    )
    tsunami_flag = int(features.get("tsunami_flag") or nearest_event.get("tsunami_flag") or seismic_meta.get("tsunami_flag") or 0)

    # Check official alerts
    ncs_alerts = [a for a in official_alerts if any(k in str(a.get("hazard", "")).lower() for k in ("earthquake", "seismic", "quake"))]
    has_official = len(ncs_alerts) > 0

    sources = ["USGS Earthquake Catalog"]
    if has_official or "ncs" in str(d_built.get("official_india", {})).lower():
        sources.append("NCS India (MoES)")

    # Evidence dictionary
    evidence = {
        "event_magnitude": round(mag, 1),
        "event_depth_km": round(depth, 1),
        "event_distance_km": round(dist_km, 1) if dist_km is not None else None,
        "event_age_hours": round(age_hours, 1) if age_hours is not None else None,
        "event_age_days": round(age_days, 1) if age_days is not None else None,
        "tsunami_flag": tsunami_flag,
        "events_24h": eq_hist.get("events_24h", 0),
        "events_7d": eq_hist.get("events_7d", 0),
        "events_30d": eq_hist.get("events_30d", 0),
        "events_90d": eq_hist.get("events_90d", 0),
        "official_alert_count": len(ncs_alerts),
    }

    if has_official:
        top_a = ncs_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "earthquake",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "source": sources,
            "reason": f"Official NCS seismic bulletin: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    # Evaluate physical relevance policy
    if dist_km is not None and age_days is not None and mag > 0:
        if age_days <= SEISMIC_THRESHOLDS["active_recency_days"]:
            if (mag >= SEISMIC_THRESHOLDS["major_mag"] and dist_km <= SEISMIC_THRESHOLDS["major_radius_km"]) or \
               (mag >= SEISMIC_THRESHOLDS["strong_mag"] and dist_km <= SEISMIC_THRESHOLDS["strong_radius_km"]) or \
               (mag >= SEISMIC_THRESHOLDS["moderate_mag"] and dist_km <= SEISMIC_THRESHOLDS["moderate_radius_km"]):
                sev = "CRITICAL" if mag >= 6.5 else ("SEVERE" if mag >= 5.5 else "HIGH")
                return {
                    "hazard": "earthquake",
                    "active": True,
                    "severity": sev,
                    "confidence_type": "evidence_based",
                    "sources": sources,
                    "source": sources,
                    "reason": f"Active local earthquake: M{mag:.1f} at {dist_km:.0f}km depth {depth:.0f}km ({age_hours:.1f}h ago).",
                    "evidence": evidence,
                }
        elif age_days <= SEISMIC_THRESHOLDS["monitoring_days"]:
            if dist_km <= SEISMIC_THRESHOLDS["monitoring_radius_km"] and mag >= 4.5:
                return {
                    "hazard": "earthquake",
                    "active": False,
                    "severity": "MODERATE",
                    "confidence_type": "evidence_based",
                    "sources": sources,
                    "source": sources,
                    "reason": f"Recent seismic activity (monitoring): M{mag:.1f} at {dist_km:.0f}km ({age_days:.1f}d ago). Not active local emergency.",
                    "evidence": evidence,
                }

    # Nominal / Inactive
    desc = f"nearest event M{mag:.1f} is {dist_km:.0f}km away ({age_days:.1f}d ago)" if (dist_km is not None and age_days is not None) else "no events within 90 days"
    return {
        "hazard": "earthquake",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": f"No active local earthquake hazard ({desc}).",
        "evidence": evidence,
    }


# ==============================================================================
# 2. FLOOD DETECTOR
# ==============================================================================
def detect_flood(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates flood hazard combining:
    - Official CWC / State Disaster Management flood bulletins
    - GloFAS v4 live river discharge & discharge trend
    - Rainfall accumulation (24h, 48h, 3d, 7d, forecast)
    - Surface waterlogging probability (Waterlogging model)
    - Drainage congestion probability (Drainage model)
    """
    flood_hist = history_data.get("flood") or {}
    rain_hist = flood_hist.get("rainfall_accumulation") or {}

    curr_d = float(features.get("river_discharge") or flood_hist.get("current_discharge_m3s") or 0.0)
    peak_d = float(flood_hist.get("peak_discharge_7d_m3s") or flood_hist.get("peak_discharge") or 0.0)
    d_trend = str(flood_hist.get("discharge_trend", "UNKNOWN")).upper()

    rain_24h = float(features.get("rain_24h") or rain_hist.get("rain_24h_mm") or 0.0)
    rain_48h = float(features.get("rain_48h") or 0.0)
    rain_3d = float(rain_hist.get("rain_3d_mm") or 0.0)
    rain_7d = float(rain_hist.get("rain_7d_mm") or 0.0)
    forecast_rain = float(rain_hist.get("forecast_24h_mm") or 0.0)

    wl_p = float(d_built.get("waterlogging_probability") or 0.0)
    dr_p = float(d_built.get("drainage_congestion_probability") or 0.0)

    # Check Sentinel-1 SAR surface water where available
    sar_data = d_built.get("sentinel1_metadata") or d_built.get("sar_surface_water") or {}
    sar_water_change = bool(
        sar_data.get("high_risk_flag")
        or sar_data.get("water_change_detected")
        or (float(sar_data.get("flood_risk_score") or 0.0) >= 0.5)
        or ("water" in str(sar_data.get("sar_water_change_status", "")).lower())
    )

    # Check official flood alerts (CWC)
    cwc_alerts = [a for a in official_alerts if a.get("hazard") == "flood"]
    has_cwc = len(cwc_alerts) > 0

    sources = ["Open-Meteo Flood (GloFAS v4)", "Hydrological Telemetry"]
    if has_cwc:
        sources.insert(0, "Central Water Commission (CWC)")
    if wl_p > 0:
        sources.append("Surface Waterlogging Model")
    if dr_p > 0:
        sources.append("Drainage Congestion Model")
    if sar_water_change:
        sources.append("Sentinel-1 SAR Surface Water")

    evidence = {
        "river_discharge_m3s": round(curr_d, 2),
        "peak_discharge_7d_m3s": round(peak_d, 2),
        "discharge_trend": d_trend,
        "rain_24h_mm": round(rain_24h, 1),
        "rain_48h_mm": round(rain_48h, 1),
        "rain_3d_mm": round(rain_3d, 1),
        "rain_7d_mm": round(rain_7d, 1),
        "forecast_rain_24h_mm": round(forecast_rain, 1),
        "waterlogging_probability": round(wl_p, 4),
        "drainage_congestion_probability": round(dr_p, 4),
        "sentinel1_water_change_detected": sar_water_change,
        "official_cwc_alert_count": len(cwc_alerts),
    }

    # 1. Official CWC Alert Priority
    if has_cwc:
        top_a = cwc_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "flood",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "source": sources,
            "reason": f"Official CWC flood bulletin: {top_a.get('title') or top_a.get('description') or 'Active riverine flood forecast'}.",
            "evidence": evidence,
        }

    # 2. Critical / Severe Inundation
    severe_discharge = (curr_d >= 150.0) or (curr_d >= 60.0 and d_trend == "RISING")
    severe_rainfall = (rain_24h >= 50.0) or (rain_48h >= 80.0) or (rain_3d >= 100.0)
    severe_combo = (
        (curr_d >= 40.0 and rain_24h >= 30.0)
        or (wl_p >= 0.50 and dr_p >= 0.50 and (rain_24h >= 25.0 or curr_d >= 20.0))
        or (sar_water_change and (curr_d >= 40.0 or rain_24h >= 30.0 or (wl_p >= 0.45 and dr_p >= 0.45)))
    )

    if severe_discharge or severe_rainfall or severe_combo:
        reasons = []
        if curr_d >= 60.0:
            reasons.append(f"high river discharge ({curr_d:.1f} m³/s, trend: {d_trend})")
        if rain_24h >= 30.0 or rain_3d >= 80.0:
            reasons.append(f"heavy rainfall ({rain_24h:.1f} mm/24h)")
        if wl_p >= 0.45 and dr_p >= 0.45:
            reasons.append(f"severe surface waterlogging ({wl_p:.1%}) & drainage congestion ({dr_p:.1%})")
        if sar_water_change:
            reasons.append("satellite radar surface water expansion (Sentinel-1 SAR)")
        reason_str = ", ".join(reasons) if reasons else f"discharge {curr_d:.1f} m³/s"
        sev = "CRITICAL" if (curr_d >= 200.0 or rain_24h >= 75.0) else "HIGH"
        return {
            "hazard": "flood",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Active flood hazard corroborated by {reason_str}.",
            "evidence": evidence,
        }

    # 3. Moderate Flood Risk
    moderate_discharge = (curr_d >= 30.0) or (peak_d >= 50.0 and d_trend == "RISING")
    moderate_rainfall = (rain_24h >= 20.0) or (forecast_rain >= 25.0) or (rain_7d >= 60.0)
    moderate_combo = (
        (wl_p >= 0.35 or dr_p >= 0.40) and (rain_24h >= 10.0 or curr_d >= 10.0)
        or (sar_water_change and (rain_24h >= 10.0 or curr_d >= 10.0 or wl_p >= 0.25))
    )

    if moderate_discharge or moderate_rainfall or moderate_combo:
        reasons = []
        if curr_d >= 30.0 or peak_d >= 50.0:
            reasons.append(f"elevated discharge ({curr_d:.1f} m³/s)")
        if rain_24h >= 20.0 or forecast_rain >= 25.0:
            reasons.append(f"accumulating rain ({rain_24h:.1f} mm/24h, forecast: {forecast_rain:.1f} mm)")
        if wl_p >= 0.30:
            reasons.append(f"waterlogging risk ({wl_p:.1%})")
        if sar_water_change:
            reasons.append("Sentinel-1 SAR inundation indicators")
        reason_str = ", ".join(reasons) if reasons else "hydrological indicators"
        return {
            "hazard": "flood",
            "active": True,
            "severity": "MODERATE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Elevated flood risk supported by {reason_str}.",
            "evidence": evidence,
        }

    # 4. Inactive
    return {
        "hazard": "flood",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": f"River discharge ({curr_d:.1f} m³/s) and rainfall ({rain_24h:.1f} mm/24h) remain within nominal bounds.",
        "evidence": evidence,
    }


# ==============================================================================
# 3. CYCLONE DETECTOR
# ==============================================================================
def detect_cyclone(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates tropical cyclone hazard using GDACS active cyclone feed, IMD alerts,
    wind speed, gusts, and barometric pressure.
    """
    gdacs_meta = d_built.get("gdacs_metadata") or {}
    active_gdacs = (features.get("gdacs_cyclone") == 1) or (gdacs_meta.get("gdacs_cyclone") == 1)
    cyclone_info = gdacs_meta.get("active_cyclone_details") or {}

    wind_speed = float(features.get("wind_speed_10m") or 0.0)
    wind_gusts = float(features.get("wind_gusts_10m") or 0.0)
    pressure = float(features.get("surface_pressure") or 1013.25)

    imd_cyclone_alerts = [a for a in official_alerts if a.get("hazard") == "cyclone"]
    has_imd = len(imd_cyclone_alerts) > 0

    sources = ["GDACS Tropical Cyclone Feed"]
    if has_imd:
        sources.insert(0, "IMD Cyclone Warning Division")
    sources.append("Atmospheric Telemetry")

    dist = cyclone_info.get("distance_km") or gdacs_meta.get("distance_km")
    name = cyclone_info.get("name") or gdacs_meta.get("name") or gdacs_meta.get("event_type") or "Tropical Cyclone"

    evidence = {
        "active_gdacs_cyclone": active_gdacs,
        "cyclone_name": name,
        "distance_km": dist,
        "wind_speed_kmh": round(wind_speed, 1),
        "wind_gusts_kmh": round(wind_gusts, 1),
        "surface_pressure_hpa": round(pressure, 1),
        "official_cyclone_alert_count": len(imd_cyclone_alerts),
    }

    if has_imd:
        top_a = imd_cyclone_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "SEVERE"))
        return {
            "hazard": "cyclone",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "source": sources,
            "reason": f"Official IMD cyclone alert: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    if active_gdacs:
        dist_str = f"at {dist:.0f}km" if dist is not None else "in regional zone"
        sev = "CRITICAL" if (dist is not None and dist < 250.0) or (wind_gusts >= 80.0) else "HIGH"
        return {
            "hazard": "cyclone",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Active tropical cyclone ({name}) verified {dist_str} via GDACS.",
            "evidence": evidence,
        }

    # Severe gale / barometric depression check
    if wind_gusts >= 80.0 or (wind_speed >= 60.0 and pressure <= 985.0):
        return {
            "hazard": "cyclone",
            "active": True,
            "severity": "HIGH",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Severe cyclonic pressure depression ({pressure:.1f} hPa) and gale gusts ({wind_gusts:.1f} km/h).",
            "evidence": evidence,
        }

    return {
        "hazard": "cyclone",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": "No active tropical cyclone alert within regional proximity.",
        "evidence": evidence,
    }


# ==============================================================================
# 4. DROUGHT DETECTOR
# ==============================================================================
def detect_drought(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates agricultural and meteorological drought using 7-day and 30-day cumulative rainfall,
    rainfall deficit percentage, and temperature stress. Never declares drought from 1 dry day.
    """
    drought_hist = history_data.get("drought") or {}
    rain_hist = (
        (history_data.get("flood") or {}).get("rainfall_accumulation")
        or history_data.get("rainfall")
        or {}
    )

    rain_7d = float(
        features.get("rain_7d")
        or rain_hist.get("rain_7d_mm")
        or rain_hist.get("accumulation_7d_mm")
        or 0.0
    )
    rain_30d = float(
        features.get("rain_30d")
        or rain_hist.get("rain_30d_mm")
        or rain_hist.get("accumulation_30d_mm")
        or drought_hist.get("accumulation_30d_mm")
        or (rain_7d * 4.0 if rain_7d > 0 else 0.0)
    )
    deficit_pct = float(drought_hist.get("rainfall_deficit_pct") or 0.0)
    temp_2m = float(features.get("temp_2m") or 0.0)
    soil_moist = float(features.get("soil_moisture") or (history_data.get("drought") or {}).get("soil_moisture") or d_built.get("soil_moisture") or 0.25)

    is_deficit = (
        drought_hist.get("is_drought_indicated", False)
        or drought_hist.get("deficit") is True
        or "deficit" in str(drought_hist.get("status", "")).lower()
        or deficit_pct <= -50.0
        or (rain_30d < 20.0 and rain_7d < 5.0)
        or drought_hist.get("status") in ("SEVERE_DEFICIT", "MODERATE_DEFICIT", "DROUGHT", "RAINFALL_DEFICIT")
    )
    if rain_7d >= 20.0:
        is_deficit = False

    sources = ["Precipitation Anomaly Baseline", "Agricultural Hydro-Climatic Feed"]
    if soil_moist < 0.20:
        sources.append("Root-Zone Soil Moisture Sensors")

    evidence = {
        "rain_7d_mm": round(rain_7d, 1),
        "rain_30d_mm": round(rain_30d, 1),
        "rainfall_deficit_pct": round(deficit_pct, 1),
        "temp_2m_c": round(temp_2m, 1),
        "soil_moisture": round(soil_moist, 3),
        "drought_status": drought_hist.get("status", "NORMAL"),
    }

    # Official drought alerts if available
    drought_alerts = [a for a in official_alerts if "drought" in str(a.get("hazard", "")).lower()]
    if drought_alerts:
        top_a = drought_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        sources.insert(0, "Ministry of Agriculture / State Drought Relief")
        return {
            "hazard": "drought",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "source": sources,
            "reason": f"Official drought advisory: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    # Prolonged severe deficit over 30 days
    if is_deficit and (rain_7d < 5.0 or deficit_pct <= -60.0) and (temp_2m >= 30.0 or soil_moist < 0.18):
        sev = "HIGH" if (deficit_pct <= -75.0 or temp_2m >= 38.0 or soil_moist < 0.12) else "MODERATE"
        return {
            "hazard": "drought",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Prolonged multi-week rainfall deficit ({deficit_pct:.0f}% deficit, {rain_30d:.1f}mm in 30d) under elevated temperatures ({temp_2m:.1f}°C).",
            "evidence": evidence,
        }

    return {
        "hazard": "drought",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": f"Recent rainfall ({rain_7d:.1f} mm in 7d, {rain_30d:.1f} mm in 30d) does not indicate active meteorological drought.",
        "evidence": evidence,
    }


# ==============================================================================
# 5. HEAT WAVE DETECTOR
# ==============================================================================
def detect_heat_wave(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates heat wave hazard according to IMD India criteria (Plains >= 40°C, Hills >= 30°C).
    """
    temp_2m = float(features.get("temp_2m") or 0.0)
    elev = float(features.get("elevation") or 0.0)
    is_hill = elev > 500.0

    imd_heat_alerts = [a for a in official_alerts if "heat" in str(a.get("hazard", "")).lower()]
    has_imd = len(imd_heat_alerts) > 0

    sources = ["Open-Meteo Thermal Sensors"]
    if has_imd:
        sources.insert(0, "IMD Weather Warning")

    evidence = {
        "temp_2m_c": round(temp_2m, 1),
        "elevation_m": round(elev, 0),
        "is_hill_region": is_hill,
        "official_heat_alert_count": len(imd_heat_alerts),
    }

    if has_imd:
        top_a = imd_heat_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "heat_wave",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "reason": f"Official IMD Heatwave bulletin: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    # Physical criteria
    threshold_severe = 45.0 if not is_hill else 35.0
    threshold_mod = 40.0 if not is_hill else 30.0

    if temp_2m >= threshold_severe:
        return {
            "hazard": "heat_wave",
            "active": True,
            "severity": "CRITICAL" if temp_2m >= 47.0 else "SEVERE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Severe heatwave conditions: live ambient temperature {temp_2m:.1f}°C exceeds threshold ({threshold_severe}°C).",
            "evidence": evidence,
        }
    elif temp_2m >= threshold_mod:
        sev = "HIGH" if temp_2m >= 42.0 else "MODERATE"
        return {
            "hazard": "heat_wave",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Elevated thermal stress: live temperature {temp_2m:.1f}°C exceeds threshold ({threshold_mod}°C).",
            "evidence": evidence,
        }

    return {
        "hazard": "heat_wave",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": f"Ambient temperature {temp_2m:.1f}°C is within safe thermal bounds.",
        "evidence": evidence,
    }


# ==============================================================================
# 6. COLD WAVE DETECTOR
# ==============================================================================
def detect_cold_wave(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates cold wave hazard according to IMD India criteria (Plains <= 4°C, Hills <= 0°C).
    """
    temp_2m = float(features.get("temp_2m") or 25.0)
    elev = float(features.get("elevation") or 0.0)
    is_hill = elev > 500.0

    imd_cold_alerts = [a for a in official_alerts if "cold" in str(a.get("hazard", "")).lower()]
    has_imd = len(imd_cold_alerts) > 0

    sources = ["Open-Meteo Thermal Sensors"]
    if has_imd:
        sources.insert(0, "IMD Weather Warning")

    evidence = {
        "temp_2m_c": round(temp_2m, 1),
        "elevation_m": round(elev, 0),
        "is_hill_region": is_hill,
        "official_cold_alert_count": len(imd_cold_alerts),
    }

    if has_imd:
        top_a = imd_cold_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "cold_wave",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "source": sources,
            "reason": f"Official IMD Cold Wave bulletin: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    threshold_severe = 2.0 if not is_hill else -2.0
    threshold_high = 3.5 if not is_hill else -0.5
    threshold_mod = 4.5 if not is_hill else 0.5

    if temp_2m <= threshold_severe:
        sev = "CRITICAL" if temp_2m <= 0.0 else "SEVERE"
        return {
            "hazard": "cold_wave",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Severe coldwave conditions: temperature {temp_2m:.1f}°C below freeze/cold threshold ({threshold_severe}°C).",
            "evidence": evidence,
        }
    elif temp_2m <= threshold_high:
        return {
            "hazard": "cold_wave",
            "active": True,
            "severity": "HIGH",
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Cold wave conditions: temperature {temp_2m:.1f}°C satisfies IMD cold wave threshold ({threshold_high}°C).",
            "evidence": evidence,
        }
    elif temp_2m <= threshold_mod:
        return {
            "hazard": "cold_wave",
            "active": True,
            "severity": "MODERATE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Cold wave conditions: temperature {temp_2m:.1f}°C below standard IMD threshold ({threshold_mod}°C).",
            "evidence": evidence,
        }

    return {
        "hazard": "cold_wave",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": f"Ambient temperature {temp_2m:.1f}°C is above cold wave thresholds.",
        "evidence": evidence,
    }


# ==============================================================================
# 7. THUNDERSTORM DETECTOR
# ==============================================================================
def detect_thunderstorm(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates thunderstorm hazard using WMO convective weather codes (95, 96, 99),
    precipitation intensity, wind gusts, and official IMD nowcasts.
    """
    w_code = int(features.get("weather_code") or 0)
    precip_1h = float(features.get("precip_1h") or 0.0)
    gusts = float(features.get("wind_gusts_10m") or 0.0)

    # WMO codes: 95 (Thunderstorm slight/moderate), 96 (with slight hail), 99 (with heavy hail)
    is_wmo_storm = w_code in (95, 96, 99, 17, 29)
    storm_alerts = [a for a in official_alerts if any(t in str(a.get("hazard", "")).lower() for t in ("thunder", "lightning", "storm"))]
    has_alert = len(storm_alerts) > 0

    sources = ["Open-Meteo Convective Telemetry"]
    if has_alert:
        sources.insert(0, "IMD Nowcast / Weather Warning")

    evidence = {
        "weather_code": w_code,
        "precip_1h_mm": round(precip_1h, 1),
        "wind_gusts_kmh": round(gusts, 1),
        "official_storm_alert_count": len(storm_alerts),
    }

    if has_alert:
        top_a = storm_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "MODERATE"))
        return {
            "hazard": "thunderstorm",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "reason": f"Official IMD convective thunderstorm alert: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    if is_wmo_storm or (precip_1h >= 10.0 and gusts >= 50.0):
        sev = "SEVERE" if (w_code == 99 or gusts >= 75.0) else ("HIGH" if (w_code == 96 or gusts >= 60.0) else "MODERATE")
        hail_note = " with hail" if w_code in (96, 99) else ""
        return {
            "hazard": "thunderstorm",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Active convective thunderstorm{hail_note} observed (WMO code {w_code}, gusts {gusts:.0f} km/h).",
            "evidence": evidence,
        }

    return {
        "hazard": "thunderstorm",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": f"No convective storm activity detected (weather code {w_code}).",
        "evidence": evidence,
    }


# ==============================================================================
# 8. HIGH WIND DETECTOR
# ==============================================================================
def detect_high_wind(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates strong wind and gale hazard independent of tropical cyclones.
    """
    wind_speed = float(features.get("wind_speed_10m") or 0.0)
    wind_gusts = float(features.get("wind_gusts_10m") or 0.0)

    wind_alerts = [a for a in official_alerts if "wind" in str(a.get("hazard", "")).lower() and "cyclone" not in str(a.get("hazard", "")).lower()]
    has_alert = len(wind_alerts) > 0

    sources = ["Open-Meteo Anemometers"]
    if has_alert:
        sources.insert(0, "IMD Wind Advisory")

    evidence = {
        "wind_speed_kmh": round(wind_speed, 1),
        "wind_gusts_kmh": round(wind_gusts, 1),
        "official_wind_alert_count": len(wind_alerts),
    }

    if has_alert:
        top_a = wind_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "MODERATE"))
        return {
            "hazard": "high_wind",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "reason": f"Official wind warning: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    if wind_gusts >= 75.0 or wind_speed >= 55.0:
        return {
            "hazard": "high_wind",
            "active": True,
            "severity": "SEVERE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Dangerous gale-force winds observed: live gusts of {wind_gusts:.1f} km/h (speed {wind_speed:.1f} km/h).",
            "evidence": evidence,
        }
    elif wind_gusts >= 55.0 or wind_speed >= 40.0:
        return {
            "hazard": "high_wind",
            "active": True,
            "severity": "HIGH",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"High wind hazard: live gusts of {wind_gusts:.1f} km/h.",
            "evidence": evidence,
        }
    elif wind_gusts >= 42.0:
        return {
            "hazard": "high_wind",
            "active": True,
            "severity": "MODERATE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Breezy to squally wind conditions (gusts {wind_gusts:.1f} km/h).",
            "evidence": evidence,
        }

    return {
        "hazard": "high_wind",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": f"Wind speeds ({wind_speed:.1f} km/h, gusts {wind_gusts:.1f} km/h) are within safe ranges.",
        "evidence": evidence,
    }


# ==============================================================================
# 9. TSUNAMI DETECTOR
# ==============================================================================
def detect_tsunami(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates tsunami hazard. Requires explicit seismic tsunami flag or INCOIS/GDACS oceanic advisory.
    Never activates purely from ML argmax.
    """
    tsunami_flag = int(features.get("tsunami_flag") or 0)
    seismic_meta = d_built.get("seismic_metadata") or {}
    tsunami_alerts = [a for a in official_alerts if "tsunami" in str(a.get("hazard", "")).lower()]
    has_official = len(tsunami_alerts) > 0

    sources = ["USGS / INCOIS Oceanic Network"]
    evidence = {
        "tsunami_flag": tsunami_flag,
        "official_tsunami_alerts": len(tsunami_alerts),
    }

    if has_official:
        top_a = tsunami_alerts[0]
        return {
            "hazard": "tsunami",
            "active": True,
            "severity": "CRITICAL",
            "confidence_type": "official_bulletin",
            "sources": ["INCOIS Official Tsunami Advisory"],
            "reason": f"Official INCOIS tsunami bulletin: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    if tsunami_flag == 1 or seismic_meta.get("tsunami_flag") == 1:
        return {
            "hazard": "tsunami",
            "active": True,
            "severity": "CRITICAL",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": "Active oceanic tsunami watch/warning generated by regional undersea seismic rupture.",
            "evidence": evidence,
        }

    return {
        "hazard": "tsunami",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": "No active oceanic tsunami alerts or undersea seismic triggers.",
        "evidence": evidence,
    }


# ==============================================================================
# 10. VOLCANO DETECTOR
# ==============================================================================
def detect_volcano(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates volcanic eruption and ash plume hazards from GDACS live feeds.
    Mainland Indian locations never activate without verified regional ash plume evidence.
    """
    gdacs_meta = d_built.get("gdacs_metadata") or {}
    active_gdacs = (features.get("gdacs_volcano") == 1) or (gdacs_meta.get("gdacs_volcano") == 1)
    volcano_info = gdacs_meta.get("active_volcano_details") or {}
    dist = volcano_info.get("distance_km") or gdacs_meta.get("distance_km")
    name = volcano_info.get("name") or gdacs_meta.get("name") or gdacs_meta.get("event_type") or "Volcanic Eruption"

    sources = ["GDACS Global Volcano Watch"]
    evidence = {
        "active_gdacs_volcano": active_gdacs,
        "volcano_details": volcano_info,
        "distance_km": dist,
        "volcano_name": name,
    }

    if active_gdacs:
        dist_str = f"at {dist:.0f}km" if dist is not None else "in regional impact radius"
        return {
            "hazard": "volcano",
            "active": True,
            "severity": "CRITICAL" if (dist is not None and dist < 150) else "HIGH",
            "confidence_type": "evidence_based",
            "sources": sources,
            "source": sources,
            "reason": f"Active volcanic eruption / ash advisory verified for {name} ({dist_str}) via GDACS.",
            "evidence": evidence,
        }

    return {
        "hazard": "volcano",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": "No active volcanic eruptions or ash plume advisories in regional proximity.",
        "evidence": evidence,
    }


# ==============================================================================
# 11. AIR POLLUTION DETECTOR
# ==============================================================================
def detect_air_pollution(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates hazardous ambient particulate matter and air pollution using US AQI and PM2.5.
    """
    aqi = float(features.get("us_aqi") or 0.0)
    pm25 = float(features.get("pm2_5") or 0.0)

    sources = ["Open-Meteo Atmospheric Chemistry / CPCB Criteria"]
    evidence = {
        "us_aqi": round(aqi, 0),
        "pm2_5_ugm3": round(pm25, 1),
    }

    if aqi >= 300.0 or pm25 >= 250.0:
        return {
            "hazard": "air_pollution",
            "active": True,
            "severity": "CRITICAL",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Severe/Hazardous air quality emergency: AQI {aqi:.0f}, PM2.5 {pm25:.1f} µg/m³ (exceeds emergency threshold).",
            "evidence": evidence,
        }
    elif aqi >= 200.0 or pm25 >= 120.0:
        return {
            "hazard": "air_pollution",
            "active": True,
            "severity": "HIGH",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Very unhealthy air pollution: AQI {aqi:.0f}, PM2.5 {pm25:.1f} µg/m³.",
            "evidence": evidence,
        }
    elif aqi >= 120.0 or pm25 >= 60.0:
        return {
            "hazard": "air_pollution",
            "active": True,
            "severity": "MODERATE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Unhealthy air quality for sensitive groups: AQI {aqi:.0f}, PM2.5 {pm25:.1f} µg/m³.",
            "evidence": evidence,
        }

    return {
        "hazard": "air_pollution",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": f"Air quality within moderate or satisfactory ranges (AQI {aqi:.0f}, PM2.5 {pm25:.1f} µg/m³).",
        "evidence": evidence,
    }


# ==============================================================================
# 12. LANDSLIDE DETECTOR
# ==============================================================================
def detect_landslide(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates landslide hazard combining steep slope (>15°) and antecedent rainfall (24h/3d)
    or official Geological Survey of India (GSI) alerts.
    """
    elev = float(features.get("elevation") or 0.0)
    rain_24h = float(features.get("rain_24h") or 0.0)
    rain_hist = (history_data.get("flood") or {}).get("rainfall_accumulation") or {}
    rain_3d = float(rain_hist.get("rain_3d_mm") or 0.0)
    slope = float(d_built.get("slope") or (18.0 if elev > 800.0 else 2.0))

    landslide_alerts = [a for a in official_alerts if "landslide" in str(a.get("hazard", "")).lower()]
    has_official = len(landslide_alerts) > 0

    sources = ["Copernicus DEM Terrain Topography", "Hydrological Saturation Telemetry"]
    if has_official:
        sources.insert(0, "Geological Survey of India (GSI)")

    evidence = {
        "elevation_m": round(elev, 0),
        "slope_deg": round(slope, 1),
        "rain_24h_mm": round(rain_24h, 1),
        "rain_3d_mm": round(rain_3d, 1),
        "official_landslide_alerts": len(landslide_alerts),
    }

    if has_official:
        top_a = landslide_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "landslide",
            "active": True,
            "severity": sev,
            "confidence_type": "official_bulletin",
            "sources": sources,
            "reason": f"Official GSI landslide warning: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    # Mountainous terrain with heavy torrential rainfall
    is_mountainous = elev >= 400.0 or slope >= 15.0
    if is_mountainous and (rain_24h >= 60.0 or rain_3d >= 120.0):
        sev = "CRITICAL" if (rain_24h >= 100.0 or rain_3d >= 180.0) else "HIGH"
        return {
            "hazard": "landslide",
            "active": True,
            "severity": sev,
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"High landslide risk: steep slope ({slope:.1f}°) with torrential rainfall ({rain_24h:.1f} mm/24h, {rain_3d:.1f} mm/3d).",
            "evidence": evidence,
        }
    elif is_mountainous and (rain_24h >= 35.0 or rain_3d >= 70.0):
        return {
            "hazard": "landslide",
            "active": True,
            "severity": "MODERATE",
            "confidence_type": "evidence_based",
            "sources": sources,
            "reason": f"Moderate slope instability risk from accumulated rainfall ({rain_24h:.1f} mm/24h) on elevated terrain ({elev:.0f}m).",
            "evidence": evidence,
        }

    return {
        "hazard": "landslide",
        "active": False,
        "severity": "NONE",
        "confidence_type": "evidence_based",
        "sources": sources,
        "reason": "Terrain slope and antecedent precipitation do not trigger slope instability.",
        "evidence": evidence,
    }


# ==============================================================================
# 13. WILDFIRE / FOREST FIRE DETECTOR (NASA FIRMS VIIRS)
# ==============================================================================
def detect_wildfire(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates wildfire and forest fire risk using NASA FIRMS VIIRS (NOAA-20/21 NRT)
    satellite thermal anomaly detections and official Forest Department / SACHET advisories.
    Requires locally relevant recent detections (<= 50km, <= 48h).
    """
    firms_data = d_built.get("firms_fire") or features.get("firms_fire") or {}

    # Check official alerts for fire/wildfire
    fire_alerts = [
        a for a in official_alerts
        if any(kw in str(a.get("hazard", "")).lower() or kw in str(a.get("title", "")).lower() for kw in ("wildfire", "forest_fire", "forest fire"))
    ]
    has_official_fire = len(fire_alerts) > 0

    # Provider failure handling
    if firms_data.get("missing_provider") or firms_data.get("status") == "unavailable":
        if has_official_fire:
            top_a = fire_alerts[0]
            sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
            return {
                "hazard": "wildfire",
                "active": True,
                "severity": sev,
                "confidence_type": "official_warning",
                "sources": ["NDMA SACHET", "State Forest Department"],
                "source": ["NDMA SACHET", "State Forest Department"],
                "reason": f"Official forest fire advisory active: {top_a.get('title') or top_a.get('description')}.",
                "evidence": {"official_alerts": fire_alerts, "provider_status": "unavailable"},
            }
        return {
            "hazard": "wildfire",
            "active": False,
            "severity": "NONE",
            "confidence_type": "insufficient_data",
            "sources": ["NASA FIRMS (Unavailable)"],
            "source": ["NASA FIRMS (Unavailable)"],
            "reason": "NASA FIRMS active fire satellite telemetry is currently unavailable.",
            "evidence": {
                "provider_status": "unavailable",
                "missing_provider": True,
                "reason": firms_data.get("reason", "FIRMS provider unreachable."),
            },
        }

    hotspots = firms_data.get("hotspots", [])
    nearest_km = firms_data.get("nearest_fire_distance_km")
    det_10km = firms_data.get("detections_10km", 0)
    det_25km = firms_data.get("detections_25km", 0)
    det_50km = firms_data.get("detections_50km", 0)
    det_100km = firms_data.get("detections_100km", 0)
    max_frp = float(firms_data.get("max_frp") or 0.0)
    age_hrs = firms_data.get("latest_detection_age_hours")

    sources = ["NASA FIRMS VIIRS NOAA-21", "NASA FIRMS VIIRS NOAA-20"]
    if has_official_fire:
        sources.insert(0, "NDMA SACHET / Forest Survey of India")

    evidence = {
        "nearest_fire_distance_km": nearest_km,
        "detections_10km": det_10km,
        "detections_25km": det_25km,
        "detections_50km": det_50km,
        "detections_100km": det_100km,
        "max_frp": round(max_frp, 1),
        "latest_detection_age_hours": age_hrs,
        "hotspots_count": len(hotspots),
        "hotspots": hotspots[:25],
    }

    # If official alert is active
    if has_official_fire:
        top_a = fire_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        return {
            "hazard": "wildfire",
            "active": True,
            "severity": sev,
            "confidence_type": "official_warning",
            "sources": sources,
            "source": sources,
            "reason": f"Official wildfire bulletin active: {top_a.get('title') or top_a.get('description')}",
            "evidence": evidence,
        }

    # 1. No detections or all detections older than 48 hours
    if not hotspots or nearest_km is None or (age_hrs is not None and age_hrs > 48.0):
        return {
            "hazard": "wildfire",
            "active": False,
            "severity": "NONE",
            "confidence_type": "satellite_observed",
            "sources": sources,
            "source": sources,
            "reason": "No recent active fire thermal anomalies detected by satellite.",
            "evidence": evidence,
        }

    # 2. Distant detections (> 50 km) ignored as local hazard
    if nearest_km > 50.0:
        return {
            "hazard": "wildfire",
            "active": False,
            "severity": "NONE",
            "confidence_type": "satellite_observed",
            "sources": sources,
            "source": sources,
            "reason": f"Distant thermal anomalies detected ({nearest_km:.1f} km away) beyond localized hazard threshold.",
            "evidence": evidence,
        }

    # 3. Active local detections (<= 50 km and <= 48 hours)
    if nearest_km <= 10.0:
        sev = "CRITICAL" if (max_frp >= 50.0 or det_10km >= 3) else ("SEVERE" if max_frp >= 25.0 else "HIGH")
        return {
            "hazard": "wildfire",
            "active": True,
            "severity": sev,
            "confidence_type": "satellite_observed",
            "sources": sources,
            "source": sources,
            "reason": f"{det_10km} active fire detection(s) within {nearest_km:.1f} km (Peak FRP: {max_frp:.1f} MW, Age: {age_hrs:.1f}h).",
            "evidence": evidence,
        }
    elif nearest_km <= 25.0:
        sev = "HIGH" if (max_frp >= 35.0 or det_25km >= 3) else "MODERATE"
        return {
            "hazard": "wildfire",
            "active": True,
            "severity": sev,
            "confidence_type": "satellite_observed",
            "sources": sources,
            "source": sources,
            "reason": f"{det_25km} active fire detection(s) within {nearest_km:.1f} km (Peak FRP: {max_frp:.1f} MW).",
            "evidence": evidence,
        }
    else:  # 25 to 50 km
        if max_frp >= 50.0:
            return {
                "hazard": "wildfire",
                "active": True,
                "severity": "MODERATE",
                "confidence_type": "satellite_observed",
                "sources": sources,
                "source": sources,
                "reason": f"Intense regional thermal anomaly cluster ({nearest_km:.1f} km, Peak FRP: {max_frp:.1f} MW).",
                "evidence": evidence,
            }
        return {
            "hazard": "wildfire",
            "active": False,
            "severity": "LOW",
            "confidence_type": "satellite_observed",
            "sources": sources,
            "source": sources,
            "reason": f"Active thermal detections monitored at safe distance ({nearest_km:.1f} km).",
            "evidence": evidence,
        }


# ==============================================================================
# 14. AVALANCHE DETECTOR (NDMA SACHET / DGRE ALERTS + TOPOGRAPHY)
# ==============================================================================
def detect_avalanche(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates snow avalanche hazard according to official NDMA SACHET / DGRE bulletins
    supported by high-altitude terrain elevation, slope gradient, and ambient temperature.
    Environmental conditions alone never trigger an official avalanche alert without official corroboration.
    """
    elev = float(features.get("elevation") or d_built.get("elevation") or 0.0)
    slope = float(features.get("slope") or d_built.get("slope") or 0.0)
    temp_2m = float(features.get("temp_2m") or 0.0)

    # SACHET provider status check
    sachet_status = d_built.get("provider_status", {}).get("sachet") or d_built.get("sachet_status")

    # Filter official avalanche bulletins
    avalanche_alerts = [
        a for a in official_alerts
        if "avalanche" in str(a.get("hazard", "")).lower() or "dgre" in str(a.get("source", "")).lower()
    ]
    has_official_avalanche = len(avalanche_alerts) > 0

    sources = ["NDMA SACHET", "DGRE (Defence Geoinformatics Research Establishment)"]

    evidence = {
        "elevation_m": round(elev, 0),
        "slope_deg": round(slope, 1),
        "temp_2m_c": round(temp_2m, 1),
        "official_avalanche_alert_count": len(avalanche_alerts),
    }

    if sachet_status == "unavailable" and not has_official_avalanche:
        return {
            "hazard": "avalanche",
            "active": False,
            "severity": "NONE",
            "confidence_type": "insufficient_data",
            "sources": ["NDMA SACHET (Unavailable)"],
            "source": ["NDMA SACHET (Unavailable)"],
            "reason": "NDMA SACHET / DGRE avalanche warning feed currently unavailable.",
            "evidence": {**evidence, "provider_status": "unavailable", "missing_provider": True},
        }

    if has_official_avalanche:
        top_a = avalanche_alerts[0]
        sev = _normalize_official_severity(top_a.get("severity", "HIGH"))
        evidence["effective"] = top_a.get("effective") or top_a.get("issued_at")
        evidence["expires"] = top_a.get("expires") or top_a.get("expires_at")
        evidence["official_area"] = top_a.get("location")
        return {
            "hazard": "avalanche",
            "active": True,
            "severity": sev,
            "confidence_type": "official_warning",
            "sources": sources,
            "source": sources,
            "reason": f"Official avalanche warning active for {top_a.get('location') or 'Himalayan district'}: {top_a.get('title') or top_a.get('description')}.",
            "evidence": evidence,
        }

    # If no official alert exists: active remains False
    is_high_altitude_slope = elev >= 2000.0 and slope >= 25.0 and temp_2m <= 2.0
    monitoring_sev = "LOW" if is_high_altitude_slope else "NONE"
    reason_str = (
        f"High-altitude steep terrain ({elev:.0f}m, slope: {slope:.1f}°), but no official DGRE/SACHET avalanche warning active."
        if is_high_altitude_slope
        else "No official avalanche warning in effect for this region."
    )

    return {
        "hazard": "avalanche",
        "active": False,
        "severity": monitoring_sev,
        "confidence_type": "evidence_based",
        "sources": sources,
        "source": sources,
        "reason": reason_str,
        "evidence": evidence,
    }


# ==============================================================================
# 15. LOCAL FLASH FLOOD / URBAN RUNOFF DETECTOR
# ==============================================================================
def detect_flash_flood(
    features: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    official_alerts: List[Dict[str, Any]],
    location_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates local flash flood and urban surface runoff risk from intense short-duration rainfall
    (1h, 3h, 6h), waterlogging and drainage model probabilities, terrain slope, and Sentinel-1 SAR.
    Does NOT require GloFAS river discharge (distinct from large riverine flood).
    """
    precip_1h = float(features.get("precip_1h") or d_built.get("precip_1h") or 0.0)
    rain_3h = float(
        features.get("rain_3h")
        or d_built.get("rain_3h")
        or (history_data.get("flood") or {}).get("rain_3h_mm")
        or (precip_1h * 2.5 if precip_1h > 0 else 0.0)
    )
    rain_6h = float(
        features.get("rain_6h")
        or d_built.get("rain_6h")
        or (history_data.get("flood") or {}).get("rain_6h_mm")
        or (rain_3h * 1.5 if rain_3h > 0 else 0.0)
    )
    rain_24h = float(features.get("rain_24h") or (history_data.get("flood") or {}).get("rain_24h_mm") or 0.0)

    wl_prob = (
        d_built.get("waterlogging_probability")
        if d_built.get("waterlogging_probability") is not None
        else features.get("waterlogging_probability")
    )
    dr_prob = (
        d_built.get("drainage_congestion_probability")
        if d_built.get("drainage_congestion_probability") is not None
        else features.get("drainage_congestion_probability")
    )
    wl_p = float(wl_prob) if wl_prob is not None else 0.0
    dr_p = float(dr_prob) if dr_prob is not None else 0.0

    slope = float(features.get("slope") or d_built.get("slope") or 1.0)
    elev = float(features.get("elevation") or d_built.get("elevation") or 0.0)

    sar_meta = d_built.get("sentinel1_metadata") or {}
    sar_water_change = bool(sar_meta.get("surface_water_change_detected") or sar_meta.get("water_change_detected"))

    sources = ["Open-Meteo Short-Duration Telemetry", "Waterlogging Model", "Drainage Model"]
    if sar_water_change:
        sources.append("Sentinel-1 SAR Surface Water")

    evidence = {
        "rain_1h_mm": round(precip_1h, 1),
        "rain_3h_mm": round(rain_3h, 1),
        "rain_6h_mm": round(rain_6h, 1),
        "rain_24h_mm": round(rain_24h, 1),
        "waterlogging_probability": round(wl_p, 3),
        "drainage_congestion_probability": round(dr_p, 3),
        "surface_water_change": sar_water_change,
        "slope_deg": round(slope, 1),
        "elevation_m": round(elev, 0),
    }

    # Rule 1: Cloudburst / Extreme Torrential Downpour (1h >= 50mm)
    if precip_1h >= 50.0:
        return {
            "hazard": "flash_flood",
            "active": True,
            "severity": "CRITICAL",
            "confidence_type": "multi_source_evidence",
            "sources": sources,
            "source": sources,
            "reason": f"Extreme cloudburst intensity rainfall ({precip_1h:.1f} mm/1h) triggering critical flash flooding.",
            "evidence": evidence,
        }

    # Rule 2: Severe flash runoff / drainage failure
    if (
        (precip_1h >= 30.0)
        or (rain_3h >= 65.0)
        or (precip_1h >= 20.0 and (wl_p >= 0.60 or dr_p >= 0.60))
        or (precip_1h >= 25.0 and sar_water_change)
    ):
        sev = "CRITICAL" if (precip_1h >= 40.0 or (wl_p >= 0.70 and dr_p >= 0.70)) else "SEVERE"
        return {
            "hazard": "flash_flood",
            "active": True,
            "severity": sev,
            "confidence_type": "multi_source_evidence",
            "sources": sources,
            "source": sources,
            "reason": f"Intense downpour ({precip_1h:.1f} mm/1h, {rain_3h:.1f} mm/3h) with high surface runoff and drainage failure.",
            "evidence": evidence,
        }

    # Rule 3: High flash flood / urban waterlogging
    if (
        (precip_1h >= 20.0)
        or (rain_3h >= 45.0)
        or (precip_1h >= 12.0 and wl_p >= 0.45 and dr_p >= 0.45)
        or (precip_1h >= 10.0 and sar_water_change and wl_p >= 0.40)
    ):
        return {
            "hazard": "flash_flood",
            "active": True,
            "severity": "HIGH",
            "confidence_type": "multi_source_evidence",
            "sources": sources,
            "source": sources,
            "reason": f"Intense short-duration rainfall ({precip_1h:.1f} mm/1h) combined with high waterlogging ({wl_p:.0%}) and drainage congestion ({dr_p:.0%}).",
            "evidence": evidence,
        }

    # Rule 4: Moderate urban runoff
    if (
        (precip_1h >= 10.0)
        or (rain_3h >= 25.0)
        or (precip_1h >= 8.0 and (wl_p >= 0.40 or dr_p >= 0.40))
    ):
        if wl_p >= 0.35 or dr_p >= 0.35 or precip_1h >= 15.0:
            return {
                "hazard": "flash_flood",
                "active": True,
                "severity": "MODERATE",
                "confidence_type": "multi_source_evidence",
                "sources": sources,
                "source": sources,
                "reason": f"Elevated urban surface runoff ({precip_1h:.1f} mm/1h) with moderate drainage congestion.",
                "evidence": evidence,
            }

    # Low / No runoff scenario: e.g. moderate 24h rain but low 1h intensity and low waterlogging
    return {
        "hazard": "flash_flood",
        "active": False,
        "severity": "NONE",
        "confidence_type": "multi_source_evidence",
        "sources": sources,
        "source": sources,
        "reason": "Short-duration precipitation intensity and drainage capacity do not indicate flash flooding.",
        "evidence": evidence,
    }


# ==============================================================================
# MAIN API-FIRST MULTI-HAZARD EVALUATOR
# ==============================================================================
def evaluate_api_hazards(
    features: Optional[Dict[str, Any]] = None,
    d_built: Optional[Dict[str, Any]] = None,
    history_data: Optional[Dict[str, Any]] = None,
    official_alerts: Optional[List[Dict[str, Any]]] = None,
    location_meta: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main Multi-Hazard Operational Engine:
    Evaluates all 15 supported hazard categories independently.
    Constructs:
    1. hazard_status: Map of all 15 categories with active, severity, evidence, sources, and reason.
    2. verified_hazards: List of active hazards sorted by operational severity weight.
    3. primary_hazard: Single top operational hazard (or nominal baseline) for backwards compatibility.
    """
    features = features if features is not None else (kwargs.get("features") or {})
    d_built = d_built if d_built is not None else (kwargs.get("live") or kwargs.get("d_built") or {})
    history_data = history_data if history_data is not None else (kwargs.get("history") or kwargs.get("history_data") or {})
    if official_alerts is None:
        official_alerts = kwargs.get("official") or kwargs.get("official_alerts") or list(d_built.get("official_alerts") or [])
    if isinstance(official_alerts, dict):
        official_alerts = [official_alerts]
    location_meta = location_meta if location_meta is not None else (kwargs.get("location_meta") or kwargs.get("location") or {})

    # Evaluate each hazard independently
    h_earthquake = detect_earthquake(features, d_built, history_data, official_alerts, location_meta)
    h_flood = detect_flood(features, d_built, history_data, official_alerts, location_meta)
    h_cyclone = detect_cyclone(features, d_built, history_data, official_alerts, location_meta)
    h_drought = detect_drought(features, d_built, history_data, official_alerts, location_meta)
    h_heat_wave = detect_heat_wave(features, d_built, history_data, official_alerts, location_meta)
    h_cold_wave = detect_cold_wave(features, d_built, history_data, official_alerts, location_meta)
    h_thunderstorm = detect_thunderstorm(features, d_built, history_data, official_alerts, location_meta)
    h_high_wind = detect_high_wind(features, d_built, history_data, official_alerts, location_meta)
    h_tsunami = detect_tsunami(features, d_built, history_data, official_alerts, location_meta)
    h_volcano = detect_volcano(features, d_built, history_data, official_alerts, location_meta)
    h_air_pollution = detect_air_pollution(features, d_built, history_data, official_alerts, location_meta)
    h_landslide = detect_landslide(features, d_built, history_data, official_alerts, location_meta)
    h_wildfire = detect_wildfire(features, d_built, history_data, official_alerts, location_meta)
    h_avalanche = detect_avalanche(features, d_built, history_data, official_alerts, location_meta)
    h_flash_flood = detect_flash_flood(features, d_built, history_data, official_alerts, location_meta)

    hazard_status = {
        "flood": h_flood,
        "flash_flood": h_flash_flood,
        "thunderstorm": h_thunderstorm,
        "high_wind": h_high_wind,
        "heat_wave": h_heat_wave,
        "cold_wave": h_cold_wave,
        "earthquake": h_earthquake,
        "cyclone": h_cyclone,
        "drought": h_drought,
        "air_pollution": h_air_pollution,
        "landslide": h_landslide,
        "tsunami": h_tsunami,
        "volcano": h_volcano,
        "wildfire": h_wildfire,
        "avalanche": h_avalanche,
    }

    # Normalize keys: ensure both "sources" and "source" exist on every hazard
    for h in hazard_status.values():
        if "sources" in h and "source" not in h:
            h["source"] = list(h["sources"])
        elif "source" in h and "sources" not in h:
            h["sources"] = list(h["source"])

    # Extract all active verified hazards
    verified_hazards = [
        item for item in hazard_status.values()
        if item.get("active") is True
    ]

    HAZARD_BASE_PRIORITY = {
        "cyclone": 15,
        "tsunami": 15,
        "avalanche": 14,
        "earthquake": 12,
        "volcano": 12,
        "flash_flood": 11,
        "flood": 10,
        "wildfire": 10,
        "landslide": 9,
        "thunderstorm": 8,
        "heat_wave": 7,
        "cold_wave": 7,
        "high_wind": 6,
        "drought": 5,
        "air_pollution": 4,
    }

    # Sort deterministically by severity weight descending, then hazard base priority
    verified_hazards.sort(
        key=lambda h: (
            SEVERITY_WEIGHTS.get(h.get("severity", "LOW"), 0),
            HAZARD_BASE_PRIORITY.get(h.get("hazard", ""), 0),
        ),
        reverse=True
    )

    # Determine primary operational hazard for backward compatibility
    if verified_hazards:
        top_h = verified_hazards[0]
        if top_h.get("hazard") == "drought" and top_h.get("confidence_type") != "official_bulletin" and top_h.get("severity") != "CRITICAL":
            op_status = "SUPPORTED_HAZARD_SIGNAL"
        elif top_h.get("severity") in ("CRITICAL", "SEVERE", "HIGH"):
            op_status = "VERIFIED_ACTIVE_HAZARD"
        else:
            op_status = "SUPPORTED_HAZARD_SIGNAL"

        primary_hazard = {
            "hazard": top_h["hazard"],
            "status": op_status,
            "active": True,
            "severity": top_h.get("severity", "MODERATE"),
            "confidence_type": top_h.get("confidence_type", "evidence_based"),
            "sources": top_h.get("sources", []),
            "source": top_h.get("source", top_h.get("sources", [])),
            "reason": top_h.get("reason", ""),
            "alert_title": f"Verified {top_h['hazard'].replace('_', ' ').title()} Hazard — {top_h.get('severity', 'HIGH')}",
            "evidence": top_h.get("evidence", {}),
        }
    else:
        primary_hazard = {
            "hazard": None,
            "status": "NO_ACTIVE_HAZARD",
            "active": False,
            "severity": "LOW",
            "confidence_type": "evidence_based",
            "sources": ["National Telemetry Baseline"],
            "source": ["National Telemetry Baseline"],
            "reason": "Nominal baseline conditions: No severe disaster hazards detected across official feeds or live telemetry.",
            "alert_title": "No Active Disaster Threat · Nominal Conditions",
            "evidence": {},
        }

    return {
        "verified_hazards": verified_hazards,
        "hazard_status": hazard_status,
        "primary_hazard": primary_hazard,
        "active_hazard_count": len(verified_hazards),
    }
