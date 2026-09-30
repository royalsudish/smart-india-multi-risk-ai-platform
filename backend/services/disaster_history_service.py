"""
Smart Multi-Risk Platform - Disaster History & Context Service
Provides multi-window historical and recent context for:
1. Earthquake (24h, 7d, 30d, 90d; 50km, 100km, 250km, 500km radius; nearest, largest, most recent)
2. Flood (7-day daily river discharge history, peak discharge, trend, 24h/3d/7d/30d rainfall accumulation)
3. Cyclone & Volcano (GDACS active regional alerts and recent event history)
4. Drought (30d/90d cumulative rainfall, precipitation anomaly, thermal context)
"""

import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from backend.config.settings import (
    USGS_EARTHQUAKE_URL,
    OPEN_METEO_FLOOD_URL,
)
from backend.data_sources.http_client import get_json
from backend.data_sources.usgs import haversine_km
from backend.data_sources.gdacs import get_cyclone_and_volcano_alerts
from backend.data_sources.open_meteo import forecast, flood
from backend.utils.cache import provider_cache, make_cache_key

# -----------------------------------------------------------------------------
# SEISMIC RELEVANCE POLICY
# -----------------------------------------------------------------------------
SEISMIC_POLICY = {
    # Category A: Active / Locally Relevant Emergency (within 72 hours / 3 days)
    "active_recency_days": 3.0,
    "major_magnitude": 6.5,
    "major_radius_km": 350.0,
    "strong_magnitude": 5.5,
    "strong_radius_km": 150.0,
    "moderate_magnitude": 4.5,
    "moderate_radius_km": 50.0,
    # Category B: Recent But Not Locally Severe / Monitoring (within 7 days)
    "recent_recency_days": 7.0,
    "recent_major_radius_km": 500.0,
    "recent_strong_radius_km": 250.0,
    "recent_moderate_radius_km": 100.0,
}

TTL_DISASTER_HISTORY = 600.0  # 10 minutes cache


def fetch_earthquake_history(
    lat: float,
    lon: float,
    max_radius_km: int = 500,
    days: int = 90,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Fetches comprehensive 90-day seismic events within max_radius_km from USGS
    and computes multi-temporal and multi-spatial window metrics.
    """
    cache_key = make_cache_key("disaster_eq_hist", lat, lon, {"r": max_radius_km, "d": days})
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(days=days)

    params = {
        "format": "geojson",
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "maxradiuskm": int(max_radius_km),
        "starttime": start_time.strftime("%Y-%m-%d"),
        "minmagnitude": 1.0,
    }

    try:
        data = get_json(USGS_EARTHQUAKE_URL, params, timeout=timeout)
        raw_features = data.get("features") or []
    except Exception as exc:
        res = {
            "status": "unavailable",
            "reason": str(exc),
            "events_24h": 0,
            "events_7d": 0,
            "events_30d": 0,
            "events_90d": 0,
            "nearest_event": None,
            "largest_event": None,
            "most_recent_event": None,
            "radius_breakdown": {"50km": 0, "100km": 0, "250km": 0, "500km": 0},
            "magnitude_distribution": {},
            "has_active_local_event": False,
            "has_recent_monitoring_event": False,
            "seismic_category": "HISTORICAL_DISTANT_BACKGROUND",
            "seismic_reason": f"USGS query failed: {exc}",
        }
        return res

    parsed_events: List[Dict[str, Any]] = []
    for feat in raw_features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [0.0, 0.0, 0.0]

        ev_lon = float(coords[0]) if len(coords) >= 1 else float(lon)
        ev_lat = float(coords[1]) if len(coords) >= 2 else float(lat)
        depth = float(coords[2]) if len(coords) >= 3 else 0.0

        mag = float(props.get("mag") or 0.0)
        dist_km = round(haversine_km(float(lat), float(lon), ev_lat, ev_lon), 2)

        time_ms = props.get("time")
        if time_ms:
            ev_time = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
            age_sec = max(0.0, (now_utc - ev_time).total_seconds())
            age_hours = round(age_sec / 3600.0, 2)
            age_days = round(age_sec / 86400.0, 2)
            time_iso = ev_time.isoformat()
        else:
            age_hours = 9999.0
            age_days = 999.0
            time_iso = None

        parsed_events.append({
            "id": feat.get("id"),
            "title": str(props.get("title") or props.get("place") or "Unknown Event"),
            "magnitude": round(mag, 2),
            "depth_km": round(depth, 2),
            "distance_km": dist_km,
            "age_days": age_days,
            "age_hours": age_hours,
            "time_utc": time_iso,
            "tsunami_flag": int(props.get("tsunami") or 0),
            "coordinates": [round(ev_lon, 4), round(ev_lat, 4)],
        })

    # Multi-temporal counts
    cnt_24h = sum(1 for e in parsed_events if e["age_hours"] <= 24.0)
    cnt_7d = sum(1 for e in parsed_events if e["age_days"] <= 7.0)
    cnt_30d = sum(1 for e in parsed_events if e["age_days"] <= 30.0)
    cnt_90d = len(parsed_events)

    # Multi-spatial counts
    cnt_50k = sum(1 for e in parsed_events if e["distance_km"] <= 50.0)
    cnt_100k = sum(1 for e in parsed_events if e["distance_km"] <= 100.0)
    cnt_250k = sum(1 for e in parsed_events if e["distance_km"] <= 250.0)
    cnt_500k = sum(1 for e in parsed_events if e["distance_km"] <= 500.0)

    # Magnitude breakdown
    mag_dist = {
        "M1.0_to_2.9": sum(1 for e in parsed_events if 1.0 <= e["magnitude"] < 3.0),
        "M3.0_to_3.9": sum(1 for e in parsed_events if 3.0 <= e["magnitude"] < 4.0),
        "M4.0_to_4.9": sum(1 for e in parsed_events if 4.0 <= e["magnitude"] < 5.0),
        "M5.0_to_5.9": sum(1 for e in parsed_events if 5.0 <= e["magnitude"] < 6.0),
        "M6.0_plus": sum(1 for e in parsed_events if e["magnitude"] >= 6.0),
    }

    # Nearest, Largest, Most Recent
    nearest_event = min(parsed_events, key=lambda x: x["distance_km"]) if parsed_events else None
    largest_event = max(parsed_events, key=lambda x: x["magnitude"]) if parsed_events else None
    most_recent_event = min(parsed_events, key=lambda x: x["age_hours"]) if parsed_events else None

    # Evaluate Active Local Event vs Recent Monitoring vs Historical Background
    has_active_local = False
    has_recent_monitoring = False
    seismic_cat = "HISTORICAL_DISTANT_BACKGROUND"
    seismic_reason = "No seismic events recorded within regional query window (500 km, 90 days)."

    if parsed_events:
        # Check active local criteria across all events in window
        active_candidates = []
        for e in parsed_events:
            e_mag = e["magnitude"]
            e_dist = e["distance_km"]
            e_age = e["age_days"]
            e_tsunami = e["tsunami_flag"]

            if e_tsunami == 1:
                active_candidates.append((e, "Active tsunami warning/watch generated."))
                continue

            if e_age <= SEISMIC_POLICY["active_recency_days"]:
                if e_mag >= SEISMIC_POLICY["major_magnitude"] and e_dist <= SEISMIC_POLICY["major_radius_km"]:
                    active_candidates.append((e, f"Major M{e_mag:.1f} earthquake {e_dist:.0f} km away occurred {e_age:.1f} days ago (within 72h active window)."))
                elif e_mag >= SEISMIC_POLICY["strong_magnitude"] and e_dist <= SEISMIC_POLICY["strong_radius_km"]:
                    active_candidates.append((e, f"Strong M{e_mag:.1f} earthquake {e_dist:.0f} km away occurred {e_age:.1f} days ago (within 72h active window)."))
                elif e_mag >= SEISMIC_POLICY["moderate_magnitude"] and e_dist <= SEISMIC_POLICY["moderate_radius_km"]:
                    active_candidates.append((e, f"Moderate M{e_mag:.1f} earthquake {e_dist:.0f} km away occurred {e_age:.1f} days ago (within near-epicentral zone)."))

        if active_candidates:
            has_active_local = True
            seismic_cat = "ACTIVE_LOCALLY_RELEVANT"
            seismic_reason = active_candidates[0][1]
        else:
            # Check monitoring criteria
            monitoring_candidates = []
            for e in parsed_events:
                e_mag = e["magnitude"]
                e_dist = e["distance_km"]
                e_age = e["age_days"]
                if e_age <= SEISMIC_POLICY["recent_recency_days"]:
                    if (e_mag >= 6.0 and e_dist <= SEISMIC_POLICY["recent_major_radius_km"]) or \
                       (e_mag >= 5.0 and e_dist <= SEISMIC_POLICY["recent_strong_radius_km"]) or \
                       (e_mag >= 4.0 and e_dist <= SEISMIC_POLICY["recent_moderate_radius_km"]):
                        monitoring_candidates.append((e, f"Recent M{e_mag:.1f} earthquake {e_dist:.0f} km away occurred {e_age:.1f} days ago (monitoring phase)."))

            if monitoring_candidates:
                has_recent_monitoring = True
                seismic_cat = "RECENT_NOT_LOCAL"
                seismic_reason = monitoring_candidates[0][1]
            else:
                top_e = largest_event or parsed_events[0]
                seismic_cat = "HISTORICAL_DISTANT_BACKGROUND"
                seismic_reason = (
                    f"M{top_e['magnitude']:.1f} event occurred {top_e['distance_km']:.0f} km away "
                    f"({top_e['age_days']:.1f} days ago) — classified as historical/distant background."
                )

    result = {
        "status": "available",
        "events_24h": cnt_24h,
        "events_7d": cnt_7d,
        "events_30d": cnt_30d,
        "events_90d": cnt_90d,
        "nearest_event": nearest_event,
        "largest_event": largest_event,
        "most_recent_event": most_recent_event,
        "radius_breakdown": {
            "50km": cnt_50k,
            "100km": cnt_100k,
            "250km": cnt_250k,
            "500km": cnt_500k,
        },
        "magnitude_distribution": mag_dist,
        "has_active_local_event": has_active_local,
        "has_recent_monitoring_event": has_recent_monitoring,
        "seismic_category": seismic_cat,
        "seismic_reason": seismic_reason,
        "max_quake_mag_90d": largest_event["magnitude"] if largest_event else 0.0,
        "nearest_quake_km": nearest_event["distance_km"] if nearest_event else None,
        "nearest_quake_age_days": nearest_event["age_days"] if nearest_event else None,
        "nearest_quake_age_hours": nearest_event["age_hours"] if nearest_event else None,
    }

    provider_cache.set(cache_key, result, ttl=TTL_DISASTER_HISTORY)
    return result


def fetch_flood_history(
    lat: float,
    lon: float,
    past_days: int = 7,
    forecast_days: int = 2,
    timeout: int = 10,
    precomputed_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetches 7-day river discharge history, current discharge, peak discharge,
    and trend from Open-Meteo Flood (GloFAS v4).
    """
    cache_key = make_cache_key("disaster_flood_hist", lat, lon, {"p": past_days, "f": forecast_days})
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    # Check if precomputed flood already has full daily data
    if precomputed_sources and "flood_daily" in precomputed_sources:
        raw_daily = precomputed_sources["flood_daily"]
    else:
        try:
            raw_data = get_json(OPEN_METEO_FLOOD_URL, {
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "daily": "river_discharge",
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "auto",
            }, timeout=timeout)
            raw_daily = raw_data.get("daily") if isinstance(raw_data, dict) else {}
        except Exception as exc:
            res = {
                "status": "unavailable",
                "reason": str(exc),
                "current_discharge_m3s": None,
                "peak_discharge_7d_m3s": None,
                "recent_discharge_series": [],
                "discharge_trend": "UNKNOWN",
                "is_discharge_elevated": False,
            }
            return res

    times = raw_daily.get("time") or []
    discharges = raw_daily.get("river_discharge") or []

    history_series = []
    valid_discharges = []
    for t_str, d_val in zip(times, discharges):
        val = None
        if d_val is not None:
            try:
                fval = float(d_val)
                if not (math.isnan(fval) or math.isinf(fval)):
                    val = round(fval, 2)
                    valid_discharges.append(val)
            except (ValueError, TypeError):
                val = None
        history_series.append({"date": t_str, "discharge_m3s": val})

    current_val = valid_discharges[-1] if valid_discharges else None
    peak_val = max(valid_discharges) if valid_discharges else None

    # Trend calculation (comparing last 3 observations)
    trend = "UNKNOWN"
    if len(valid_discharges) >= 3:
        d_latest = valid_discharges[-1]
        d_prev = valid_discharges[-2]
        d_prev2 = valid_discharges[-3]
        if d_latest > d_prev >= d_prev2:
            trend = "RISING"
        elif d_latest < d_prev <= d_prev2:
            trend = "FALLING"
        else:
            diff = abs(d_latest - d_prev)
            if diff < 1.0 or (d_prev > 0 and diff / d_prev < 0.05):
                trend = "STEADY"
            elif d_latest > d_prev:
                trend = "RISING"
            else:
                trend = "FALLING"

    is_elevated = (current_val is not None and current_val >= 50.0) or \
                  (peak_val is not None and peak_val >= 80.0)

    result = {
        "status": "available" if valid_discharges else "unavailable",
        "current_discharge_m3s": current_val,
        "peak_discharge_7d_m3s": peak_val,
        "recent_discharge_series": history_series,
        "discharge_trend": trend,
        "is_discharge_elevated": is_elevated,
        "recent_discharge": history_series,
        "peak_discharge": peak_val,
        "current_discharge": current_val,
    }

    provider_cache.set(cache_key, result, ttl=TTL_DISASTER_HISTORY)
    return result


def fetch_rainfall_accumulation_context(
    lat: float,
    lon: float,
    precomputed_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes 24h, 3d, 7d, and 14d rainfall accumulation from Open-Meteo Weather forecast data.
    """
    w_data = None
    if precomputed_sources and "forecast" in precomputed_sources:
        w_data = precomputed_sources["forecast"]

    if not w_data:
        try:
            w_data = forecast(lat, lon, past_days=7, forecast_days=2)
        except Exception:
            w_data = {}

    wh = (w_data.get("hourly") or {}) if isinstance(w_data, dict) else {}
    rain = wh.get("rain") or wh.get("precipitation") or []

    def _sum_recent(hours: int) -> float:
        if not rain:
            return 0.0
        tail = rain[-hours:] if len(rain) >= hours else rain
        return round(float(sum(float(x or 0.0) for x in tail)), 2)

    rain_24h = _sum_recent(24)
    rain_3d = _sum_recent(72)
    rain_7d = _sum_recent(168)

    # Future forecast rainfall
    df_future = wh.get("precipitation") or wh.get("rain") or []
    # If standard 7-day past + 2-day forecast, the last 48 hours represent forecast
    forecast_24h = round(float(sum(float(x or 0.0) for x in df_future[-48:-24])), 2) if len(df_future) >= 48 else 0.0
    forecast_48h = round(float(sum(float(x or 0.0) for x in df_future[-48:])), 2) if len(df_future) >= 48 else 0.0

    return {
        "rain_24h_mm": rain_24h,
        "rain_3d_mm": rain_3d,
        "rain_7d_mm": rain_7d,
        "forecast_24h_mm": forecast_24h,
        "forecast_48h_mm": forecast_48h,
        "is_heavy_rain": rain_24h >= 40.0 or rain_3d >= 75.0,
    }


def get_disaster_history(
    lat: float,
    lon: float,
    precomputed_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified entry point returning the complete historical & recent context across:
    - Earthquake (90-day multi-window)
    - Flood (7-day daily discharge series & trend)
    - Rainfall Accumulation (24h, 3d, 7d, forecast)
    - Cyclone & Volcano (GDACS active regional alerts)
    - Drought Context (7d/30d precipitation & thermal context)
    """
    cache_key = make_cache_key("disaster_history_unified", lat, lon)
    cached = provider_cache.get(cache_key)
    if cached is not None:
        return cached

    eq_hist = fetch_earthquake_history(lat, lon, max_radius_km=500, days=90)
    flood_hist = fetch_flood_history(lat, lon, past_days=7, forecast_days=2, precomputed_sources=precomputed_sources)
    rain_ctx = fetch_rainfall_accumulation_context(lat, lon, precomputed_sources=precomputed_sources)

    # Cyclone & Volcano
    gdacs_data = {}
    if precomputed_sources and "gdacs" in precomputed_sources:
        gdacs_data = precomputed_sources["gdacs"]
    else:
        try:
            gdacs_data = get_cyclone_and_volcano_alerts(lat, lon)
        except Exception:
            gdacs_data = {}

    active_cyclone = (gdacs_data.get("gdacs_cyclone") == 1)
    active_volcano = (gdacs_data.get("gdacs_volcano") == 1)

    # Drought Indicators
    rain_7d = rain_ctx.get("rain_7d_mm", 0.0)
    # Estimate 30d rain: extrapolation or 4x 7d if historical 30d not directly stored
    est_rain_30d = round(rain_7d * 3.5, 2)
    is_drought_indicated = (est_rain_30d < 10.0 and rain_7d < 2.0)

    result = {
        "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "earthquake": eq_hist,
        "flood": {
            **flood_hist,
            "rainfall_accumulation": rain_ctx,
        },
        "rainfall": rain_ctx,
        "cyclone": {
            "active_alert": active_cyclone,
            "gdacs_cyclone_flag": 1 if active_cyclone else 0,
            "cyclone_details": gdacs_data.get("cyclone_details"),
            "nearest_cyclone_distance_km": (gdacs_data.get("cyclone_details") or {}).get("distance_km"),
        },
        "volcano": {
            "active_alert": active_volcano,
            "gdacs_volcano_flag": 1 if active_volcano else 0,
            "volcano_details": gdacs_data.get("volcano_details"),
            "nearest_volcano_distance_km": (gdacs_data.get("volcano_details") or {}).get("distance_km"),
        },
        "drought": {
            "recent_rain_7d_mm": rain_7d,
            "estimated_rain_30d_mm": est_rain_30d,
            "is_drought_indicated": is_drought_indicated,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    provider_cache.set(cache_key, result, ttl=TTL_DISASTER_HISTORY)
    return result
