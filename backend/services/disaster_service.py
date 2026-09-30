"""
Smart Multi-Risk Platform - Disaster Intelligence & Evidence Corroboration Pipeline
Coordinates:
1. 20-Feature Input Construction (exact training contract with granular provenance)
2. Trained 20-Feature XGBoost Model Inference (Raw Model Output: prediction, confidence, probabilities)
3. Multi-Window Historical & Live Evidence (USGS 90d, GloFAS River Flow, Open-Meteo, GDACS)
4. Official Indian Authoritative Evidence (CWC Flood, IMD Weather Warnings, NCS Seismology)
5. Evidence Corroboration Engine:
   - Validates model prediction against real-world geographical and temporal evidence.
   - Evaluates active local hazards vs uncorroborated/historical background signals.
   - Handles multiple simultaneous hazards (verified_hazards list).
   - Prioritizes official alerts without hiding raw model predictions.
   - Assigns operational verified hazard states:
     VERIFIED_ACTIVE_HAZARD, SUPPORTED_HAZARD_SIGNAL, UNCORROBORATED_MODEL_SIGNAL,
     LOW_CONFIDENCE_MODEL_SIGNAL, NO_ACTIVE_HAZARD, DATA_INCOMPLETE.
"""

import time
import math
from typing import Dict, Any, List, Optional

from backend.services.model_service import registry
from backend.services.disaster_features import (
    build_disaster_features,
    FEATURE_ORDER,
    get_feature_contract_audit,
)
from backend.services.disaster_history_service import get_disaster_history, SEISMIC_POLICY
from backend.services.pan_india_catalog import resolve_location
from backend.services.api_hazard_engine import evaluate_api_hazards
from backend.data_sources.firms_fire import fetch_firms_hotspots


def corroborate_disaster_evidence(
    raw_model_result: Dict[str, Any],
    d_built: Dict[str, Any],
    history_data: Dict[str, Any],
    location_meta: Optional[Dict[str, Any]] = None,
    official_alerts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evidence Corroboration Engine:
    Combines Raw Model Output + Live Evidence + Historical Context + Official Indian Alerts
    to produce the Verified Operational Hazard Output without modifying the raw model.
    Supports multiple simultaneous concurrent hazards (verified_hazards array).
    """
    raw_pred = str(raw_model_result.get("prediction", "nominal")).lower()
    raw_conf = float(raw_model_result.get("confidence", 0.0))
    raw_probs = raw_model_result.get("probabilities", {})

    features = d_built.get("features") or {}
    eq_hist = history_data.get("earthquake") or {}
    flood_hist = history_data.get("flood") or {}
    cyclone_hist = history_data.get("cyclone") or {}
    volcano_hist = history_data.get("volcano") or {}
    drought_hist = history_data.get("drought") or {}
    rain_hist = flood_hist.get("rainfall_accumulation") or {}

    # Official Alerts (Phase 15 priority)
    if official_alerts is None:
        official_alerts = list(d_built.get("official_alerts") or [])
    official_india = d_built.get("official_india") or {}

    # 1. Structure Live Telemetry & Evidence
    nearest_quake = eq_hist.get("nearest_event") or {}
    seismic_meta = d_built.get("seismic_metadata") or {}

    nearest_quake_km = (
        eq_hist.get("nearest_quake_km")
        or nearest_quake.get("distance_km")
        or seismic_meta.get("event_distance_km")
    )
    nearest_quake_age_days = (
        eq_hist.get("nearest_quake_age_days")
        or nearest_quake.get("age_days")
        or (seismic_meta.get("event_age_hours") / 24.0 if seismic_meta.get("event_age_hours") is not None else None)
    )
    nearest_quake_age_hours = (
        eq_hist.get("nearest_quake_age_hours")
        or (nearest_quake_age_days * 24.0 if nearest_quake_age_days is not None else None)
        or seismic_meta.get("event_age_hours")
    )
    max_quake_mag = (
        features.get("max_quake_mag")
        or nearest_quake.get("mag")
        or nearest_quake.get("magnitude")
        or seismic_meta.get("max_quake_mag")
        or 0.0
    )

    active_tsunami_alert = (features.get("tsunami_flag") == 1) or (nearest_quake.get("tsunami_flag") == 1)
    active_gdacs_cyclone = (
        (features.get("gdacs_cyclone") == 1)
        or ((d_built.get("gdacs_metadata") or {}).get("gdacs_cyclone") == 1)
        or cyclone_hist.get("active_alert", False)
    )
    active_gdacs_volcano = (
        (features.get("gdacs_volcano") == 1)
        or ((d_built.get("gdacs_metadata") or {}).get("gdacs_volcano") == 1)
        or volcano_hist.get("active_alert", False)
    )

    # Seismic status string for human-readable live telemetry
    if active_tsunami_alert:
        seismic_status = "Active tsunami alert"
    elif eq_hist.get("has_active_local_event"):
        seismic_status = "Active local emergency"
    elif eq_hist.get("has_recent_monitoring_event"):
        seismic_status = "Recent monitoring (not local emergency)"
    elif nearest_quake_km is not None:
        seismic_status = "Not locally active"
    else:
        seismic_status = "No regional events in 90 days"

    river_discharge = features.get("river_discharge")
    curr_d = float(river_discharge) if river_discharge is not None else 0.0
    peak_d = float(flood_hist.get("peak_discharge_7d_m3s") or 0.0)
    d_trend = str(flood_hist.get("discharge_trend", "UNKNOWN")).upper()

    rain_24h = float(features.get("rain_24h") or rain_hist.get("rain_24h_mm") or 0.0)
    rain_48h = float(features.get("rain_48h") or 0.0)
    rain_3d = float(rain_hist.get("rain_3d_mm") or 0.0)
    rain_7d = float(rain_hist.get("rain_7d_mm") or 0.0)
    forecast_rain = float(rain_hist.get("forecast_24h_mm") or 0.0)

    temp_2m = float(features.get("temp_2m") or 0.0)
    precip_1h = float(features.get("precip_1h") or 0.0)
    wind_speed = float(features.get("wind_speed_10m") or 0.0)
    wind_gusts = float(features.get("wind_gusts_10m") or 0.0)
    surface_pressure = float(features.get("surface_pressure") or 1013.25)
    weather_code = int(features.get("weather_code") or 0)
    us_aqi = features.get("us_aqi")
    pm2_5 = features.get("pm2_5")

    wl_prob = (
        d_built.get("waterlogging_probability")
        if d_built.get("waterlogging_probability") is not None
        else (
            (history_data.get("waterlogging") or {}).get("positive_probability")
            if isinstance(history_data.get("waterlogging"), dict)
            else None
        )
    )
    drainage_prob = (
        d_built.get("drainage_congestion_probability")
        if d_built.get("drainage_congestion_probability") is not None
        else (
            (history_data.get("drainage") or {}).get("probability")
            if isinstance(history_data.get("drainage"), dict)
            else None
        )
    )
    wl_p = float(wl_prob) if wl_prob is not None else 0.0
    dr_p = float(drainage_prob) if drainage_prob is not None else 0.0

    live_evidence = {
        "nearest_quake_km": nearest_quake_km,
        "nearest_quake_age_days": nearest_quake_age_days,
        "nearest_quake_age_hours": nearest_quake_age_hours,
        "seismic_status": seismic_status,
        "active_tsunami_alert": active_tsunami_alert,
        "active_gdacs_cyclone": active_gdacs_cyclone,
        "active_gdacs_volcano": active_gdacs_volcano,
        "active_flood_warning": any(a.get("hazard") == "flood" for a in official_alerts),
        "active_cyclone_warning": any(a.get("hazard") == "cyclone" for a in official_alerts),
        "river_discharge_m3s": round(river_discharge, 2) if river_discharge is not None else None,
        "discharge_trend": d_trend,
        "rain_24h_mm": round(rain_24h, 2),
        "rain_48h_mm": round(rain_48h, 2),
        "rain_3d_mm": round(rain_3d, 2),
        "rain_7d_mm": round(rain_7d, 2),
        "forecast_rain_24h_mm": round(forecast_rain, 2),
        "waterlogging_probability": round(float(wl_prob), 4) if wl_prob is not None else None,
        "drainage_congestion_probability": round(float(drainage_prob), 4) if drainage_prob is not None else None,
        "precip_1h_mm": round(precip_1h, 2),
        "temp_2m_c": round(temp_2m, 2),
        "wind_speed_10m_kmh": round(wind_speed, 2),
        "wind_gusts_10m_kmh": round(wind_gusts, 2),
        "surface_pressure_hpa": round(surface_pressure, 2),
        "weather_code": weather_code,
        "us_aqi": us_aqi,
        "pm2_5": pm2_5,
        "official_alerts": official_alerts,
        "firms_fire": d_built.get("firms_fire", {}),
        "fire_hotspots": (d_built.get("firms_fire") or {}).get("hotspots", []),
        "nearest_fire_distance_km": (d_built.get("firms_fire") or {}).get("nearest_fire_distance_km"),
        "fire_detections_10km": (d_built.get("firms_fire") or {}).get("detections_10km", 0),
        "fire_detections_25km": (d_built.get("firms_fire") or {}).get("detections_25km", 0),
        "max_frp": (d_built.get("firms_fire") or {}).get("max_frp", 0.0),
        "sachet_alerts": [a for a in official_alerts if "sachet" in str(a.get("source", "")).lower() or "dgre" in str(a.get("source", "")).lower()],
    }

    # 2. Structure Historical Context
    historical_context = {
        "quake_count_24h": eq_hist.get("events_24h", 0),
        "quake_count_7d": eq_hist.get("events_7d", 0),
        "quake_count_30d": eq_hist.get("events_30d", 0),
        "quake_count_90d": eq_hist.get("events_90d", 0),
        "largest_quake_mag_90d": eq_hist.get("max_quake_mag_90d", 0.0),
        "nearest_event_title": nearest_quake.get("title"),
        "nearest_event_mag": nearest_quake.get("magnitude"),
        "nearest_event_distance_km": nearest_quake_km,
        "nearest_event_age_days": nearest_quake_age_days,
        "current_discharge": round(river_discharge, 2) if river_discharge is not None else flood_hist.get("current_discharge_m3s"),
        "recent_discharge": flood_hist.get("recent_discharge_series") or flood_hist.get("recent_discharge", []),
        "peak_discharge": flood_hist.get("peak_discharge_7d_m3s") or flood_hist.get("peak_discharge"),
        "peak_discharge_7d_m3s": flood_hist.get("peak_discharge_7d_m3s"),
        "discharge_trend": d_trend,
        "rain_24h": round(rain_24h, 2),
        "rain_3d": round(rain_3d, 2),
        "rain_7d": round(rain_7d, 2),
        "recent_rain_7d_mm": round(rain_7d, 2),
        "forecast_rain": round(forecast_rain, 2),
        "forecast_rain_24h_mm": round(forecast_rain, 2),
        "waterlogging_probability": round(float(wl_prob), 4) if wl_prob is not None else None,
        "drainage_congestion_probability": round(float(drainage_prob), 4) if drainage_prob is not None else None,
        "is_drought_indicated": drought_hist.get("is_drought_indicated", False),
    }

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 3. API-FIRST MULTI-HAZARD EVALUATION ENGINE
    # -------------------------------------------------------------------------
    # Evaluates all supported hazard categories independently from live APIs & official bulletins.
    hazard_eval = evaluate_api_hazards(
        features=features,
        d_built=d_built,
        history_data=history_data,
        official_alerts=official_alerts,
        location_meta=location_meta,
    )

    verified_hazards_list = hazard_eval.get("verified_hazards", [])
    hazard_status = hazard_eval.get("hazard_status", {})

    # Determine primary operational verified_hazard for backward compatibility
    if verified_hazards_list:
        top_h = verified_hazards_list[0]
        if top_h.get("hazard") == "drought" and top_h.get("confidence_type") != "official_bulletin" and top_h.get("severity") != "CRITICAL":
            op_status = "SUPPORTED_HAZARD_SIGNAL"
        elif top_h.get("severity") in ("CRITICAL", "SEVERE", "HIGH"):
            op_status = "VERIFIED_ACTIVE_HAZARD"
        else:
            op_status = "SUPPORTED_HAZARD_SIGNAL"

        verified_hazard = {
            "hazard": top_h.get("hazard"),
            "status": op_status,
            "active": True,
            "severity": top_h.get("severity", "MODERATE"),
            "confidence_type": top_h.get("confidence_type", "evidence_based"),
            "sources": top_h.get("sources", []),
            "source": top_h.get("source", []),
            "reason": top_h.get("reason", ""),
            "alert_title": f"Verified {str(top_h.get('hazard')).replace('_', ' ').title()} Hazard — {top_h.get('severity', 'HIGH')}",
            "evidence": top_h.get("evidence", {}),
        }
    else:
        # No active operational hazard from APIs/telemetry
        if raw_pred != "nominal" and raw_conf >= 0.25:
            if raw_pred == "earthquake":
                uncorr_reason = (
                    f"The XGBoost model produced an earthquake pattern (confidence: {raw_conf:.1%}), "
                    f"but no recent locally relevant seismic event or official bulletin corroborates an emergency."
                )
            else:
                uncorr_reason = (
                    f"The XGBoost model produced a {raw_pred.replace('_', ' ')} pattern (confidence: {raw_conf:.1%}), "
                    f"but no active live telemetry or official bulletins corroborate an emergency."
                )
            verified_hazard = {
                "hazard": None,
                "status": "UNCORROBORATED_MODEL_SIGNAL",
                "active": False,
                "severity": "LOW",
                "confidence_type": "model_signal_only",
                "sources": ["Model Raw Output"],
                "source": ["Model Raw Output"],
                "reason": uncorr_reason,
                "alert_title": f"Uncorroborated {raw_pred.replace('_', ' ')} model signal",
                "evidence": {},
            }
        else:
            verified_hazard = {
                "hazard": None,
                "status": "NO_ACTIVE_HAZARD",
                "active": False,
                "severity": "LOW",
                "confidence_type": "evidence_based",
                "sources": ["National Telemetry Baseline"],
                "source": ["National Telemetry Baseline"],
                "reason": "Nominal baseline conditions: No severe disaster hazards detected across official feeds or live telemetry.",
                "alert_title": "No Active Disaster Threat",
                "evidence": {},
            }

    # -------------------------------------------------------------------------
    # 4. RAW MODEL OUTPUT (Secondary Live ML Intelligence)
    # -------------------------------------------------------------------------
    raw_model_out = {
        "prediction": raw_pred,
        "confidence": round(raw_conf, 4),
        "probabilities": {k: round(float(v), 4) for k, v in raw_probs.items()},
        "model_version": "1.0.0",
        "raw_model_prediction": raw_pred,
        "raw_model_confidence": round(raw_conf, 4),
        "raw_model_probabilities": {k: round(float(v), 4) for k, v in raw_probs.items()},
    }

    return {
        "raw_model": raw_model_out,
        "verified_hazards": verified_hazards_list,
        "hazard_status": hazard_status,
        "verified_hazard": verified_hazard,
        "official_live_evidence": {
            "active_flood_warning": live_evidence.get("active_flood_warning", False),
            "active_cyclone_warning": live_evidence.get("active_cyclone_warning", False),
            "official_alerts": official_alerts,
            "authorities": official_india.get("authorities", {}),
        },
        "live_evidence": live_evidence,
        "historical_context": historical_context,
        "official_alerts": official_alerts,
    }


def predict_disaster_live(
    lat: float,
    lon: float,
    overrides: Optional[Dict[str, Any]] = None,
    precomputed_sources: Optional[Dict[str, Any]] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes live disaster hazard inference with complete 4-layer architecture:
    Layer 1: 20-Feature Input Construction (exact trained model contract with granular provenance)
    Layer 2: Trained XGBoost inference (raw model prediction, confidence, probabilities)
    Layer 3: Multi-window historical & live evidence corroboration (USGS, GloFAS, GDACS, NASA FIRMS, SACHET, Official India)
    Layer 4: Operational Verified Hazard determination (handles multiple concurrent hazards)
    """
    t_start = time.perf_counter()
    overrides = overrides or {}
    precomputed = precomputed_sources or {}

    # 1. Resolve Location
    resolved = resolve_location(lat, lon)
    st = state or resolved.get("state") or "India"
    dt = district or resolved.get("district") or ""

    # 2. Build 20-Feature Vector with Granular Provenance (Phase 4 & 5)
    t_fb = time.perf_counter()
    d_built = build_disaster_features(
        lat=lat,
        lon=lon,
        overrides=overrides,
        precomputed_sources=precomputed,
        state=st,
        district=dt,
    )
    feature_build_ms = round((time.perf_counter() - t_fb) * 1000, 2)

    # Attach NASA FIRMS active fire hotspots telemetry
    if "firms_fire" in overrides:
        d_built["firms_fire"] = overrides["firms_fire"]
    elif "firms_fire" in precomputed:
        d_built["firms_fire"] = precomputed["firms_fire"]
    else:
        firms_res = fetch_firms_hotspots(lat, lon, overrides=overrides)
        d_built["firms_fire"] = firms_res
        if firms_res.get("missing_provider") or firms_res.get("status") == "unavailable":
            d_built.setdefault("missing_sources", []).append("NASA_FIRMS")
            d_built.setdefault("provider_status", {})["firms"] = "unavailable"
        else:
            d_built.setdefault("provider_status", {})["firms"] = "available"

    # SACHET provider status check
    if d_built.get("provider_status", {}).get("sachet") == "unavailable":
        d_built.setdefault("missing_sources", []).append("NDMA_SACHET")

    # Attach supporting surface hydrology indicators (waterlogging & drainage)
    if "waterlogging" in precomputed and precomputed["waterlogging"] is not None:
        wl_item = precomputed["waterlogging"]
        if isinstance(wl_item, dict):
            wl_p = (
                wl_item.get("positive_probability")
                or (wl_item.get("result") or {}).get("positive_probability")
            )
            if wl_p is not None:
                d_built["waterlogging_probability"] = float(wl_p)
    if "drainage" in precomputed and precomputed["drainage"] is not None:
        dr_item = precomputed["drainage"]
        if isinstance(dr_item, dict):
            dr_p = (
                dr_item.get("probability")
                or (dr_item.get("result") or {}).get("probability")
            )
            if dr_p is not None:
                d_built["drainage_congestion_probability"] = float(dr_p)
    if overrides.get("waterlogging_probability") is not None:
        d_built["waterlogging_probability"] = float(overrides["waterlogging_probability"])
    if overrides.get("drainage_congestion_probability") is not None:
        d_built["drainage_congestion_probability"] = float(overrides["drainage_congestion_probability"])

    missing_external = d_built.get("missing_external_features", [])
    if missing_external:
        return {
            "success": False,
            "error": f"Missing external disaster features: {', '.join(missing_external)}",
            "missing_features": missing_external,
            "input": d_built,
            "data_quality": {
                "status": "DATA_INCOMPLETE",
                "overall_quality": "DATA_INCOMPLETE",
                "fallback_sources": d_built.get("fallback_sources", []),
                "missing_sources": d_built.get("missing_sources", []),
                "missing_providers": d_built.get("missing_sources", []),
                "provider_status": d_built.get("provider_status", {}),
            },
            "verified_hazard": {
                "status": "DATA_INCOMPLETE",
                "hazard": None,
                "active": False,
                "severity": "LOW",
                "reason": f"Essential hazard features are missing: {', '.join(missing_external)}",
                "alert_title": "Disaster data incomplete",
            },
            "verified_hazards": [],
            "hazard_status": {},
        }

    # 3. Model Inference (Preserves Exact 20-Feature Contract & RobustScaler) (Phase 6)
    t_inf = time.perf_counter()
    raw_res = registry.predict("disaster", d_built["features"])
    inference_ms = round((time.perf_counter() - t_inf) * 1000, 2)

    # 4. Fetch Multi-Window Historical Context (Phase 7)
    t_hist = time.perf_counter()
    history_data = get_disaster_history(lat, lon, precomputed_sources=precomputed)
    hist_ms = round((time.perf_counter() - t_hist) * 1000, 2)

    # 5. Evidence Corroboration Engine (Phase 8-17)
    corroboration = corroborate_disaster_evidence(
        raw_model_result=raw_res,
        d_built=d_built,
        history_data=history_data,
        location_meta={"lat": lat, "lon": lon, "state": st, "district": dt},
    )

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    missing_provs = list(dict.fromkeys(d_built.get("missing_sources", [])))
    dq_status = (
        "DATA_INCOMPLETE" if d_built.get("model_feature_quality") == "DATA_INCOMPLETE"
        else ("PARTIAL_DATA" if (d_built.get("model_feature_quality") in ("PARTIAL_DATA", "FALLBACK_USED") or bool(missing_provs))
        else "FULL_DATA")
    )

    # 6. Smart Alert Engine Lifecycle Processing (Event-driven, fails gracefully)
    alerts_result = {
        "created_alerts": [],
        "updated_alerts": [],
        "resolved_alerts": [],
        "expired_alerts": [],
        "active_alerts": [],
    }
    try:
        from backend.services.alert_engine import default_alert_engine
        alerts_result = default_alert_engine.process_operational_hazards(
            verified_hazards=corroboration.get("verified_hazards", []),
            location={"lat": lat, "lon": lon, "state": st, "district": dt},
            live_evidence=corroboration.get("live_evidence", {}),
            data_quality={
                "status": dq_status,
                "missing_providers": missing_provs,
                "provider_status": d_built.get("provider_status", {}),
            },
        )
    except Exception as a_exc:
        # Section 14: Alert processing must fail gracefully without breaking prediction
        alerts_result["error"] = str(a_exc)

    return {
        "success": True,
        "location": {"lat": round(lat, 4), "lon": round(lon, 4), "state": st, "district": dt},
        # Operational Multi-Hazard Architecture
        "raw_model": corroboration["raw_model"],
        "verified_hazards": corroboration["verified_hazards"],
        "hazard_status": corroboration.get("hazard_status", {}),
        "verified_hazard": corroboration["verified_hazard"],
        "official_live_evidence": corroboration["official_live_evidence"],
        "live_evidence": corroboration["live_evidence"],
        "historical_context": corroboration["historical_context"],
        "official_alerts": corroboration["official_alerts"],
        "fire_hotspots": corroboration["live_evidence"].get("fire_hotspots", []),
        "alerts": alerts_result,
        "active_alerts": alerts_result.get("active_alerts", []),

        "data_quality": {
            "status": dq_status,
            "overall_quality": d_built.get("model_feature_quality", "FULL_DATA"),
            "fallback_sources": d_built.get("fallback_sources", []),
            "missing_sources": missing_provs,
            "missing_providers": missing_provs,
            "provider_status": d_built.get("provider_status", {}),
        },
        # Granular Data Quality Provenance (Phase 18)
        "model_feature_quality": d_built.get("model_feature_quality", "FULL_DATA"),
        "features_provenance": d_built.get("features_provenance", {}),
        "provider_status": d_built.get("provider_status", {}),
        "fallback_sources": d_built.get("fallback_sources", []),
        "missing_sources": missing_provs,
        # Backward compatibility aliases
        "raw_model_prediction": corroboration["raw_model"]["prediction"],
        "raw_model_confidence": corroboration["raw_model"]["confidence"],
        "raw_model_probabilities": corroboration["raw_model"]["probabilities"],
        "input": d_built,
        "result": raw_res,
        "prediction": corroboration["raw_model"]["prediction"],
        "confidence": corroboration["raw_model"]["confidence"],
        "probabilities": corroboration["raw_model"]["probabilities"],
        "timings": {
            "feature_build_ms": feature_build_ms,
            "inference_ms": inference_ms,
            "history_ms": hist_ms,
            "total_ms": total_ms,
        },
    }

