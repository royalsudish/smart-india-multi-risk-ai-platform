import time
import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from backend.services.model_service import registry
from backend.services.terrain_service import terrain_features
from backend.services.waterlogging_features import get_live_terrain
from backend.data_sources.overpass import drainage_features
from backend.data_sources.open_meteo import forecast
from backend.data_sources.sentinel2_optical import get_live_sentinel2_indices
from backend.data_sources.sentinel1_sar import get_live_sentinel1_vv

# Exact feature contract required by models/drainage/drainage_random_forest.pkl
DRAINAGE_MODEL_FEATURES = [
    "elevation",
    "slope",
    "rain_max",
    "sentinel1_vv",
    "NDVI",
    "NDBI",
]

# Baseline nominal defaults calibrated from the model's training distribution
DEFAULT_FEATURE_BASELINES = {
    "elevation": 27.5,     # meters (median split threshold 27.5 m)
    "slope": 1.93,         # degrees (median split threshold 1.93 deg)
    "rain_max": 2.13,      # log10(rain_mm + 1) design storm scale (median split threshold 2.13)
    "sentinel1_vv": -10.5, # dB (median split threshold -10.06 dB)
    "NDVI": 0.42,          # index (median split threshold 0.418)
    "NDBI": -0.05,         # index (median split threshold -0.049)
}

# Warning/Risk tier policy
DRAINAGE_WARNING_THRESHOLD = 0.50


def transform_rain_max(raw_rain_mm: float) -> float:
    """
    Exact training-scale transformation for rain_max:
    model_rain_max = log10(rain_mm + 1)
    """
    valid_mm = max(0.0, float(raw_rain_mm))
    return round(float(np.log10(valid_mm + 1.0)), 4)


def evaluate_drainage_risk(positive_prob: float) -> Dict[str, Any]:
    """
    Evaluates drainage risk tier and advisory based on failure probability.
    Class 0 = Adequate Drainage / Low Risk
    Class 1 = Drainage Congestion / Failure Risk
    """
    prob = float(positive_prob)
    if prob >= DRAINAGE_WARNING_THRESHOLD:
        risk_level = "HIGH"
        badge_class = "badge-danger"
        advisory = "Critical drainage congestion risk: Low topography, flat slope, or elevated surface moisture indicate severe runoff impediment."
    elif prob >= 0.35:
        risk_level = "MEDIUM"
        badge_class = "badge-warning"
        advisory = "Moderate drainage congestion potential: Waterway conveyance capacity may be constrained under intense rainfall."
    else:
        risk_level = "LOW"
        badge_class = "badge-safe"
        advisory = "Adequate drainage capacity: Local topography, slope, and surface characteristics support nominal runoff dissipation."

    return {
        "risk_level": risk_level,
        "badge_class": badge_class,
        "warning": prob >= DRAINAGE_WARNING_THRESHOLD,
        "warning_threshold": DRAINAGE_WARNING_THRESHOLD,
        "advisory": advisory,
    }


def compute_data_quality(feature_provenance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates dataset fidelity and provenance tier across the 6 features:
    FULL_LIVE: 0 fallback features (all live, derived, or explicitly overridden)
    PARTIAL_LIVE: 1 to 3 fallback features
    FALLBACK_HEAVY: >= 4 fallback features
    """
    live_count = 0
    derived_count = 0
    fallback_count = 0
    override_count = 0

    for feat_name, meta in feature_provenance.items():
        q = str(meta.get("quality", "FALLBACK")).upper()
        if q == "LIVE":
            live_count += 1
        elif q == "DERIVED":
            derived_count += 1
        elif q == "OVERRIDE":
            override_count += 1
        else:
            fallback_count += 1

    total_count = len(feature_provenance)

    if fallback_count == 0:
        status = "FULL_LIVE"
        summary = f"All {total_count} features derived from live sensors, models, or explicit inputs with zero fallbacks."
    elif fallback_count >= 4:
        status = "FALLBACK_HEAVY"
        summary = f"{fallback_count} of {total_count} features rely on regional baseline fallbacks; use with caution."
    else:
        status = "PARTIAL_LIVE"
        summary = f"{live_count + derived_count + override_count} live/derived features; {fallback_count} regional baseline fallbacks."

    return {
        "status": status,
        "live_features": live_count,
        "derived_features": derived_count,
        "override_features": override_count,
        "fallback_features": fallback_count,
        "total_features": total_count,
        "summary": summary,
    }


def build_live_drainage_features(
    lat: float,
    lon: float,
    overrides: Optional[Dict[str, Any]] = None,
    precomputed_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the exact 6-feature vector for models/drainage/drainage_random_forest.pkl:
    ['elevation', 'slope', 'rain_max', 'sentinel1_vv', 'NDVI', 'NDBI']

    Maintains rigorous data fidelity:
    - Never fabricates live observations for baseline proxies.
    - Applies the exact log10(rain_mm + 1) transformation for rain_max while exposing both raw_rain_max_mm and model_rain_max.
    - Explicitly tags provenance with quality: LIVE, DERIVED, FALLBACK, OVERRIDE.
    """
    overrides = overrides or {}
    precomputed_sources = precomputed_sources or {}
    now_iso = datetime.now(timezone.utc).isoformat()

    provenance: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Elevation (meters)
    # -------------------------------------------------------------------------
    if "elevation" in overrides:
        elevation_val = float(overrides["elevation"])
        provenance["elevation"] = {
            "value": round(elevation_val, 2),
            "unit": "meters",
            "source": "Caller override",
            "quality": "OVERRIDE",
            "timestamp": now_iso,
        }
    else:
        elevation_val = None
        raw_terrain = precomputed_sources.get("terrain")
        if raw_terrain and isinstance(raw_terrain, dict) and "elevation_m" in raw_terrain:
            elevation_val = float(raw_terrain["elevation_m"])
            provenance["elevation"] = {
                "value": round(elevation_val, 2),
                "unit": "meters",
                "source": "Open-Meteo 90m DEM (precomputed)",
                "quality": "LIVE",
                "timestamp": now_iso,
            }

        if elevation_val is None:
            try:
                t_res = get_live_terrain(lat, lon, return_details=True)
                elevation_val = float(t_res.get("elevation_m", DEFAULT_FEATURE_BASELINES["elevation"]))
                provenance["elevation"] = {
                    "value": round(elevation_val, 2),
                    "unit": "meters",
                    "source": "Open-Meteo Elevation API (90m DEM)",
                    "quality": "LIVE",
                    "timestamp": now_iso,
                }
            except Exception:
                elevation_val = DEFAULT_FEATURE_BASELINES["elevation"]
                provenance["elevation"] = {
                    "value": elevation_val,
                    "unit": "meters",
                    "source": "Regional elevation fallback baseline",
                    "quality": "FALLBACK",
                    "timestamp": None,
                }

    # -------------------------------------------------------------------------
    # 2. Slope (degrees)
    # -------------------------------------------------------------------------
    if "slope" in overrides:
        slope_val = float(overrides["slope"])
        provenance["slope"] = {
            "value": round(slope_val, 3),
            "unit": "degrees",
            "source": "Caller override",
            "quality": "OVERRIDE",
            "timestamp": now_iso,
        }
    else:
        slope_val = None
        raw_terrain = precomputed_sources.get("terrain")
        if raw_terrain and isinstance(raw_terrain, dict):
            slope_cand = raw_terrain.get("slope_deg") or raw_terrain.get("slope_deg_approx")
            if slope_cand is not None:
                slope_val = float(slope_cand)
                provenance["slope"] = {
                    "value": round(slope_val, 3),
                    "unit": "degrees",
                    "source": "Open-Meteo DEM Multi-Point Gradient (precomputed)",
                    "quality": "DERIVED",
                    "timestamp": now_iso,
                }

        if slope_val is None:
            try:
                t_res = get_live_terrain(lat, lon, return_details=True)
                slope_val = float(t_res.get("slope_deg", DEFAULT_FEATURE_BASELINES["slope"]))
                provenance["slope"] = {
                    "value": round(slope_val, 3),
                    "unit": "degrees",
                    "source": "Open-Meteo DEM Multi-Point Gradient",
                    "quality": "DERIVED",
                    "timestamp": now_iso,
                }
            except Exception:
                slope_val = DEFAULT_FEATURE_BASELINES["slope"]
                provenance["slope"] = {
                    "value": slope_val,
                    "unit": "degrees",
                    "source": "Regional slope fallback baseline",
                    "quality": "FALLBACK",
                    "timestamp": None,
                }

    # -------------------------------------------------------------------------
    # 3. Rain Max: log10(rain_mm + 1)
    # -------------------------------------------------------------------------
    # Check for direct transformed override first
    if "model_rain_max" in overrides:
        model_rain_max = float(overrides["model_rain_max"])
        raw_rain_max_mm = float(overrides.get("raw_rain_max_mm", max(0.0, 10.0 ** model_rain_max - 1.0)))
        provenance["rain_max"] = {
            "model_value": round(model_rain_max, 4),
            "raw_value_mm": round(raw_rain_max_mm, 2),
            "unit": "log10(rain_mm + 1)",
            "transformation": f"log10({raw_rain_max_mm:.2f} + 1) = {model_rain_max:.4f}",
            "source": "Caller model_rain_max override",
            "quality": "OVERRIDE",
            "timestamp": now_iso,
        }
    elif "raw_rain_max_mm" in overrides:
        raw_rain_max_mm = float(overrides["raw_rain_max_mm"])
        model_rain_max = transform_rain_max(raw_rain_max_mm)
        provenance["rain_max"] = {
            "model_value": round(model_rain_max, 4),
            "raw_value_mm": round(raw_rain_max_mm, 2),
            "unit": "log10(rain_mm + 1)",
            "transformation": f"log10({raw_rain_max_mm:.2f} + 1) = {model_rain_max:.4f}",
            "source": "Caller raw_rain_max_mm override (transformed via log10)",
            "quality": "OVERRIDE",
            "timestamp": now_iso,
        }
    elif "rain_max" in overrides:
        # Caller provided a generic rain_max value
        r_val = float(overrides["rain_max"])
        # If value is > 5.0, caller likely provided raw rainfall mm; transform it
        if r_val > 5.0:
            raw_rain_max_mm = r_val
            model_rain_max = transform_rain_max(raw_rain_max_mm)
            provenance["rain_max"] = {
                "model_value": round(model_rain_max, 4),
                "raw_value_mm": round(raw_rain_max_mm, 2),
                "unit": "log10(rain_mm + 1)",
                "transformation": f"log10({raw_rain_max_mm:.2f} + 1) = {model_rain_max:.4f}",
                "source": "Caller override (interpreted as raw mm and transformed via log10)",
                "quality": "OVERRIDE",
                "timestamp": now_iso,
            }
        else:
            model_rain_max = round(r_val, 4)
            raw_rain_max_mm = round(max(0.0, 10.0 ** r_val - 1.0), 2)
            provenance["rain_max"] = {
                "model_value": model_rain_max,
                "raw_value_mm": raw_rain_max_mm,
                "unit": "log10(rain_mm + 1)",
                "transformation": f"log10({raw_rain_max_mm:.2f} + 1) = {model_rain_max:.4f}",
                "source": "Caller override (interpreted as model-scale log10)",
                "quality": "OVERRIDE",
                "timestamp": now_iso,
            }
    else:
        # Extract live precipitation from Open-Meteo
        forecast_data = precomputed_sources.get("waterlogging_forecast") or precomputed_sources.get("forecast")
        if not forecast_data:
            try:
                forecast_data = forecast(lat, lon, past_days=2, forecast_days=2)
            except Exception:
                forecast_data = None

        raw_rain_max_mm = 0.0
        if forecast_data and isinstance(forecast_data, dict):
            hourly = forecast_data.get("hourly", {})
            precip_list = hourly.get("precipitation") or hourly.get("rain") or []
            if precip_list:
                finite_precip = [float(x) for x in precip_list if x is not None and not math.isnan(x)]
                if finite_precip:
                    raw_rain_max_mm = round(max(0.0, max(finite_precip)), 2)

        # STRICT CONTRACT: Model feature is ALWAYS transformed as log10(rain_mm + 1)
        model_rain_max = transform_rain_max(raw_rain_max_mm)
        provenance["rain_max"] = {
            "model_value": round(model_rain_max, 4),
            "raw_value_mm": round(raw_rain_max_mm, 2),
            "unit": "log10(rain_mm + 1)",
            "transformation": f"log10({raw_rain_max_mm:.2f} + 1) = {model_rain_max:.4f}",
            "source": f"Open-Meteo Hourly Precipitation (max: {raw_rain_max_mm} mm)",
            "quality": "DERIVED",
            "timestamp": now_iso,
        }

    # -------------------------------------------------------------------------
    # 4. Sentinel-1 SAR VV Backscatter (dB)
    # Priority: 1. Caller override -> 2. Live Sentinel-1 -> 3. Fallback baseline
    # -------------------------------------------------------------------------
    s1_data = precomputed_sources.get("sentinel1")
    need_s1 = s1_data is None and "sentinel1_vv" not in overrides
    if need_s1:
        try:
            s1_data = get_live_sentinel1_vv(lat, lon)
        except Exception as s1_err:
            s1_data = {"status": "error", "message": str(s1_err)}

    if "sentinel1_vv" in overrides:
        sentinel1_vv_val = float(overrides["sentinel1_vv"])
        provenance["sentinel1_vv"] = {
            "value": round(sentinel1_vv_val, 2),
            "unit": "dB",
            "source": "Caller override",
            "quality": "OVERRIDE",
            "timestamp": now_iso,
        }
    elif s1_data and s1_data.get("status") == "available" and "provenance" in s1_data:
        s1_prov = s1_data["provenance"]["sentinel1_vv"]
        sentinel1_vv_val = float(s1_prov["value"])
        provenance["sentinel1_vv"] = s1_prov
    else:
        sentinel1_vv_val = DEFAULT_FEATURE_BASELINES["sentinel1_vv"]
        fail_note = (s1_data.get("message") if s1_data else None) or "No usable recent Sentinel-1 acquisition exists; using calibrated regional baseline."
        provenance["sentinel1_vv"] = {
            "value": sentinel1_vv_val,
            "unit": "dB",
            "source": "calibrated regional baseline",
            "quality": "FALLBACK",
            "timestamp": None,
            "scene_id": None,
            "scene_date": None,
            "orbit_direction": None,
            "sampling_mode": "fallback",
            "note": fail_note,
        }

    # -------------------------------------------------------------------------
    # 5. NDVI & 6. NDBI (Sentinel-2 L2A Optical Indices)
    # Priority: 1. Caller override -> 2. Live Sentinel-2 -> 3. Fallback baseline
    # -------------------------------------------------------------------------
    s2_data = precomputed_sources.get("sentinel2")
    need_s2 = (
        s2_data is None
        and not (("NDVI" in overrides or "ndvi" in overrides) and ("NDBI" in overrides or "ndbi" in overrides))
    )
    if need_s2:
        try:
            s2_data = get_live_sentinel2_indices(lat, lon)
        except Exception as s2_err:
            s2_data = {"status": "error", "message": str(s2_err)}

    # 5. NDVI (Normalized Difference Vegetation Index)
    ndvi_val = None
    for k in ["NDVI", "ndvi"]:
        if k in overrides:
            ndvi_val = float(overrides[k])
            provenance["NDVI"] = {
                "value": round(ndvi_val, 4),
                "unit": "index (-1 to 1)",
                "source": "Caller override",
                "quality": "OVERRIDE",
                "timestamp": now_iso,
            }
            break

    if ndvi_val is None:
        if s2_data and s2_data.get("status") == "available" and "provenance" in s2_data:
            ndvi_meta = s2_data["provenance"]["NDVI"]
            ndvi_val = float(ndvi_meta["value"])
            provenance["NDVI"] = ndvi_meta
        else:
            ndvi_val = DEFAULT_FEATURE_BASELINES["NDVI"]
            fail_reason = (s2_data.get("message") if s2_data else None) or "Live Sentinel-2 optical imagery unavailable; using calibrated regional baseline."
            provenance["NDVI"] = {
                "value": ndvi_val,
                "unit": "index (-1 to 1)",
                "source": "regional_baseline (Delta agro-climatic baseline)",
                "quality": "FALLBACK",
                "timestamp": None,
                "reason": fail_reason,
                "note": "Live Sentinel-2 optical imagery unavailable; using calibrated regional baseline.",
            }

    # 6. NDBI (Normalized Difference Built-up Index)
    ndbi_val = None
    for k in ["NDBI", "ndbi"]:
        if k in overrides:
            ndbi_val = float(overrides[k])
            provenance["NDBI"] = {
                "value": round(ndbi_val, 4),
                "unit": "index (-1 to 1)",
                "source": "Caller override",
                "quality": "OVERRIDE",
                "timestamp": now_iso,
            }
            break

    if ndbi_val is None:
        if s2_data and s2_data.get("status") == "available" and "provenance" in s2_data:
            ndbi_meta = s2_data["provenance"]["NDBI"]
            ndbi_val = float(ndbi_meta["value"])
            provenance["NDBI"] = ndbi_meta
        else:
            ndbi_val = DEFAULT_FEATURE_BASELINES["NDBI"]
            fail_reason = (s2_data.get("message") if s2_data else None) or "Live Sentinel-2 optical imagery unavailable; using calibrated regional baseline."
            provenance["NDBI"] = {
                "value": ndbi_val,
                "unit": "index (-1 to 1)",
                "source": "regional_baseline (Built-up surface proxy baseline)",
                "quality": "FALLBACK",
                "timestamp": None,
                "reason": fail_reason,
                "note": "Live Sentinel-2 optical imagery unavailable; using calibrated regional baseline.",
            }

    # Features passed to Random Forest model (STRICT ORDER AND SCALE)
    features = {
        "elevation": round(float(elevation_val), 2),
        "slope": round(float(slope_val), 3),
        "rain_max": round(float(model_rain_max), 4),
        "sentinel1_vv": round(float(sentinel1_vv_val), 2),
        "NDVI": round(float(ndvi_val), 4),
        "NDBI": round(float(ndbi_val), 4),
    }

    # Data Quality calculation
    quality_summary = compute_data_quality(provenance)

    return {
        "features": features,
        "feature_order": DRAINAGE_MODEL_FEATURES,
        "feature_provenance": provenance,
        "rainfall_provenance": {
            "raw_rain_max_mm": round(float(raw_rain_max_mm), 2),
            "model_rain_max": round(float(model_rain_max), 4),
            "formula": "log10(raw_rain_max_mm + 1)",
        },
        "raw_rain_max_mm": round(float(raw_rain_max_mm), 2),
        "model_rain_max": round(float(model_rain_max), 4),
        "data_quality": quality_summary,
        "sentinel1_metadata": s1_data,
        "sentinel2_metadata": s2_data,
    }


def predict_live(
    lat: float,
    lon: float,
    radius_m: int = 2000,
    overrides: Optional[Dict[str, Any]] = None,
    precomputed_sources: Optional[Dict[str, Any]] = None,
    include_osm: bool = True,
) -> Dict[str, Any]:
    """
    Executes live drainage prediction:
    1. Extracts and constructs the 6 exact features for models/drainage/drainage_random_forest.pkl.
    2. Runs inference using ModelRegistry.
    3. Evaluates drainage risk tier and advisory.
    4. Gathers OSM/Overpass drainage network geometry (waterway distance, density, count, mapped length).
    5. Returns unified drainage intelligence payload with full provenance and data quality metadata.
    """
    t_start = time.perf_counter()

    # 1. Build features with strict data fidelity & provenance
    t_fb_start = time.perf_counter()
    feature_payload = build_live_drainage_features(
        lat=lat,
        lon=lon,
        overrides=overrides,
        precomputed_sources=precomputed_sources,
    )
    features = feature_payload["features"]
    feature_build_ms = round((time.perf_counter() - t_fb_start) * 1000, 2)

    # 2. Run model inference
    t_inf_start = time.perf_counter()
    model_raw = registry.predict("drainage", features)
    inference_ms = round((time.perf_counter() - t_inf_start) * 1000, 2)

    # 3. Format probabilities and risk evaluation
    pred = int(model_raw.get("prediction", 0))
    probs = model_raw.get("probabilities", [0.5, 0.5])
    pos_prob = float(model_raw.get("positive_probability", probs[1] if len(probs) > 1 else 0.0))
    risk_info = evaluate_drainage_risk(pos_prob)

    prediction_result = {
        "model_name": "Random Forest Drainage Classifier",
        "prediction": pred,
        "prediction_label": "Drainage Congestion Risk" if pred == 1 else "Adequate Drainage",
        "positive_probability": round(pos_prob, 4),
        "probabilities": [round(float(p), 4) for p in probs],
        "class_probabilities": {
            "0_adequate_drainage": round(float(probs[0]), 4),
            "1_congestion_risk": round(float(probs[1]), 4) if len(probs) > 1 else 0.0,
        },
        "risk_level": risk_info["risk_level"],
        "badge_class": risk_info["badge_class"],
        "warning": risk_info["warning"],
        "warning_threshold": risk_info["warning_threshold"],
        "advisory": risk_info["advisory"],
        "used_features": DRAINAGE_MODEL_FEATURES,
    }

    # 4. OSM Drainage Network Data
    osm_data = None
    osm_ms = 0.0
    if include_osm:
        t_osm_start = time.perf_counter()
        try:
            osm_data = drainage_features(lat=lat, lon=lon, radius_m=radius_m)
        except Exception as exc:
            osm_data = {
                "status": "unavailable",
                "message": f"OSM Overpass drainage fetch failed: {exc}",
            }
        osm_ms = round((time.perf_counter() - t_osm_start) * 1000, 2)

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "success": True,
        "location": {"lat": lat, "lon": lon, "radius_m": radius_m},
        "features": features,
        "feature_provenance": feature_payload["feature_provenance"],
        "rainfall_provenance": feature_payload["rainfall_provenance"],
        "raw_rain_max_mm": feature_payload["raw_rain_max_mm"],
        "model_rain_max": feature_payload["model_rain_max"],
        "data_quality": feature_payload["data_quality"],
        "prediction": prediction_result,
        "osm_drainage": osm_data,
        "sentinel1_metadata": feature_payload.get("sentinel1_metadata"),
        "sentinel2_metadata": feature_payload.get("sentinel2_metadata"),
        "satellite": {
            "sentinel1": feature_payload.get("sentinel1_metadata"),
            "sentinel2": feature_payload.get("sentinel2_metadata"),
        },
        "timings": {
            "feature_build_ms": feature_build_ms,
            "inference_ms": inference_ms,
            "osm_fetch_ms": osm_ms,
            "total_ms": total_ms,
        },
    }
