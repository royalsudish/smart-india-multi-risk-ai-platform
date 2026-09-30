from pathlib import Path
import json
import joblib

from backend.services.model_service import registry
from backend.services.waterlogging_features import build_live_features

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = (
    ROOT / "models" / "waterlogging"
    / "waterlogging_alert_threshold.joblib"
)
METADATA_PATH = (
    ROOT / "models" / "waterlogging"
    / "metadata.json"
)

DEFAULT_THRESHOLD = 0.36


def get_warning_threshold():
    if THRESHOLD_PATH.exists():
        try:
            obj = joblib.load(THRESHOLD_PATH)
            if isinstance(obj, dict) and "threshold" in obj:
                return float(obj["threshold"])
            if isinstance(obj, (int, float)):
                return float(obj)
        except Exception:
            pass

    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if "warning_threshold" in meta:
                return float(meta["warning_threshold"])
        except Exception:
            pass

    return DEFAULT_THRESHOLD


def risk_label(probability, threshold):
    p = float(probability)

    if p < threshold * 0.50:
        return "LOW"
    if p < threshold:
        return "MEDIUM"
    return "HIGH"


import time


def predict_live(lat, lon, state="LIVE", district="LIVE", precomputed_sources=None):
    t_start = time.perf_counter()

    raw_hourly = None
    raw_terrain = None
    if precomputed_sources:
        raw_hourly = precomputed_sources.get("waterlogging_forecast") or precomputed_sources.get("waterlogging")
        raw_terrain = precomputed_sources.get("terrain")

    t_fb_start = time.perf_counter()
    built = build_live_features(
        float(lat),
        float(lon),
        state=state,
        district=district,
        raw_hourly_forecast=raw_hourly,
        raw_terrain=raw_terrain,
    )
    feature_build_ms = round((time.perf_counter() - t_fb_start) * 1000, 2)
    network_ms = 0.0 if (raw_hourly is not None and raw_terrain is not None) else feature_build_ms

    t_inf_start = time.perf_counter()
    result = registry.predict(
        "waterlogging",
        built["features"],
    )
    inference_ms = round((time.perf_counter() - t_inf_start) * 1000, 2)

    probs = result.get("probabilities")
    positive_probability = None

    if isinstance(probs, list) and len(probs) >= 2:
        positive_probability = float(probs[1])
    elif result.get("positive_probability") is not None:
        positive_probability = float(
            result["positive_probability"]
        )

    threshold = get_warning_threshold()

    if positive_probability is not None:
        result["positive_probability"] = positive_probability
        result["warning_threshold"] = threshold
        result["warning"] = bool(
            positive_probability >= threshold
        )
        result["risk_level"] = risk_label(
            positive_probability,
            threshold,
        )

    # Attach Phase A rainfall transparency & outlook structures
    result["recent_rainfall"] = built.get("recent_rainfall")
    result["forecast_rainfall"] = built.get("forecast_rainfall")
    result["future_outlook"] = built.get("future_outlook")
    result["analysis_time_ist"] = built.get("analysis_time_ist")
    result["terrain"] = built.get("terrain")

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "input": built,
        "result": result,
        "timings": {
            "network_ms": network_ms,
            "feature_build_ms": feature_build_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }
