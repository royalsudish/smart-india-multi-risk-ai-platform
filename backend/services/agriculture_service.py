from backend.data_sources.nominatim import search_place
from backend.services.agriculture_features import assemble_agriculture_payload
from backend.services.model_service import registry


def resolve_location(place_name=None, lat=None, lon=None):
    if lat is not None and lon is not None:
        return {
            "lat": float(lat),
            "lon": float(lon),
            "display_name": None,
        }

    if not place_name:
        raise ValueError(
            "Provide either place_name or both lat and lon."
        )

    results = search_place(place_name, limit=1)
    if not results:
        raise ValueError(
            f"Could not resolve place '{place_name}'."
        )

    first = results[0]
    return {
        "lat": float(first["lat"]),
        "lon": float(first["lon"]),
        "display_name": first.get("display_name"),
    }


import time


def predict_live(
    crop,
    crop_stage,
    place_name=None,
    lat=None,
    lon=None,
    precomputed_sources=None,
):
    t_start = time.perf_counter()
    loc = resolve_location(
        place_name=place_name,
        lat=lat,
        lon=lon,
    )

    raw_daily = (precomputed_sources or {}).get("agriculture_forecast")
    if raw_daily is None and precomputed_sources:
        raw_daily = precomputed_sources.get("agriculture")

    t_fb_start = time.perf_counter()
    payload = assemble_agriculture_payload(
        loc["lat"],
        loc["lon"],
        crop,
        crop_stage,
        raw_daily_forecast=raw_daily,
    )
    t_fb_end = time.perf_counter()
    feature_build_ms = round((t_fb_end - t_fb_start) * 1000, 2)
    network_ms = 0.0 if raw_daily is not None else feature_build_ms

    t_inf_start = time.perf_counter()
    result = registry.predict(
        "agriculture",
        payload,
    )
    inference_ms = round((time.perf_counter() - t_inf_start) * 1000, 2)
    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "location": loc,
        "crop": crop,
        "crop_stage": crop_stage,
        "input_features": payload,
        "result": result,
        "timings": {
            "network_ms": network_ms,
            "feature_build_ms": feature_build_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }
