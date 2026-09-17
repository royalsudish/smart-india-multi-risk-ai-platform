from flask import Blueprint, request
from backend.utils.response import ok, fail
from backend.services.model_service import registry, MODEL_NAMES
from backend.services.weather_features import build_live_weather_features
from backend.services.disaster_features import build_disaster_features
from backend.services.terrain_service import terrain_features
from backend.services.waterlogging_history import query as query_waterlogging_history
from backend.services.waterlogging_service import predict_live as predict_waterlogging_live
from backend.services.agriculture_service import predict_live as predict_agriculture_live
from backend.data_sources.nominatim import search_place
from backend.data_sources.overpass import drainage_features
from backend.data_sources.bhuvan import list_layers, getmap_url
from backend.data_sources.mosdac import info as mosdac_info
from backend.data_sources.open_meteo import (
    forecast, historical, air_quality, elevation, marine
)

api_bp = Blueprint("api", __name__)

def _coords():
    return float(request.args["lat"]), float(request.args["lon"])

@api_bp.get("/health")
def health():
    return ok({
        "service": "Smart Multi-Risk Platform",
        "model_files": registry.status(try_load=False),
    })

@api_bp.get("/models/status")
def models_status():
    try_load = request.args.get("load", "0") in ("1", "true", "yes")
    return ok(registry.status(try_load=try_load))

@api_bp.post("/models/reload")
def reload_models():
    registry.reload()
    return ok(registry.status(False), "Model cache cleared")

@api_bp.post("/predict/<model_name>")
def predict_one(model_name):
    if model_name not in MODEL_NAMES:
        return fail(f"Unknown model: {model_name}", 404)
    payload = request.get_json(silent=True) or {}
    features = payload.get("features", payload)
    try:
        return ok(registry.predict(model_name, features))
    except Exception as exc:
        return fail("Prediction failed", 400, str(exc))

@api_bp.post("/predict/weather/live")
def predict_weather_live():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
        built = build_live_weather_features(lat, lon)
        result = registry.predict("weather", built["features"])
        return ok({"input": built, "result": result})
    except Exception as exc:
        return fail("Live weather prediction failed", 400, str(exc))

@api_bp.post("/predict/disaster/live")
def predict_disaster_live():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
        built = build_disaster_features(lat, lon, payload.get("overrides") or {})
        if built["missing_external_features"]:
            return fail(
                "More disaster features are required before inference",
                400,
                built
            )
        result = registry.predict("disaster", built["features"])
        return ok({"input": built, "result": result})
    except Exception as exc:
        return fail("Live disaster prediction failed", 400, str(exc))

@api_bp.get("/sources/geocode")
def geocode():
    try:
        return ok(search_place(request.args.get("q"), request.args.get("limit", 5)))
    except Exception as exc:
        return fail("Geocoding failed", 400, str(exc))

@api_bp.get("/sources/weather")
def source_weather():
    try:
        lat, lon = _coords()
        return ok(forecast(lat, lon))
    except Exception as exc:
        return fail("Weather API failed", 400, str(exc))

@api_bp.get("/sources/weather-features")
def source_weather_features():
    try:
        lat, lon = _coords()
        return ok(build_live_weather_features(lat, lon))
    except Exception as exc:
        return fail("Weather feature creation failed", 400, str(exc))

@api_bp.get("/sources/air-quality")
def source_air():
    try:
        lat, lon = _coords()
        return ok(air_quality(lat, lon))
    except Exception as exc:
        return fail("Air-quality API failed", 400, str(exc))

@api_bp.get("/sources/elevation")
def source_elevation():
    try:
        lat, lon = _coords()
        return ok({"elevation_m": elevation(lat, lon)})
    except Exception as exc:
        return fail("Elevation API failed", 400, str(exc))

@api_bp.get("/sources/terrain")
def source_terrain():
    try:
        lat, lon = _coords()
        return ok(terrain_features(lat, lon))
    except Exception as exc:
        return fail("Terrain feature creation failed", 400, str(exc))

@api_bp.get("/sources/marine")
def source_marine():
    try:
        lat, lon = _coords()
        return ok(marine(lat, lon))
    except Exception as exc:
        return fail("Marine API failed", 400, str(exc))

@api_bp.get("/sources/drainage")
def source_drainage():
    try:
        lat, lon = _coords()
        radius = int(request.args.get("radius_m", 2000))
        return ok(drainage_features(lat, lon, radius))
    except Exception as exc:
        return fail("Overpass drainage query failed", 400, str(exc))

@api_bp.get("/sources/bhuvan/layers")
def bhuvan_layers():
    try:
        return ok(list_layers(int(request.args.get("limit", 100))))
    except Exception as exc:
        return fail("Bhuvan WMS capabilities failed", 400, str(exc))

@api_bp.post("/sources/bhuvan/getmap-url")
def bhuvan_map():
    payload = request.get_json(silent=True) or {}
    try:
        return ok({
            "url": getmap_url(
                payload["layer"],
                payload["bbox"],
                payload.get("width", 800),
                payload.get("height", 600)
            )
        })
    except Exception as exc:
        return fail("Bhuvan GetMap URL failed", 400, str(exc))

@api_bp.get("/sources/mosdac/info")
def mosdac():
    return ok(mosdac_info())





@api_bp.post("/predict/agriculture/live")
def agriculture_live_prediction():
    payload = request.get_json(silent=True) or {}
    try:
        crop = payload["crop"]
        crop_stage = payload["crop_stage"]

        result = predict_agriculture_live(
            crop=crop,
            crop_stage=crop_stage,
            place_name=payload.get("place_name"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
        )
        return ok(result)
    except Exception as exc:
        return fail(
            "Live agriculture prediction failed",
            400,
            str(exc),
        )


@api_bp.post("/predict/waterlogging/live")
def waterlogging_live_prediction():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
        state = payload.get("state", "LIVE")
        district = payload.get("district", "LIVE")

        return ok(
            predict_waterlogging_live(
                lat,
                lon,
                state=state,
                district=district,
            )
        )
    except Exception as exc:
        return fail(
            "Live waterlogging prediction failed",
            400,
            str(exc),
        )


@api_bp.get("/waterlogging/history")
def waterlogging_history():
    try:
        return ok(query_waterlogging_history(
            request.args.get("state"),
            request.args.get("district"),
            request.args.get("limit", 200)
        ))
    except Exception as exc:
        return fail("Waterlogging prediction history failed", 400, str(exc))
