import time
from concurrent.futures import ThreadPoolExecutor
from flask import Blueprint, request, jsonify
from backend.utils.response import ok, fail
from backend.services.model_service import registry, MODEL_NAMES
from backend.services.weather_features import build_live_weather_features
from backend.services.disaster_features import build_disaster_features
from backend.services.terrain_service import terrain_features
from backend.services.waterlogging_history import query as query_waterlogging_history
from backend.services.waterlogging_service import predict_live as predict_waterlogging_live
from backend.services.agriculture_service import predict_live as predict_agriculture_live
from backend.services.drainage_service import predict_live as predict_drainage_live
from backend.data_sources.nominatim import search_place, reverse_geocode
from backend.data_sources.overpass import drainage_features
from backend.data_sources.bhuvan import list_layers, getmap_url
from backend.data_sources.mosdac import info as mosdac_info
from backend.config.settings import OPEN_METEO_FORECAST_URL
from backend.data_sources.http_client import get_json
from backend.data_sources.open_meteo import (
    forecast, historical, air_quality, elevation, marine, flood
)
from backend.data_sources.usgs import get_recent_earthquake_features
from backend.data_sources.gdacs import get_cyclone_and_volcano_alerts
from backend.services.waterlogging_features import get_live_terrain
from backend.data_sources.spatial_rainfall import aggregate_spatial_rainfall
from backend.data_sources.sentinel1_sar import detect_surface_water_changes, get_live_sentinel1_vv
from backend.data_sources.sentinel2_optical import get_live_sentinel2_indices
from backend.services.disaster_service import predict_disaster_live
from backend.services.disaster_history_service import get_disaster_history
from backend.services.spatial_risk_service import (
    get_india_risk_summary,
    get_state_risk_summary,
    get_district_context,
    compute_spatial_grid_risk,
)
from backend.services.pan_india_catalog import (
    get_all_states,
    find_state,
    get_districts_for_state,
    find_district,
    resolve_location,
)
from backend.utils.cache import (
    provider_cache, make_cache_key,
    TTL_TERRAIN, TTL_FLOOD, TTL_USGS, TTL_AIR,
    TTL_WEATHER, TTL_AG_DAILY, TTL_WL_HOURLY, TTL_GDACS,
)

api_bp = Blueprint("api", __name__)

def _coords():
    if request.is_json and request.get_json(silent=True):
        body = request.get_json(silent=True)
        if "lat" in body and "lon" in body:
            return float(body["lat"]), float(body["lon"])
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if lat is None or lon is None:
        raise ValueError("Missing coordinates: lat and lon are required")
    return float(lat), float(lon)

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

@api_bp.route("/predict/drainage/live", methods=["GET", "POST"])
def predict_drainage_live_route():
    payload = request.get_json(silent=True) or {}
    lat_val = request.args.get("lat") or payload.get("lat")
    lon_val = request.args.get("lon") or payload.get("lon")
    if lat_val is None or lon_val is None:
        return fail("Coordinates lat and lon are required for drainage prediction", 400)
    try:
        lat = float(lat_val)
        lon = float(lon_val)
        radius_m = int(request.args.get("radius_m") or payload.get("radius_m") or 2000)
        overrides = payload.get("overrides") or payload.get("features") or {}
        include_osm = request.args.get("include_osm", "1") in ("1", "true", "yes") and payload.get("include_osm", True)
        res = predict_drainage_live(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            overrides=overrides,
            include_osm=include_osm,
        )
        return ok(res)
    except Exception as exc:
        return fail("Live drainage prediction failed", 400, str(exc))

@api_bp.get("/sources/geocode")
def geocode():
    try:
        return ok(search_place(request.args.get("q"), request.args.get("limit", 5)))
    except Exception as exc:
        return fail("Geocoding failed", 400, str(exc))

@api_bp.get("/sources/reverse-geocode")
def reverse_geocode_route():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
        return ok(reverse_geocode(lat, lon))
    except Exception as exc:
        return fail("Reverse geocoding failed", 400, str(exc))

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

@api_bp.route("/waterlogging/drainage", methods=["GET", "POST"])
@api_bp.route("/sources/drainage", methods=["GET", "POST"])
def source_drainage():
    try:
        lat, lon = _coords()
        if request.is_json and request.get_json(silent=True):
            radius = int(request.get_json(silent=True).get("radius_m", 2000))
        else:
            radius = int(request.args.get("radius_m", 2000))
        osm_res = drainage_features(lat, lon, radius)
        try:
            m_res = predict_drainage_live(lat, lon, radius_m=radius, include_osm=False)
            osm_res["model_prediction"] = m_res.get("prediction")
            osm_res["drainage_prediction"] = m_res.get("prediction")
            osm_res["model_features"] = m_res.get("features")
            osm_res["feature_provenance"] = m_res.get("feature_provenance")
            osm_res["data_quality"] = m_res.get("data_quality")
            osm_res["rainfall_provenance"] = m_res.get("rainfall_provenance")
            osm_res["sentinel1_metadata"] = m_res.get("sentinel1_metadata")
            osm_res["sentinel2_metadata"] = m_res.get("sentinel2_metadata")
            osm_res["satellite"] = m_res.get("satellite")
        except Exception as m_exc:
            osm_res["model_prediction_error"] = str(m_exc)
        return ok(osm_res)
    except Exception as exc:
        return fail("Overpass drainage query failed", 400, str(exc))

@api_bp.route("/waterlogging/terrain", methods=["GET", "POST"])
def waterlogging_terrain():
    try:
        lat, lon = _coords()
        return ok(get_live_terrain(lat, lon, return_details=True))
    except Exception as exc:
        return fail("Enhanced terrain retrieval failed", 400, str(exc))

@api_bp.route("/waterlogging/rainfall/spatial", methods=["GET", "POST"])
@api_bp.route("/sources/rainfall/spatial", methods=["GET", "POST"])
def waterlogging_rainfall_spatial():
    try:
        lat, lon = _coords()
        if request.is_json and request.get_json(silent=True):
            body = request.get_json(silent=True)
            delta = float(body.get("delta_deg", 0.5))
        else:
            delta = float(request.args.get("delta_deg", 0.5))
        return ok(aggregate_spatial_rainfall(lat, lon, delta_deg=delta))
    except Exception as exc:
        return fail("Open-Meteo spatial rainfall retrieval failed", 400, str(exc))

@api_bp.route("/waterlogging/sar/sentinel1_vv", methods=["GET", "POST"])
@api_bp.route("/sources/sar/sentinel1_vv", methods=["GET", "POST"])
def waterlogging_sar_sentinel1_vv():
    try:
        lat, lon = _coords()
        days_back = int(request.args.get("days_back", 90))
        force_refresh = request.args.get("force_refresh", "0") in ("1", "true", "yes")
        res = get_live_sentinel1_vv(lat, lon, days_back=days_back, force_refresh=force_refresh)
        return ok(res)
    except Exception as exc:
        return fail("Sentinel-1 SAR VV retrieval failed", 400, str(exc))

@api_bp.route("/waterlogging/sar/sentinel1", methods=["GET", "POST"])
@api_bp.route("/sources/sar/sentinel1", methods=["GET", "POST"])
def waterlogging_sar_sentinel1():
    try:
        lat, lon = _coords()
        if request.is_json and request.get_json(silent=True):
            body = request.get_json(silent=True)
            radius_km = float(body.get("radius_km", 10.0))
        else:
            radius_km = float(request.args.get("radius_km", 10.0))

        if not (5.0 <= radius_km <= 25.0):
            return fail("radius_km must be between 5.0 and 25.0 km.", 400)

        res = detect_surface_water_changes(lat, lon, radius_km=radius_km)
        if res.get("status") == "invalid_parameters":
            return fail(res.get("message", "Invalid parameters"), 400)
        return ok(res)
    except Exception as exc:
        return fail("Sentinel-1 SAR surface water detection failed", 400, str(exc))

@api_bp.route("/waterlogging/optical/sentinel2", methods=["GET", "POST"])
@api_bp.route("/sources/optical/sentinel2", methods=["GET", "POST"])
def waterlogging_optical_sentinel2():
    try:
        lat, lon = _coords()
        max_cloud = float(request.args.get("max_cloud", 60.0))
        days_back = int(request.args.get("days_back", 90))
        res = get_live_sentinel2_indices(lat, lon, max_cloud_cover=max_cloud, days_back=days_back)
        return ok(res)
    except Exception as exc:
        return fail("Sentinel-2 optical retrieval failed", 400, str(exc))

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

@api_bp.route("/predict/drainage/live", methods=["GET", "POST"])
def drainage_live_prediction():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload.get("lat") or request.args.get("lat"))
        lon = float(payload.get("lon") or request.args.get("lon"))
        overrides = payload.get("overrides")
        precomputed = payload.get("precomputed_sources")
        include_osm = payload.get("include_osm", False)
        return ok(
            predict_drainage_live(
                lat=lat,
                lon=lon,
                overrides=overrides,
                precomputed_sources=precomputed,
                include_osm=include_osm,
            )
        )
    except Exception as exc:
        return fail("Live drainage prediction failed", 400, str(exc))

@api_bp.route("/disaster/history", methods=["GET"])
@api_bp.route("/sources/disaster/history", methods=["GET"])
def disaster_history_route():
    try:
        lat, lon = _coords()
        return ok(get_disaster_history(lat, lon))
    except Exception as exc:
        return fail("Disaster history retrieval failed", 400, str(exc))

@api_bp.route("/predict/disaster/live", methods=["GET", "POST"])
@api_bp.route("/sources/disaster/live", methods=["GET", "POST"])
def disaster_live_prediction_route():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload.get("lat") or request.args.get("lat"))
        lon = float(payload.get("lon") or request.args.get("lon"))
        overrides = payload.get("overrides") or payload.get("disaster")
        precomputed = payload.get("precomputed_sources")
        state = payload.get("state") or request.args.get("state")
        district = payload.get("district") or request.args.get("district")
        return ok(predict_disaster_live(
            lat, lon, overrides=overrides, precomputed_sources=precomputed, state=state, district=district
        ))
    except Exception as exc:
        return fail("Live disaster prediction failed", 400, str(exc))


@api_bp.route("/disaster/fire_hotspots", methods=["GET", "POST"])
@api_bp.route("/sources/disaster/fire_hotspots", methods=["GET", "POST"])
def disaster_fire_hotspots_route():
    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload.get("lat") or request.args.get("lat"))
        lon = float(payload.get("lon") or request.args.get("lon"))
        radius_km = float(payload.get("radius_km") or request.args.get("radius_km") or 100.0)
        from backend.data_sources.firms_fire import fetch_firms_hotspots
        return ok(fetch_firms_hotspots(lat, lon, radius_km=radius_km))
    except Exception as exc:
        return fail("FIRMS fire hotspots retrieval failed", 400, str(exc))



@api_bp.route("/disaster/contract", methods=["GET"])
@api_bp.route("/sources/disaster/contract", methods=["GET"])
def disaster_contract_audit_route():
    try:
        from backend.services.disaster_features import get_feature_contract_audit
        return ok(get_feature_contract_audit())
    except Exception as exc:
        return fail("Disaster contract audit failed", 500, str(exc))


# ==============================================================================
# SMART OPERATIONAL ALERT ENGINE & NOTIFICATION ROUTES
# ==============================================================================

@api_bp.route("/alerts/active", methods=["GET"])
def alerts_active_route():
    """
    Returns currently open/active operational alerts across India.
    Filters: hazard, state, district, min_severity, limit.
    """
    try:
        from backend.data_sources.alert_repository import get_active_alerts
        hazard = request.args.get("hazard")
        state = request.args.get("state")
        district = request.args.get("district")
        min_severity = request.args.get("min_severity")
        limit = int(request.args.get("limit") or 100)

        alerts = get_active_alerts(
            hazard=hazard,
            state=state,
            district=district,
            min_severity=min_severity,
            limit=limit,
        )
        return ok({
            "count": len(alerts),
            "alerts": alerts,
        })
    except Exception as exc:
        return fail("Active alerts query failed", 400, str(exc))


@api_bp.route("/alerts/history", methods=["GET"])
def alerts_history_route():
    """
    Returns lifecycle event history for alerts.
    Filters: alert_id, hazard, limit.
    """
    try:
        from backend.data_sources.alert_repository import get_alert_events, get_recent_alert_events
        alert_id = request.args.get("alert_id")
        hazard = request.args.get("hazard")
        limit = int(request.args.get("limit") or 50)

        if alert_id:
            events = get_alert_events(alert_id)
        else:
            events = get_recent_alert_events(limit=limit, hazard=hazard)

        return ok({
            "count": len(events),
            "events": events,
        })
    except Exception as exc:
        return fail("Alert history query failed", 400, str(exc))


@api_bp.route("/alerts/<alert_id>", methods=["GET"])
def alert_detail_route(alert_id: str):
    """
    Returns single alert details with complete chronological lifecycle event timeline.
    """
    try:
        from backend.data_sources.alert_repository import get_alert, get_alert_events
        alert = get_alert(alert_id)
        if not alert:
            return fail(f"Alert not found: {alert_id}", 404)
        
        events = get_alert_events(alert_id)
        alert["events"] = events
        return ok(alert)
    except Exception as exc:
        return fail("Alert lookup failed", 400, str(exc))


@api_bp.route("/alerts/subscriptions", methods=["GET", "POST"])
def alert_subscriptions_route():
    """
    Manages user/location alert subscriptions.
    GET: lists registered subscriptions (filter by user_id).
    POST: registers or updates a subscription.
    """
    from backend.data_sources.alert_repository import save_subscription, get_subscriptions
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        try:
            saved = save_subscription(payload)
            return ok(saved, "Subscription registered successfully")
        except Exception as exc:
            return fail("Subscription registration failed", 400, str(exc))
    else:
        user_id = request.args.get("user_id")
        try:
            subs = get_subscriptions(user_id=user_id, enabled_only=False)
            return ok({
                "count": len(subs),
                "subscriptions": subs,
            })
        except Exception as exc:
            return fail("Subscription retrieval failed", 400, str(exc))


@api_bp.route("/alerts/subscriptions/<sub_id>", methods=["DELETE"])
def alert_subscription_delete_route(sub_id: str):
    """
    Deletes an alert subscription by ID.
    """
    from backend.data_sources.alert_repository import delete_subscription
    try:
        success = delete_subscription(sub_id)
        if success:
            return ok({"deleted": True, "subscription_id": sub_id}, "Subscription deleted")
        return fail("Subscription not found", 404)
    except Exception as exc:
        return fail("Subscription deletion failed", 400, str(exc))


@api_bp.route("/alerts/notifications", methods=["GET"])
def alert_notifications_route():
    """
    Returns recent in-app notifications.
    """
    try:
        from backend.data_sources.alert_repository import get_in_app_notifications
        limit = int(request.args.get("limit") or 50)
        notifs = get_in_app_notifications(limit=limit)
        return ok({
            "count": len(notifs),
            "notifications": notifs,
        })
    except Exception as exc:
        return fail("Notifications query failed", 400, str(exc))



# ==============================================================================
# PAN-INDIA HIERARCHICAL SPATIAL RISK ROUTES
# ==============================================================================

@api_bp.route("/risk/india", methods=["GET"])
@api_bp.route("/sources/risk/india", methods=["GET"])
def risk_india_route():
    try:
        force = request.args.get("refresh", "0") in ("1", "true", "yes")
        return ok(get_india_risk_summary(force_refresh=force))
    except Exception as exc:
        return fail("India-wide risk summary failed", 500, str(exc))


@api_bp.route("/risk/states", methods=["GET"])
@api_bp.route("/sources/risk/states", methods=["GET"])
def risk_states_catalog_route():
    try:
        return ok(get_all_states())
    except Exception as exc:
        return fail("States catalog retrieval failed", 500, str(exc))


@api_bp.route("/risk/state", methods=["GET"])
@api_bp.route("/sources/risk/state", methods=["GET"])
def risk_state_route():
    state_query = request.args.get("state") or request.args.get("name") or request.args.get("code")
    if not state_query:
        return fail("Query parameter 'state' is required", 400)
    try:
        force = request.args.get("refresh", "0") in ("1", "true", "yes")
        return ok(get_state_risk_summary(state_query, force_refresh=force))
    except Exception as exc:
        return fail(f"State risk summary failed for '{state_query}'", 400, str(exc))


@api_bp.route("/risk/district", methods=["GET"])
@api_bp.route("/sources/risk/district", methods=["GET"])
def risk_district_route():
    district_query = request.args.get("district") or request.args.get("name")
    if not district_query:
        return fail("Query parameter 'district' is required", 400)
    state_query = request.args.get("state")
    try:
        return ok(get_district_context(state_query, district_query))
    except Exception as exc:
        return fail(f"District context retrieval failed for '{district_query}'", 400, str(exc))


@api_bp.route("/risk/grid", methods=["GET", "POST"])
@api_bp.route("/sources/risk/grid", methods=["GET", "POST"])
def risk_grid_route():
    payload = request.get_json(silent=True) or {}
    lat_val = payload.get("lat") or request.args.get("lat")
    lon_val = payload.get("lon") or request.args.get("lon")

    if lat_val is None or lon_val is None:
        return fail("Coordinates 'lat' and 'lon' are required for spatial grid risk calculation", 400)

    try:
        lat = float(lat_val)
        lon = float(lon_val)
        radius_m = float(payload.get("radius_m") or request.args.get("radius_m") or 5000.0)
        grid_size_m = float(payload.get("grid_size_m") or request.args.get("grid_size_m") or 500.0)
        state_name = payload.get("state") or request.args.get("state")
        district_name = payload.get("district") or request.args.get("district")
        force = (payload.get("refresh") in (True, "1", "true")) or (request.args.get("refresh") in ("1", "true", "yes"))

        grid_result = compute_spatial_grid_risk(
            center_lat=lat,
            center_lon=lon,
            radius_m=radius_m,
            grid_size_m=grid_size_m,
            state=state_name,
            district=district_name,
            force_refresh=force,
        )
        return ok(grid_result)
    except Exception as exc:
        return fail("Spatial grid risk calculation failed", 500, str(exc))


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


@api_bp.post("/predict-all")
def predict_all():
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    auto_weather = payload.get("auto_weather", True)

    results = {}

    # 1. Weather
    try:
        if payload.get("weather"):
            results["weather"] = {
                "success": True,
                "result": registry.predict("weather", payload["weather"]),
            }
        elif lat is not None and lon is not None:
            w_built = build_live_weather_features(float(lat), float(lon))
            results["weather"] = {
                "success": True,
                "input": w_built,
                "result": registry.predict("weather", w_built["features"]),
            }
        else:
            results["weather"] = {
                "success": False,
                "error": "Coordinates required for weather prediction",
            }
    except Exception as exc:
        results["weather"] = {"success": False, "error": str(exc)}

    # 2. Agriculture
    try:
        ag_data = payload.get("agriculture") or {}
        crop = ag_data.get("crop", "Wheat")
        crop_stage = ag_data.get("crop_stage", "Vegetative")
        if lat is not None and lon is not None:
            results["agriculture"] = {
                "success": True,
                "data": predict_agriculture_live(
                    crop=crop,
                    crop_stage=crop_stage,
                    place_name=ag_data.get("place_name"),
                    lat=float(lat),
                    lon=float(lon),
                ),
            }
        else:
            results["agriculture"] = {
                "success": False,
                "error": "Coordinates required for agriculture prediction",
            }
    except Exception as exc:
        results["agriculture"] = {"success": False, "error": str(exc)}

    # 3. Waterlogging
    try:
        wl_data = payload.get("waterlogging") or {}
        if lat is not None and lon is not None:
            results["waterlogging"] = {
                "success": True,
                "data": predict_waterlogging_live(
                    float(lat),
                    float(lon),
                    state=wl_data.get("state", "LIVE"),
                    district=wl_data.get("district", "LIVE"),
                ),
            }
        else:
            results["waterlogging"] = {
                "success": False,
                "error": "Coordinates required for waterlogging prediction",
            }
    except Exception as exc:
        results["waterlogging"] = {"success": False, "error": str(exc)}

    # 4. Disaster
    try:
        d_data = payload.get("disaster") or {}
        if lat is not None and lon is not None:
            results["disaster"] = predict_disaster_live(
                float(lat), float(lon), overrides=d_data
            )
        else:
            results["disaster"] = {
                "success": False,
                "error": "Coordinates required for disaster prediction",
            }
    except Exception as exc:
        results["disaster"] = {"success": False, "error": str(exc)}

    return ok(results)


# =============================================================================
# RISK SYNTHESIS POLICY CONFIGURATION
# =============================================================================
# Explicit provenance of all thresholds:
#
# 1. MODEL-DERIVED THRESHOLDS (calibrated during model training):
#    - WL_WARNING_THRESHOLD: 0.36 (from trained Waterlogging Random Forest pipeline)
#
# 2. PROVIDER-DEFINED QUERY THRESHOLDS (from USGS/GDACS API standards):
#    - USGS_REGIONAL_RADIUS_KM: 500.0 (USGS regional query feed standard)
#    - USGS_REGIONAL_WINDOW_DAYS: 30.0 (USGS regional query feed standard)
#    - GDACS_CYCLONE_RADIUS_KM: 1000.0 (GDACS tropical cyclone impact radius)
#    - GDACS_VOLCANO_RADIUS_KM: 500.0 (GDACS volcanic ash impact radius)
#
# 3. DASHBOARD / RISK SYNTHESIS POLICY THRESHOLDS (configurable synthesis policy):
#    - SEISMIC_POLICY: Configurable policy rules for classifying live seismic events into
#      Category A (Active / Locally Relevant), Category B (Recent Not Local), and
#      Category C (Historical / Distant Background).
#    - POLICY_SEVERE_DISASTER_CONF: 0.60 (policy threshold for single-model severe disaster escalation)
#    - POLICY_DISASTER_ELEVATED_CONF: 0.40 (policy threshold for non-nominal disaster escalation)
#    - POLICY_DISASTER_ADVISORY_CONF: 0.25 (policy threshold for advisory awareness)
#    - WEATHER_EXTREME_WIND_GUST_KMH: 75.0 (IMD Gale Wind warning standard)
#    - WEATHER_EXTREME_PRECIP_1H_MM: 50.0 (IMD Torrential Rain warning standard)
#    - WEATHER_EXTREME_TEMP_C: 45.0 (IMD Severe Heatwave warning standard)
# =============================================================================

SEISMIC_POLICY = {
    # Provenance: Dashboard Risk Synthesis Policy (Configurable)
    # Category A: Active / Locally Relevant Emergency
    "active_recency_days": 3.0,      # <= 72 hours (primary mainshock / severe aftershock window)
    "major_magnitude": 6.5,          # M >= 6.5
    "major_radius_km": 350.0,        # Felt / damaging ground motion reach for major earthquake
    "strong_magnitude": 5.5,         # M >= 5.5
    "strong_radius_km": 150.0,       # Significant ground motion reach for strong earthquake
    "moderate_magnitude": 4.5,       # M >= 4.5
    "moderate_radius_km": 50.0,      # Near-epicentral shaking reach for moderate earthquake
    
    # Category B: Recent But Not Locally Severe / Monitoring
    "recent_recency_days": 7.0,      # <= 7 days (secondary aftershock monitoring window)
    "recent_strong_radius_km": 250.0,# Broader regional monitoring for M >= 5.5
    "recent_major_radius_km": 500.0, # Regional monitoring for M >= 6.5
    "recent_moderate_radius_km": 100.0, # Regional monitoring for M >= 4.5
}

POLICY_SEVERE_DISASTER_CONF = 0.60
POLICY_DISASTER_ELEVATED_CONF = 0.40
POLICY_DISASTER_ADVISORY_CONF = 0.25

WEATHER_EXTREME_WIND_GUST_KMH = 75.0
WEATHER_EXTREME_PRECIP_1H_MM = 50.0
WEATHER_EXTREME_TEMP_C = 45.0


def classify_seismic_evidence(usgs_event):
    """
    Classifies live seismic evidence into one of three mutually exclusive categories:
    - ACTIVE_LOCALLY_RELEVANT (Category A): Active severe/strong event within impact window & radius.
    - RECENT_NOT_LOCAL (Category B): Recent event within 7 days or moderate radius, but not severe locally.
    - HISTORICAL_DISTANT_BACKGROUND (Category C): Event > 7 days old or beyond regional relevance.
    """
    if not usgs_event or not usgs_event.get("max_quake_mag") or float(usgs_event.get("max_quake_mag", 0)) <= 0:
        return {
            "category": "HISTORICAL_DISTANT_BACKGROUND",
            "reason": "No seismic events recorded within regional query window.",
            "is_active_local": False,
            "is_recent_non_local": False,
        }

    mag = float(usgs_event.get("max_quake_mag", 0.0))
    dist_km = float(usgs_event.get("event_distance_km") or 9999.0)
    age_days = float(usgs_event.get("event_age_days") or 999.0)
    tsunami = int(usgs_event.get("tsunami_flag") or 0)

    # 1. Tsunami Alert: Always Category A
    if tsunami == 1:
        return {
            "category": "ACTIVE_LOCALLY_RELEVANT",
            "reason": f"Active tsunami alert generated by M{mag:.1f} seismic event.",
            "is_active_local": True,
            "is_recent_non_local": False,
        }

    # 2. Category A: Active / Locally Relevant (<= 3 days / 72 hours)
    if age_days <= SEISMIC_POLICY["active_recency_days"]:
        if mag >= SEISMIC_POLICY["major_magnitude"] and dist_km <= SEISMIC_POLICY["major_radius_km"]:
            return {
                "category": "ACTIVE_LOCALLY_RELEVANT",
                "reason": f"Major M{mag:.1f} earthquake {dist_km:.0f} km away occurred {age_days:.1f} days ago (within 72h active window).",
                "is_active_local": True,
                "is_recent_non_local": False,
            }
        if mag >= SEISMIC_POLICY["strong_magnitude"] and dist_km <= SEISMIC_POLICY["strong_radius_km"]:
            return {
                "category": "ACTIVE_LOCALLY_RELEVANT",
                "reason": f"Strong M{mag:.1f} earthquake {dist_km:.0f} km away occurred {age_days:.1f} days ago (within 72h active window).",
                "is_active_local": True,
                "is_recent_non_local": False,
            }
        if mag >= SEISMIC_POLICY["moderate_magnitude"] and dist_km <= SEISMIC_POLICY["moderate_radius_km"]:
            return {
                "category": "ACTIVE_LOCALLY_RELEVANT",
                "reason": f"Moderate M{mag:.1f} earthquake {dist_km:.0f} km away occurred {age_days:.1f} days ago.",
                "is_active_local": True,
                "is_recent_non_local": False,
            }

    # 3. Category B: Recent But Not Locally Severe / Secondary Monitoring (<= 7 days)
    if age_days <= SEISMIC_POLICY["recent_recency_days"]:
        if (
            (mag >= SEISMIC_POLICY["major_magnitude"] and dist_km <= SEISMIC_POLICY["recent_major_radius_km"])
            or (mag >= SEISMIC_POLICY["strong_magnitude"] and dist_km <= SEISMIC_POLICY["recent_strong_radius_km"])
            or (mag >= SEISMIC_POLICY["moderate_magnitude"] and dist_km <= SEISMIC_POLICY["recent_moderate_radius_km"])
            or (mag >= 4.0 and dist_km <= 100.0 and age_days <= SEISMIC_POLICY["active_recency_days"])
        ):
            return {
                "category": "RECENT_NOT_LOCAL",
                "reason": f"Recent M{mag:.1f} earthquake {dist_km:.0f} km away occurred {age_days:.1f} days ago (monitoring phase).",
                "is_active_local": False,
                "is_recent_non_local": True,
            }

    # 4. Category C: Historical / Distant Background
    return {
        "category": "HISTORICAL_DISTANT_BACKGROUND",
        "reason": f"M{mag:.1f} seismic event occurred {dist_km:.0f} km away ({age_days:.1f} days ago) — classified as historical/distant background.",
        "is_active_local": False,
        "is_recent_non_local": False,
    }


def _generate_overall_summary(weather_res, disaster_res, agriculture_res, waterlogging_res, crop, crop_stage):
    """
    Deterministic summary rules based strictly on returned model outputs and live hazard evidence.
    Monotonic hierarchical precedence: CRITICAL > HIGH > MODERATE > LOW.
    """
    summary = {
        "weather_highlights": {},
        "agriculture_highlights": {
            "crop": crop,
            "crop_stage": crop_stage,
        },
        "waterlogging_highlights": {},
        "disaster_highlights": {},
        "overall_risk": "LOW",
        "overall_advisory": "",
        "rule_branch": "LOW",
        "trigger_reason": "",
    }

    # 1. Weather highlights
    w_pred = {}
    w_feat = {}
    if weather_res.get("success"):
        w_data = weather_res.get("data") or {}
        w_pred = (w_data.get("result") or {}).get("prediction") or {}
        w_feat = (w_data.get("input") or {}).get("features") or {}
        summary["weather_highlights"] = {
            "temperature_2m": w_pred.get("temperature_2m", w_feat.get("temperature_2m")),
            "precipitation": w_pred.get("precipitation", w_feat.get("precipitation")),
            "wind_speed_10m": w_pred.get("wind_speed_10m", w_feat.get("wind_speed_10m")),
            "surface_pressure": w_pred.get("surface_pressure", w_feat.get("surface_pressure")),
        }
    else:
        summary["weather_highlights"] = {"status": "Weather prediction unavailable"}

    w_gust = float(w_pred.get("wind_gusts_10m") or w_feat.get("wind_gusts_10m") or 0.0)
    w_precip = float(w_pred.get("precipitation") or w_feat.get("precipitation") or 0.0)
    w_temp = float(w_pred.get("temperature_2m") or w_feat.get("temperature_2m") or 0.0)
    weather_extreme = (
        w_gust >= WEATHER_EXTREME_WIND_GUST_KMH
        or w_precip >= WEATHER_EXTREME_PRECIP_1H_MM
        or w_temp >= WEATHER_EXTREME_TEMP_C
    )

    # 2. Agriculture highlights
    ag_severe = False
    ag_high = False
    ag_moderate = False
    spraying_suitability = "SUITABLE"
    if agriculture_res.get("success"):
        ag_data = agriculture_res.get("data") or {}
        ag_pred = (ag_data.get("result") or {}).get("prediction") or {}
        crop_weather_risk = str(ag_pred.get("crop_weather_risk", "LOW")).upper()
        heat_stress = str(ag_pred.get("heat_stress", "LOW")).upper()
        irrigation = str(ag_pred.get("irrigation", "NO_IRRIGATION")).upper()
        spraying_suitability = str(ag_pred.get("spraying_suitability", "SUITABLE")).upper()
        ag_wl_risk = str(ag_pred.get("waterlogging_risk", "LOW")).upper()

        summary["agriculture_highlights"].update({
            "irrigation_recommendation": irrigation,
            "heat_stress": heat_stress,
            "crop_weather_risk": crop_weather_risk,
            "spraying_suitability": spraying_suitability,
            "agriculture_waterlogging_risk": ag_wl_risk,
        })
        ag_severe = (crop_weather_risk == "SEVERE")
        ag_high = (
            crop_weather_risk == "HIGH"
            or heat_stress == "HIGH"
            or ag_wl_risk == "HIGH"
            or irrigation == "URGENT_IRRIGATION"
        )
        ag_moderate = (
            crop_weather_risk == "MODERATE"
            or heat_stress == "MODERATE"
            or ag_wl_risk == "MODERATE"
            or irrigation == "IRRIGATION_REQUIRED"
            or spraying_suitability == "WAIT"
        )
    else:
        summary["agriculture_highlights"]["status"] = "Agriculture prediction unavailable"

    # 3. Waterlogging highlights
    wl_risk_level = "LOW"
    wl_warning = False
    if waterlogging_res.get("success"):
        wl_data = waterlogging_res.get("data") or {}
        wl_res = wl_data.get("result") or {}
        wl_risk_level = str(wl_res.get("risk_level", "LOW")).upper()
        wl_warning = bool(wl_res.get("warning", False))
        summary["waterlogging_highlights"] = {
            "risk_level": wl_risk_level,
            "warning": wl_warning,
            "positive_probability": wl_res.get("positive_probability"),
            "warning_threshold": wl_res.get("warning_threshold"),
        }
    else:
        summary["waterlogging_highlights"] = {"status": "Waterlogging prediction unavailable"}

    # 4. Disaster highlights & Evidence Corroboration
    disaster_class = None
    disaster_confidence = 0.0
    verified_hazard = {}
    raw_model = {}

    if disaster_res.get("success"):
        d_data = disaster_res.get("data") or {}
        raw_model = d_data.get("raw_model") or {}
        verified_hazard = d_data.get("verified_hazard") or {}
        d_res = d_data.get("result") or {}
        disaster_class = raw_model.get("prediction") or d_res.get("prediction")
        disaster_confidence = float(raw_model.get("confidence") or d_res.get("confidence") or 0.0)

        summary["disaster_highlights"] = {
            "predicted_class": disaster_class,
            "confidence": disaster_confidence,
            "raw_model": raw_model,
            "verified_hazard": verified_hazard,
            "verified_status": verified_hazard.get("status", "NO_ACTIVE_HAZARD"),
            "alert_title": verified_hazard.get("alert_title", "No active local disaster alert"),
            "status": "Inference and corroboration completed",
        }
    else:
        summary["disaster_highlights"] = {
            "status": "Hazard features unavailable",
            "missing_features": disaster_res.get("missing_features", []),
            "verified_status": "DATA_INCOMPLETE",
            "alert_title": "Disaster data incomplete",
        }

    v_active = verified_hazard.get("active", False)
    v_status = verified_hazard.get("status", "NO_ACTIVE_HAZARD")
    v_severity = str(verified_hazard.get("severity", "LOW")).upper()
    v_reason = verified_hazard.get("reason", "")
    v_hazard = verified_hazard.get("hazard")

    active_external_hazard = (v_active and v_hazard in ("tsunami", "cyclone", "volcano"))

    # =========================================================================
    # DETERMINISTIC TIERED PRECEDENCE SYNTHESIS
    # CRITICAL > HIGH > MODERATE > LOW
    # =========================================================================

    # -------------------------------------------------------------------------
    # TIER 1: CRITICAL (Highest Precedence)
    # -------------------------------------------------------------------------
    if active_external_hazard:
        overall_risk = "CRITICAL"
        trigger = v_reason or "Active verified external hazard alert."
        advisory = "Critical risk alert: Immediate hazard mitigation and protective measures recommended."
    elif v_active and v_severity == "CRITICAL":
        overall_risk = "CRITICAL"
        trigger = v_reason
        advisory = "Critical risk alert: Severe verified disaster hazard detected; follow emergency safety protocols."
    elif wl_warning and wl_risk_level == "HIGH":
        overall_risk = "CRITICAL"
        trigger = "Waterlogging warning active with HIGH risk level."
        advisory = "Critical risk alert: Severe waterlogging imminent; activate drainage and field protection."
    elif ag_severe:
        overall_risk = "CRITICAL"
        trigger = "Agriculture model reports SEVERE crop weather risk."
        advisory = "Critical risk alert: Severe crop weather risk detected; emergency crop intervention required."

    # -------------------------------------------------------------------------
    # TIER 2: HIGH (Second Precedence)
    # -------------------------------------------------------------------------
    elif v_active and v_severity == "HIGH":
        overall_risk = "HIGH"
        trigger = v_reason
        advisory = "High risk detected: Verified disaster hazard in effect; monitor emergency advisories."
    elif wl_risk_level == "HIGH":
        overall_risk = "HIGH"
        trigger = "Waterlogging model reports HIGH risk level."
        advisory = "High risk detected: Waterlogging hazard elevated; clear drainage channels."
    elif wl_warning:
        overall_risk = "HIGH"
        trigger = "Waterlogging warning threshold (0.36) exceeded."
        advisory = "High risk detected: Elevated waterlogging probability; monitor soil moisture."
    elif ag_high:
        overall_risk = "HIGH"
        trigger = "Agriculture model reports HIGH risk in crop weather, heat stress, or urgent irrigation."
        advisory = "High risk detected: Significant agronomic stress; execute crop protection recommendations."
    elif weather_extreme:
        overall_risk = "HIGH"
        trigger = "Weather model forecasts extreme meteorological conditions (gale wind, intense rain, or severe heat)."
        advisory = "High risk detected: Extreme atmospheric conditions expected; safeguard field operations."

    # -------------------------------------------------------------------------
    # TIER 3: MODERATE (Third Precedence)
    # -------------------------------------------------------------------------
    elif v_status == "SUPPORTED_HAZARD_SIGNAL":
        overall_risk = "MODERATE"
        trigger = v_reason
        advisory = "Moderate risk conditions detected: Monitor regional hazard conditions."
    elif wl_risk_level in ("MEDIUM", "MODERATE"):
        overall_risk = "MODERATE"
        trigger = "Waterlogging risk level is MEDIUM."
        advisory = "Moderate risk conditions detected: Follow agronomic advisory recommendations."
    elif ag_moderate:
        overall_risk = "MODERATE"
        trigger = "Agriculture model reports MODERATE risk conditions."
        advisory = "Moderate risk conditions detected: Follow agronomic advisory recommendations."
    elif spraying_suitability in ("WAIT", "UNSAFE"):
        overall_risk = "MODERATE"
        trigger = f"Spraying suitability is {spraying_suitability}."
        advisory = "Moderate risk conditions detected: Adjust spraying and routine farm operations."

    # -------------------------------------------------------------------------
    # TIER 4: LOW (Baseline / Lowest Precedence)
    # -------------------------------------------------------------------------
    else:
        overall_risk = "LOW"
        if v_status == "UNCORROBORATED_MODEL_SIGNAL":
            trigger = v_reason
            advisory = "Nominal conditions prevail. Model signal is uncorroborated by current local observations."
        else:
            trigger = "All models report nominal or low-risk conditions."
            advisory = "Normal conditions: Continue routine farm operations and monitoring."

    summary["overall_risk"] = overall_risk
    summary["overall_advisory"] = advisory
    summary["rule_branch"] = overall_risk
    summary["trigger_reason"] = trigger
    return summary



@api_bp.post("/predict/all/live")
def predict_all_live():
    t_start = time.perf_counter()
    payload = request.get_json(silent=True) or {}

    lat_raw = payload.get("lat")
    lon_raw = payload.get("lon")
    if lat_raw is None or lon_raw is None:
        return jsonify({
            "success": False,
            "error": "Missing required coordinates: lat and lon are required"
        }), 400

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Invalid coordinates: lat and lon must be numeric"
        }), 400

    crop = str(payload.get("crop") or "Rice")
    crop_stage = str(payload.get("crop_stage") or "Flowering")
    place_name = payload.get("place_name")
    state = payload.get("state", "LIVE")
    district = payload.get("district", "LIVE")
    disaster_overrides = payload.get("overrides") or payload.get("disaster") or {}

    # 8 independent external providers definition
    provider_specs = {
        "forecast": {
            "name": "open_meteo_forecast",
            "params": {"past_days": 2, "forecast_days": 2, "timezone": "auto"},
            "ttl": TTL_WEATHER,
            "fetcher": lambda: forecast(lat, lon, past_days=2, forecast_days=2),
        },
        "air_quality": {
            "name": "open_meteo_air_quality",
            "params": {"forecast_days": 2, "timezone": "auto"},
            "ttl": TTL_AIR,
            "fetcher": lambda: air_quality(lat, lon),
        },
        "terrain": {
            "name": "open_meteo_terrain",
            "params": {
                "context_points": 8,
                "context_radius_m": 1000.0,
                "model_offset_deg": 0.01,
                "provider": "open_meteo_elevation",
                "version": "phase_c_v1",
            },
            "ttl": TTL_TERRAIN,
            "fetcher": lambda: get_live_terrain(lat, lon, return_details=True),
        },
        "flood": {
            "name": "open_meteo_flood",
            "params": {"daily": "river_discharge", "past_days": 1, "forecast_days": 1},
            "ttl": TTL_FLOOD,
            "fetcher": lambda: flood(lat, lon),
        },
        "usgs": {
            "name": "usgs_earthquake",
            "params": {"days": 30, "maxradiuskm": 500},
            "ttl": TTL_USGS,
            "fetcher": lambda: get_recent_earthquake_features(lat, lon),
        },
        "agriculture_forecast": {
            "name": "open_meteo_agriculture",
            "params": {
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum,et0_fao_evapotranspiration",
                "past_days": 14,
                "forecast_days": 1,
                "timezone": "Asia/Kolkata",
            },
            "ttl": TTL_AG_DAILY,
            "fetcher": lambda: get_json(OPEN_METEO_FORECAST_URL, {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum,et0_fao_evapotranspiration",
                "past_days": 14,
                "forecast_days": 1,
                "timezone": "Asia/Kolkata",
            }),
        },
        "waterlogging_forecast": {
            "name": "open_meteo_waterlogging",
            "params": {
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,surface_pressure",
                "past_days": 7,
                "forecast_days": 4,
                "timezone": "Asia/Kolkata",
            },
            "ttl": TTL_WL_HOURLY,
            "fetcher": lambda: get_json(OPEN_METEO_FORECAST_URL, {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,surface_pressure",
                "past_days": 7,
                "forecast_days": 4,
                "timezone": "Asia/Kolkata",
            }),
        },
        "gdacs": {
            "name": "gdacs",
            "params": {"cyclone_radius_km": 1000, "eventtypes": "TC,VO", "volcano_radius_km": 500},
            "ttl": TTL_GDACS,
            "fetcher": lambda: get_cyclone_and_volcano_alerts(lat, lon),
        },
    }

    # Parallel live data fetching across independent external APIs with exact-coordinate cache keys
    t_provider_start = time.perf_counter()
    provider_times = {}
    provider_cache_status = {}
    precomputed = {}
    tasks_to_run = {}

    for key, spec in provider_specs.items():
        ckey = make_cache_key(spec["name"], lat, lon, spec["params"])
        spec["cache_key"] = ckey
        cached_val = provider_cache.get(ckey)
        if cached_val is not None:
            precomputed[key] = cached_val
            provider_times[f"{spec['name']}_ms"] = 0.0
            provider_cache_status[spec["name"]] = {
                "cached": True,
                "fetch_ms": 0.0,
                "http_requests": 0,
            }
        else:
            tasks_to_run[key] = spec

    if tasks_to_run:
        def _execute_task(tkey, tspec):
            t0 = time.perf_counter()
            try:
                data = tspec["fetcher"]()
            except Exception:
                data = None
            ms = round((time.perf_counter() - t0) * 1000, 2)
            if data is not None:
                provider_cache.set(tspec["cache_key"], data, ttl=tspec["ttl"])
            return tkey, tspec["name"], data, ms

        with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
            futures = [
                executor.submit(_execute_task, k, s) for k, s in tasks_to_run.items()
            ]
            for fut in futures:
                tkey, name, data, ms = fut.result()
                precomputed[tkey] = data
                provider_times[f"{name}_ms"] = ms
                provider_cache_status[name] = {
                    "cached": False,
                    "fetch_ms": ms,
                    "http_requests": 1,
                }

    # Extract center-point elevation for disaster from shared 5-point terrain
    if precomputed.get("terrain") is not None:
        t_val = precomputed["terrain"]
        if isinstance(t_val, (tuple, list)) and len(t_val) >= 1:
            precomputed["elevation"] = float(t_val[0])
        elif isinstance(t_val, dict):
            precomputed["elevation"] = t_val.get("elevation_m")

    provider_times["total_provider_fetch_ms"] = round((time.perf_counter() - t_provider_start) * 1000, 2)

    # 1. Weather module (reuses pre-fetched forecast)
    t0 = time.perf_counter()
    try:
        t_fb = time.perf_counter()
        w_built = build_live_weather_features(lat, lon, raw_forecast=precomputed.get("forecast"))
        w_fb_ms = round((time.perf_counter() - t_fb) * 1000, 2)
        t_inf = time.perf_counter()
        w_result = registry.predict("weather", w_built["features"])
        w_inf_ms = round((time.perf_counter() - t_inf) * 1000, 2)
        w_total_ms = round((time.perf_counter() - t0) * 1000, 2)
        weather_timings = {
            "network_ms": 0.0,
            "feature_build_ms": w_fb_ms,
            "inference_ms": w_inf_ms,
            "total_ms": w_total_ms,
        }
        weather_res = {
            "success": True,
            "data": {
                "input": w_built,
                "result": w_result,
            }
        }
    except Exception as exc:
        w_total_ms = round((time.perf_counter() - t0) * 1000, 2)
        weather_timings = {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": w_total_ms,
        }
        weather_res = {
            "success": False,
            "error": str(exc),
        }

    # 2. Waterlogging module
    t0 = time.perf_counter()
    try:
        wl_data = predict_waterlogging_live(
            lat=lat,
            lon=lon,
            state=state,
            district=district,
            precomputed_sources=precomputed,
        )
        wl_timings = wl_data.get("timings") or {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        waterlogging_res = {
            "success": True,
            "data": wl_data,
        }
        # Provide waterlogging & drainage predictions to precomputed for disaster evidence corroboration
        if isinstance(wl_data, dict):
            precomputed["waterlogging"] = wl_data.get("result") or wl_data
            if "drainage_model_result" in wl_data:
                precomputed["drainage"] = wl_data["drainage_model_result"]
    except Exception as exc:
        wl_timings = {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        waterlogging_res = {
            "success": False,
            "error": str(exc),
        }

    # 3. Disaster module (uses pre-fetched live sources + 3 engineered formulas + caller overrides + evidence corroboration with hydrology context)
    t0 = time.perf_counter()
    try:
        d_out = predict_disaster_live(
            lat, lon, overrides=disaster_overrides, precomputed_sources=precomputed, state=state, district=district
        )
        d_total_ms = round((time.perf_counter() - t0) * 1000, 2)
        disaster_timings = d_out.get("timings") or {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": d_total_ms,
        }
        disaster_res = {
            "success": d_out.get("success", True),
            "data": d_out,
        }
    except Exception as exc:
        d_total_ms = round((time.perf_counter() - t0) * 1000, 2)
        disaster_timings = {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": d_total_ms,
        }
        disaster_res = {
            "success": False,
            "error": str(exc),
        }

    # 4. Agriculture module
    t0 = time.perf_counter()
    try:
        ag_data = predict_agriculture_live(
            crop=crop,
            crop_stage=crop_stage,
            lat=lat,
            lon=lon,
            place_name=place_name,
            precomputed_sources=precomputed,
        )
        ag_timings = ag_data.get("timings") or {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        agriculture_res = {
            "success": True,
            "data": ag_data,
        }
    except Exception as exc:
        ag_timings = {
            "network_ms": 0.0,
            "feature_build_ms": 0.0,
            "inference_ms": 0.0,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        agriculture_res = {
            "success": False,
            "error": str(exc),
        }

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Deterministic summary
    summary = _generate_overall_summary(
        weather_res, disaster_res, agriculture_res, waterlogging_res, crop, crop_stage
    )

    response_payload = {
        "success": True,
        "location": {
            "lat": lat,
            "lon": lon,
            "crop": crop,
            "crop_stage": crop_stage,
        },
        "weather": weather_res,
        "disaster": disaster_res,
        "agriculture": agriculture_res,
        "waterlogging": waterlogging_res,
        "summary": summary,
        "execution_times": {
            "weather": weather_timings,
            "disaster": disaster_timings,
            "agriculture": ag_timings,
            "waterlogging": wl_timings,
            "weather_ms": weather_timings["total_ms"],
            "disaster_ms": disaster_timings["total_ms"],
            "agriculture_ms": ag_timings["total_ms"],
            "waterlogging_ms": wl_timings["total_ms"],
            "total_ms": total_ms,
        },
        "provider_execution_times": provider_times,
        "provider_cache_status": provider_cache_status,
        "total_execution_time_ms": total_ms,
    }

    return jsonify(response_payload), 200


