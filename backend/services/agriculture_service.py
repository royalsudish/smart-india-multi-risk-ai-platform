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


def predict_live(
    crop,
    crop_stage,
    place_name=None,
    lat=None,
    lon=None,
):
    loc = resolve_location(
        place_name=place_name,
        lat=lat,
        lon=lon,
    )

    payload = assemble_agriculture_payload(
        loc["lat"],
        loc["lon"],
        crop,
        crop_stage,
    )

    result = registry.predict(
        "agriculture",
        payload,
    )

    return {
        "location": loc,
        "crop": crop,
        "crop_stage": crop_stage,
        "input_features": payload,
        "result": result,
    }
