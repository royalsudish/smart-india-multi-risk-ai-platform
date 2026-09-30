import numpy as np
import pandas as pd

from backend.config.settings import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_ELEVATION_URL,
)
from backend.data_sources.http_client import get_json

SELECTED_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "latitude",
    "longitude",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "elevation",
    "slope",
    "month",
    "hour",
    "is_monsoon",
    "is_post_monsoon",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "antecedent_rain_24h",
    "antecedent_rain_3d",
    "antecedent_rain_7d",
    "antecedent_rain_index",
    "short_term_rain_ratio",
    "flatness_score",
    "rain_flat_interaction",
    "rain_low_elevation_interaction",
]


def engineer_ml_features(data):
    """
    Same feature formulas used by the uploaded waterlogging notebook.
    """
    out = data.copy()

    out["datetime"] = pd.to_datetime(
        out["datetime"],
        errors="coerce"
    )

    out["month"] = out["datetime"].dt.month
    out["hour"] = out["datetime"].dt.hour
    out["day_of_year"] = out["datetime"].dt.dayofyear

    out["is_monsoon"] = (
        out["month"].isin([6, 7, 8, 9]).astype(int)
    )

    out["is_post_monsoon"] = (
        out["month"].isin([10, 11, 12]).astype(int)
    )

    out["month_sin"] = np.sin(
        2 * np.pi * out["month"] / 12
    )
    out["month_cos"] = np.cos(
        2 * np.pi * out["month"] / 12
    )

    out["hour_sin"] = np.sin(
        2 * np.pi * out["hour"] / 24
    )
    out["hour_cos"] = np.cos(
        2 * np.pi * out["hour"] / 24
    )

    out["antecedent_rain_24h"] = (
        out["rainfall_24h"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_3d"] = (
        out["rainfall_3d"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_7d"] = (
        out["rainfall_7d"] - out["rainfall_1h"]
    ).clip(lower=0)

    out["antecedent_rain_index"] = (
        0.50 * out["antecedent_rain_24h"]
        + 0.30 * (out["antecedent_rain_3d"] / 3)
        + 0.20 * (out["antecedent_rain_7d"] / 7)
    )

    out["short_term_rain_ratio"] = np.where(
        out["rainfall_24h"] > 0,
        out["rainfall_6h"] / out["rainfall_24h"],
        0.0
    )

    out["short_term_rain_ratio"] = (
        out["short_term_rain_ratio"]
        .clip(lower=0, upper=1)
    )

    out["flatness_score"] = (
        1 / (1 + out["slope"])
    )

    out["rain_flat_interaction"] = (
        out["rainfall_24h"] * out["flatness_score"]
    )

    safe_elevation = out["elevation"].clip(lower=0)

    out["rain_low_elevation_interaction"] = (
        out["rainfall_24h"]
        / (1 + safe_elevation / 100)
    )

    out["rain_intensity_3h"] = (
        out["rainfall_3h"] / 3
    )

    out["rain_intensity_6h"] = (
        out["rainfall_6h"] / 6
    )

    if (
        "distance_to_drain" in out.columns
        and out["distance_to_drain"].notna().any()
    ):
        out["drain_proximity"] = (
            1 / (1 + out["distance_to_drain"] / 1000)
        )
        out["rain_drain_interaction"] = (
            out["rainfall_24h"] * out["drain_proximity"]
        )

    if (
        "built_up_proxy" in out.columns
        and out["built_up_proxy"].notna().any()
    ):
        out["urban_rain_interaction"] = (
            out["rainfall_24h"] * out["built_up_proxy"]
        )

    return out


# ==============================================================================
# ENHANCED TERRAIN INTELLIGENCE POLICIES (PHASE C)
# ==============================================================================
# NOTE: These policies are operational dashboard contextual rules.
# They are NOT ML features, NOT fed into waterlogging.pkl, and NOT changing
# the trained Random Forest warning threshold (0.36).
# ==============================================================================

TERRAIN_RELATIVE_ELEVATION_POLICY = [
    {
        "max_delta_m": -1.5,
        "badge_text": "LOCALLY LOWER THAN SAMPLED SURROUNDINGS (SIGNIFICANT)",
        "badge_class": "badge-high",
        "description": "Center point is {val:.1f}m lower than the mean of {n} surrounding samples at 1000m radius. Potential localized surface runoff convergence zone.",
    },
    {
        "max_delta_m": -0.5,
        "badge_text": "LOCALLY LOWER THAN SAMPLED SURROUNDINGS (MILD)",
        "badge_class": "badge-moderate",
        "description": "Center point is {val:.1f}m lower than the mean of {n} surrounding samples at 1000m radius. Mildly low-lying relative to local neighborhood.",
    },
    {
        "max_delta_m": 0.5,
        "badge_text": "LEVEL WITH SAMPLED SURROUNDINGS",
        "badge_class": "badge-neutral",
        "description": "Center point elevation is within ±0.5m of the mean of {n} surrounding samples at 1000m radius.",
    },
    {
        "max_delta_m": float("inf"),
        "badge_text": "LOCALLY ELEVATED ABOVE SAMPLED SURROUNDINGS",
        "badge_class": "badge-low",
        "description": "Center point is {val:.1f}m higher than the mean of {n} surrounding samples at 1000m radius. Runoff typically sheds to surrounding areas.",
    },
]

TERRAIN_GRADIENT_POLICY = [
    {
        "max_gradient_m_per_km": 1.0,
        "category": "EXTREMELY FLAT",
        "description": "Negligible topographic gradient (< 1 m/km). Very slow surface runoff drainage.",
    },
    {
        "max_gradient_m_per_km": 5.0,
        "category": "VERY FLAT",
        "description": "Low topographic gradient (1–5 m/km). Slow surface drainage potential.",
    },
    {
        "max_gradient_m_per_km": 20.0,
        "category": "GENTLE GRADIENT",
        "description": "Gentle slope (5–20 m/km). Moderate surface runoff conveyance.",
    },
    {
        "max_gradient_m_per_km": 50.0,
        "category": "MODERATE SLOPE",
        "description": "Moderate slope (20–50 m/km). Good surface drainage potential.",
    },
    {
        "max_gradient_m_per_km": float("inf"),
        "category": "STEEP / HILLY",
        "description": "Steep topographic gradient (≥ 50 m/km). Rapid gravity-driven surface runoff.",
    },
]

CARDINALS_16 = [
    ("N", 0.0, "↑"), ("NNE", 22.5, "↗"), ("NE", 45.0, "↗"), ("ENE", 67.5, "↗"),
    ("E", 90.0, "→"), ("ESE", 112.5, "↘"), ("SE", 135.0, "↘"), ("SSE", 157.5, "↘"),
    ("S", 180.0, "↓"), ("SSW", 202.5, "↙"), ("SW", 225.0, "↙"), ("WSW", 247.5, "↙"),
    ("W", 270.0, "←"), ("WNW", 292.5, "↖"), ("NW", 315.0, "↖"), ("NNW", 337.5, "↖")
]


def build_terrain_manifest(lat: float, lon: float, context_radius_m: float = 1000.0, model_offset_deg: float = 0.01):
    """
    Constructs the 13-point terrain request manifest:
    - 5 points for the exact, preserved Waterlogging model stencil (lat ± 0.01, lon ± 0.01)
    - 8 points for the equal-distance 1000m circular contextual ring
    """
    lat_rad = np.radians(lat)
    cos_lat = max(float(np.cos(lat_rad)), 0.01)

    context_defs = [
        ("context_n", 0.0, "N"),
        ("context_ne", 45.0, "NE"),
        ("context_e", 90.0, "E"),
        ("context_se", 135.0, "SE"),
        ("context_s", 180.0, "S"),
        ("context_sw", 225.0, "SW"),
        ("context_w", 270.0, "W"),
        ("context_nw", 315.0, "NW"),
    ]

    manifest = [
        # Preserved 5-point model stencil (Center, North, South, East, West)
        {"id": "model_center", "lat": lat, "lon": lon, "role": "model_and_context", "bearing": None, "dir": "Center", "dist_m": 0.0},
        {"id": "model_north", "lat": lat + model_offset_deg, "lon": lon, "role": "model", "bearing": 0.0, "dir": "N_model", "dist_m": model_offset_deg * 111320.0},
        {"id": "model_south", "lat": lat - model_offset_deg, "lon": lon, "role": "model", "bearing": 180.0, "dir": "S_model", "dist_m": model_offset_deg * 111320.0},
        {"id": "model_east", "lat": lat, "lon": lon + model_offset_deg, "role": "model", "bearing": 90.0, "dir": "E_model", "dist_m": model_offset_deg * 111320.0 * cos_lat},
        {"id": "model_west", "lat": lat, "lon": lon - model_offset_deg, "role": "model", "bearing": 270.0, "dir": "W_model", "dist_m": model_offset_deg * 111320.0 * cos_lat},
    ]

    # Earth mean radius in meters for forward-geodesic calculation
    R_EARTH = 6371000.0
    delta = context_radius_m / R_EARTH
    phi1 = np.radians(lat)
    lambda1 = np.radians(lon)

    for cid, bearing_deg, dname in context_defs:
        theta = np.radians(bearing_deg)
        phi2 = np.arcsin(
            np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta)
        )
        lambda2 = lambda1 + np.arctan2(
            np.sin(theta) * np.sin(delta) * np.cos(phi1),
            np.cos(delta) - np.sin(phi1) * np.sin(phi2)
        )
        c_lat = float(np.degrees(phi2))
        c_lon = float(np.degrees(lambda2))

        manifest.append({
            "id": cid,
            "lat": c_lat,
            "lon": c_lon,
            "role": "context",
            "bearing": bearing_deg,
            "dir": dname,
            "dist_m": context_radius_m,
        })

    return manifest


def get_live_terrain(lat, lon, precomputed_terrain=None, return_details=False):
    """
    Enhanced Terrain Intelligence for Waterlogging (Phase C):
    - Preserves exact 5-point model stencil slope & center elevation bit-for-bit.
    - Computes 8-point equal-distance contextual ring at 1000m radius.
    - If return_details is False: returns (elevation, slope) tuple for 100% backward compatibility.
    - If return_details is True: returns dict with elevation_m, slope_deg, and enhanced_terrain.
    """
    if precomputed_terrain is not None:
        if isinstance(precomputed_terrain, dict):
            elev = precomputed_terrain.get("elevation_m", precomputed_terrain.get("elevation"))
            slope = precomputed_terrain.get("slope_deg", precomputed_terrain.get("slope"))
            if return_details:
                if "enhanced_terrain" in precomputed_terrain:
                    return precomputed_terrain
                return {
                    "elevation_m": float(elev) if elev is not None else None,
                    "slope_deg": float(slope) if slope is not None else None,
                    "enhanced_terrain": precomputed_terrain.get("enhanced_terrain"),
                }
            if elev is not None and slope is not None:
                return float(elev), float(slope)
            return (float(elev) if elev is not None else None, float(slope) if slope is not None else None)

        if isinstance(precomputed_terrain, (tuple, list)) and len(precomputed_terrain) >= 2:
            elev = precomputed_terrain[0]
            slope = precomputed_terrain[1]
            if return_details:
                return {
                    "elevation_m": float(elev) if elev is not None else None,
                    "slope_deg": float(slope) if slope is not None else None,
                    "enhanced_terrain": None,
                }
            return float(elev) if elev is not None else None, float(slope) if slope is not None else None

    manifest = build_terrain_manifest(lat, lon)

    try:
        data = get_json(
            OPEN_METEO_ELEVATION_URL,
            {
                "latitude": ",".join(map(str, [p["lat"] for p in manifest])),
                "longitude": ",".join(map(str, [p["lon"] for p in manifest])),
            }
        )
        elevations = data.get("elevation", [])
    except Exception:
        elevations = []

    if not elevations or len(elevations) < 5:
        # Provider failure: null semantics, never fake zero values
        if return_details:
            return {
                "elevation_m": None,
                "slope_deg": None,
                "enhanced_terrain": {
                    "status": "unavailable",
                    "center_elevation_m": None,
                    "model_slope_deg": None,
                    "surrounding_mean_elevation_m": None,
                    "relative_elevation_m": None,
                    "relative_elevation_context": None,
                    "local_min_elevation_m": None,
                    "local_max_elevation_m": None,
                    "local_relief_m": None,
                    "gradient_m_per_km": None,
                    "gradient_context": None,
                    "downslope_direction": None,
                    "profiles": {"north_south": None, "east_west": None},
                    "sample_coverage": {"valid_samples": 0, "expected_samples": 8, "coverage_pct": 0.0, "represented_quadrants": 0},
                    "samples_geojson": {"type": "FeatureCollection", "features": []},
                    "source": "Open-Meteo Elevation API",
                    "note": "Open-Meteo elevation service unavailable.",
                }
            }
        return None, None

    elev_map = {
        item["id"]: (float(elev) if elev is not None else None)
        for item, elev in zip(manifest, elevations)
    }

    # =========================================================================
    # 1. MODEL STENCIL (EXACT 5 POINTS PRESERVED BIT-FOR-BIT)
    # =========================================================================
    z_center = elev_map.get("model_center")
    z_north = elev_map.get("model_north")
    z_south = elev_map.get("model_south")
    z_east = elev_map.get("model_east")
    z_west = elev_map.get("model_west")

    offset = 0.01
    north_south_distance = 2 * offset * 111320
    east_west_distance = (
        2 * offset * 111320 * np.cos(np.radians(lat))
    )

    if all(v is not None for v in [z_center, z_north, z_south, z_east, z_west]):
        dz_dy = (z_north - z_south) / north_south_distance
        dz_dx = (z_east - z_west) / east_west_distance
        model_slope = float(
            np.degrees(
                np.arctan(
                    np.sqrt(dz_dx ** 2 + dz_dy ** 2)
                )
            )
        )
        elevation = float(z_center)
    else:
        dz_dy = None
        dz_dx = None
        model_slope = None
        elevation = float(z_center) if z_center is not None else None

    # If only model tuple was requested, return early for maximum speed
    if not return_details:
        return elevation, model_slope

    # =========================================================================
    # 2. ENHANCED CONTEXTUAL TERRAIN (8-POINT EQUAL-DISTANCE RING AT 1000m)
    # =========================================================================
    context_ids = [
        "context_n", "context_ne", "context_e", "context_se",
        "context_s", "context_sw", "context_w", "context_nw"
    ]
    valid_context = {
        cid: elev_map[cid] for cid in context_ids if elev_map.get(cid) is not None
    }
    valid_count = len(valid_context)
    coverage_pct = round((valid_count / 8.0) * 100.0, 1)

    q_map = {
        "Q1": ["context_n", "context_ne", "context_e"],
        "Q2": ["context_e", "context_se", "context_s"],
        "Q3": ["context_s", "context_sw", "context_w"],
        "Q4": ["context_w", "context_nw", "context_n"],
    }
    represented_quadrants = sum(
        1 for q_points in q_map.values() if any(p in valid_context for p in q_points)
    )

    # Per-metric sample requirements: Relative Elevation & Relief require >= 6 samples and >= 3 quadrants
    has_sufficient_context = (
        z_center is not None and valid_count >= 6 and represented_quadrants >= 3
    )

    if has_sufficient_context:
        surrounding_mean = round(float(np.mean(list(valid_context.values()))), 2)
        relative_elevation = round(float(z_center - surrounding_mean), 2)

        rel_ctx = None
        for pol in TERRAIN_RELATIVE_ELEVATION_POLICY:
            if relative_elevation <= pol["max_delta_m"]:
                rel_ctx = {
                    "badge_text": pol["badge_text"],
                    "badge_class": pol["badge_class"],
                    "description": pol["description"].format(
                        val=abs(relative_elevation), n=valid_count
                    ),
                }
                break

        all_samples = [z_center] + list(valid_context.values())
        local_min = round(float(min(all_samples)), 2)
        local_max = round(float(max(all_samples)), 2)
        local_relief = round(float(local_max - local_min), 2)
        terrain_status = "ok"
    else:
        surrounding_mean = None
        relative_elevation = None
        rel_ctx = None
        local_min = None
        local_max = None
        local_relief = None
        terrain_status = "insufficient_context_samples" if z_center is not None else "unavailable"

    # =========================================================================
    # 3. GRADIENT & DOWNSLOPE: EXPLICIT MODEL VS CONTEXTUAL DERIVATIONS
    # =========================================================================
    # A. Model Stencil Gradient & Downslope (from 5-point stencil: lat ± 0.01, lon ± 0.01)
    if model_slope is not None and dz_dx is not None and dz_dy is not None:
        model_gradient_m_per_km = round(float(np.tan(np.radians(model_slope)) * 1000.0), 2)
        grad_ctx = None
        for gpol in TERRAIN_GRADIENT_POLICY:
            if model_gradient_m_per_km <= gpol["max_gradient_m_per_km"]:
                grad_ctx = {
                    "category": gpol["category"],
                    "description": gpol["description"],
                }
                break

        if model_gradient_m_per_km < 0.5:
            model_downslope = {
                "azimuth_deg": None,
                "cardinal": "FLAT",
                "arrow": "—",
                "source": "5-point model stencil (lat ± 0.01, lon ± 0.01)",
                "description": "Indeterminate / Flat: model topographic gradient is under 0.5 m/km.",
            }
        else:
            azimuth = float(np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0)
            azimuth = round(azimuth, 1)
            card_idx = int((azimuth + 11.25) // 22.5) % 16
            cardinal, _, arrow = CARDINALS_16[card_idx]
            model_downslope = {
                "azimuth_deg": azimuth,
                "cardinal": cardinal,
                "arrow": arrow,
                "source": "5-point model stencil (lat ± 0.01, lon ± 0.01)",
                "description": f"Approximate model terrain downslope direction ({azimuth}° {cardinal}).",
            }
    else:
        model_gradient_m_per_km = None
        grad_ctx = None
        model_downslope = None

    # B. Contextual Ring Gradient & Downslope (from 1 km equal-distance ring samples)
    z_s = elev_map.get("context_s")
    z_n = elev_map.get("context_n")
    z_w = elev_map.get("context_w")
    z_e = elev_map.get("context_e")

    if all(v is not None for v in [z_s, z_n, z_w, z_e]):
        dz_dy_ctx = (z_n - z_s) / 2000.0
        dz_dx_ctx = (z_e - z_w) / 2000.0
        ctx_slope_rad = np.arctan(np.sqrt(dz_dx_ctx ** 2 + dz_dy_ctx ** 2))
        contextual_gradient_m_per_km = round(float(np.tan(ctx_slope_rad) * 1000.0), 2)

        if contextual_gradient_m_per_km < 0.5:
            contextual_downslope = {
                "azimuth_deg": None,
                "cardinal": "FLAT",
                "arrow": "—",
                "source": "1 km contextual ring (N/S/E/W at 1000m)",
                "description": "Indeterminate / Flat: contextual topographic gradient is under 0.5 m/km.",
            }
        else:
            ctx_azimuth = float(np.degrees(np.arctan2(-dz_dx_ctx, -dz_dy_ctx)) % 360.0)
            ctx_azimuth = round(ctx_azimuth, 1)
            ctx_card_idx = int((ctx_azimuth + 11.25) // 22.5) % 16
            ctx_cardinal, _, ctx_arrow = CARDINALS_16[ctx_card_idx]
            contextual_downslope = {
                "azimuth_deg": ctx_azimuth,
                "cardinal": ctx_cardinal,
                "arrow": ctx_arrow,
                "source": "1 km contextual ring (N/S/E/W at 1000m)",
                "description": f"Approximate contextual terrain downslope direction ({ctx_azimuth}° {ctx_cardinal}).",
            }
    else:
        contextual_gradient_m_per_km = None
        contextual_downslope = None

    # Terrain Cross-Sections (N-S and E-W Profiles)
    if all(v is not None for v in [z_s, z_center, z_n]):
        ns_profile = [
            {"direction": "S", "distance_m": -1000.0, "elevation_m": round(z_s, 1)},
            {"direction": "Center", "distance_m": 0.0, "elevation_m": round(z_center, 1)},
            {"direction": "N", "distance_m": 1000.0, "elevation_m": round(z_n, 1)},
        ]
    else:
        ns_profile = None

    if all(v is not None for v in [z_w, z_center, z_e]):
        ew_profile = [
            {"direction": "W", "distance_m": -1000.0, "elevation_m": round(z_w, 1)},
            {"direction": "Center", "distance_m": 0.0, "elevation_m": round(z_center, 1)},
            {"direction": "E", "distance_m": 1000.0, "elevation_m": round(z_e, 1)},
        ]
    else:
        ew_profile = None

    # GeoJSON FeatureCollection (Strict RFC 7946 [longitude, latitude])
    geojson_features = []
    for item in manifest:
        elev_val = elev_map.get(item["id"])
        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(item["lon"]), float(item["lat"])],
            },
            "properties": {
                "sample_id": item["id"],
                "sample_role": item["role"],
                "bearing_deg": item.get("bearing"),
                "direction": item.get("dir"),
                "distance_m": float(item.get("dist_m", 0.0)),
                "elevation_m": float(elev_val) if elev_val is not None else None,
            },
        })

    geojson_collection = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }

    enhanced_terrain = {
        "status": terrain_status,
        "center_elevation_m": elevation,
        "model_slope_deg": model_slope,
        "surrounding_mean_elevation_m": surrounding_mean,
        "relative_elevation_m": relative_elevation,
        "relative_elevation_context": rel_ctx,
        "local_min_elevation_m": local_min,
        "local_max_elevation_m": local_max,
        "local_relief_m": local_relief,
        "model_gradient_m_per_km": model_gradient_m_per_km,
        "gradient_m_per_km": model_gradient_m_per_km,
        "gradient_context": grad_ctx,
        "model_downslope_direction": model_downslope,
        "downslope_direction": model_downslope,
        "contextual_gradient_m_per_km": contextual_gradient_m_per_km,
        "contextual_downslope_direction": contextual_downslope,
        "profiles": {
            "north_south": ns_profile,
            "east_west": ew_profile,
        },
        "sample_coverage": {
            "valid_samples": valid_count,
            "expected_samples": 8,
            "coverage_pct": coverage_pct,
            "represented_quadrants": represented_quadrants,
        },
        "samples_geojson": geojson_collection,
        "source": "Open-Meteo Elevation API",
        "note": "Derived from 8 equal-distance contextual samples at 1000m radius. Macro-topographic DEM only.",
    }

    return {
        "elevation_m": elevation,
        "slope_deg": model_slope,
        "enhanced_terrain": enhanced_terrain,
    }


# ==============================================================================
# POLICY-DEFINED ADVISORY THRESHOLDS (DETERMINISTIC OUTLOOK ONLY)
# ==============================================================================
# NOTE: These thresholds are operational dashboard advisory rules designed for
# future rainfall risk context. They are NOT ML thresholds, NOT calibrated
# probabilities, and NOT trained Random Forest thresholds.
# The ONLY trained Waterlogging warning threshold is 0.36.
# ==============================================================================
FUTURE_RAINFALL_ADVISORY_POLICY = [
    {
        "max_24h_mm": 5.0,
        "severity": "LOW",
        "advisory_template": "Negligible or light rainfall forecast (< 5 mm in 24h). Future rainfall-induced waterlogging risk remains low.",
        "source_type": "POLICY-DEFINED ADVISORY THRESHOLDS",
    },
    {
        "max_24h_mm": 25.0,
        "severity": "MODERATE",
        "advisory_template": "Light-to-moderate showers forecast (~{val} mm in 24h). Standard drainage should manage runoff; continue routine monitoring.",
        "source_type": "POLICY-DEFINED ADVISORY THRESHOLDS",
    },
    {
        "max_24h_mm": 65.0,
        "severity": "HIGH",
        "advisory_template": "Elevated rainfall forecast (~{val} mm in 24h). Monitor low-lying terrain and local stormwater drainage channels for potential localized accumulation.",
        "source_type": "POLICY-DEFINED ADVISORY THRESHOLDS",
    },
    {
        "max_24h_mm": float("inf"),
        "severity": "CRITICAL",
        "advisory_template": "Heavy rainfall forecast (~{val} mm in 24h). High runoff expected; significant risk of surface water accumulation in low-lying areas.",
        "source_type": "POLICY-DEFINED ADVISORY THRESHOLDS",
    },
]


def evaluate_future_rainfall_advisory(next_24h_val):
    """
    Evaluates policy-defined advisory rules for upcoming 24h forecast rainfall.
    Returns (severity, advisory_text, source_type).
    """
    val = round(float(next_24h_val), 2)
    for policy in FUTURE_RAINFALL_ADVISORY_POLICY:
        if val < policy["max_24h_mm"]:
            text = policy["advisory_template"].format(val=val)
            return policy["severity"], text, policy["source_type"]
    last = FUTURE_RAINFALL_ADVISORY_POLICY[-1]
    return last["severity"], last["advisory_template"].format(val=val), last["source_type"]


def build_live_features(lat, lon, state="LIVE", district="LIVE", raw_hourly_forecast=None, raw_terrain=None):
    """
    Recreates the live inference row from the notebook:
    - last 7 days hourly Open-Meteo data
    - rainfall windows
    - elevation/slope
    - exact notebook engineered features
    """
    if raw_hourly_forecast is not None:
        data = raw_hourly_forecast
    else:
        data = get_json(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",
                    "wind_speed_10m",
                    "surface_pressure",
                ]),
                "past_days": 7,
                "forecast_days": 4,
                "timezone": "Asia/Kolkata",
            }
        )

    df_all = pd.DataFrame(data["hourly"])
    df_all["datetime"] = pd.to_datetime(df_all["time"])

    now_india = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .tz_localize(None)
        .floor("h")
    )

    # Historical/recent slice: strictly <= now_india
    hourly = (
        df_all[df_all["datetime"] <= now_india]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(hourly) < 168:
        raise ValueError(
            "Less than 7 days of hourly data available."
        )

    hourly["rainfall_1h"] = (
        hourly["rain"]
        .fillna(0)
        .clip(lower=0)
    )

    hourly["rainfall_3h"] = (
        hourly["rainfall_1h"]
        .rolling(3, min_periods=1)
        .sum()
    )

    hourly["rainfall_6h"] = (
        hourly["rainfall_1h"]
        .rolling(6, min_periods=1)
        .sum()
    )

    hourly["rainfall_12h"] = (
        hourly["rainfall_1h"]
        .rolling(12, min_periods=1)
        .sum()
    )

    hourly["rainfall_24h"] = (
        hourly["rainfall_1h"]
        .rolling(24, min_periods=1)
        .sum()
    )

    hourly["rainfall_3d"] = (
        hourly["rainfall_1h"]
        .rolling(72, min_periods=1)
        .sum()
    )

    hourly["rainfall_7d"] = (
        hourly["rainfall_1h"]
        .rolling(168, min_periods=1)
        .sum()
    )

    latest = hourly.iloc[-1]

    terrain_res = get_live_terrain(
        lat,
        lon,
        precomputed_terrain=raw_terrain,
        return_details=True,
    )
    elevation = terrain_res["elevation_m"]
    slope = terrain_res["slope_deg"]

    # Preserved 30-feature training vector for ML model
    live_row = pd.DataFrame([{
        "state": state,
        "district": district,
        "datetime": latest["datetime"],
        "temperature": latest["temperature_2m"],
        "humidity": latest["relative_humidity_2m"],
        "wind_speed": latest["wind_speed_10m"],
        "pressure": latest["surface_pressure"],
        "latitude": lat,
        "longitude": lon,
        "rainfall_1h": latest["rainfall_1h"],
        "rainfall_3h": latest["rainfall_3h"],
        "rainfall_6h": latest["rainfall_6h"],
        "rainfall_24h": latest["rainfall_24h"],
        "rainfall_3d": latest["rainfall_3d"],
        "rainfall_7d": latest["rainfall_7d"],
        "elevation": elevation,
        "slope": slope,
    }])

    live_row = engineer_ml_features(live_row)

    live_features = live_row[
        SELECTED_FEATURES
    ].copy()

    # Observed / Recent Rainfall Metadata Windows
    recent_windows = {}
    recent_defs = [
        ("1h", 1),
        ("3h", 3),
        ("6h", 6),
        ("12h", 12),
        ("24h", 24),
        ("3d", 72),
        ("7d", 168),
    ]
    for win_name, count in recent_defs:
        sub = hourly.iloc[-count:]
        future_cnt = int((sub["datetime"] > now_india).sum())
        if future_cnt > 0:
            raise RuntimeError(f"Observed rainfall window {win_name} contains {future_cnt} future records!")
        recent_windows[win_name] = {
            "value": round(float(sub["rainfall_1h"].sum()), 2),
            "unit": "mm",
            "type": "observed_recent",
            "source": "Open-Meteo Historical/Recent Hourly Model Data",
            "start_time": str(sub.iloc[0]["datetime"]),
            "end_time": str(sub.iloc[-1]["datetime"]),
            "timezone": "Asia/Kolkata",
            "record_count": len(sub),
            "future_record_count": future_cnt,
        }

    # Future Forecast Rainfall Windows (strictly > now_india)
    df_future = (
        df_all[df_all["datetime"] > now_india]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df_future["rain"] = df_future["rain"].fillna(0).clip(lower=0)

    forecast_windows = {}
    forecast_defs = [
        ("next_1h", 1),
        ("next_3h", 3),
        ("next_6h", 6),
        ("next_12h", 12),
        ("next_24h", 24),
        ("next_48h", 48),
        ("next_72h", 72),
    ]
    for win_name, count in forecast_defs:
        sub = df_future.iloc[:count]
        if len(sub) > 0:
            hist_cnt = int((sub["datetime"] <= now_india).sum())
            if hist_cnt > 0:
                raise RuntimeError(f"Forecast rainfall window {win_name} contains {hist_cnt} historical records!")
            forecast_windows[win_name] = {
                "value": round(float(sub["rain"].sum()), 2),
                "unit": "mm",
                "type": "forecast",
                "source": "Open-Meteo Hourly Forecast",
                "start_time": str(sub.iloc[0]["datetime"]),
                "end_time": str(sub.iloc[-1]["datetime"]),
                "timezone": "Asia/Kolkata",
                "record_count": len(sub),
                "historical_record_count": hist_cnt,
            }
        else:
            forecast_windows[win_name] = {
                "value": 0.0,
                "unit": "mm",
                "type": "forecast",
                "source": "Open-Meteo Hourly Forecast",
                "start_time": None,
                "end_time": None,
                "timezone": "Asia/Kolkata",
                "record_count": 0,
                "historical_record_count": 0,
            }

    # Future forecast characteristics
    f72 = df_future.iloc[:72]
    max_hourly_rain = round(float(f72["rain"].max()), 2) if len(f72) > 0 else 0.0
    max_hourly_time = None
    if len(f72) > 0 and max_hourly_rain > 0:
        max_idx = f72["rain"].idxmax()
        max_hourly_time = str(f72.loc[max_idx]["datetime"])
    total_forecast_72h = round(float(f72["rain"].sum()), 2) if len(f72) > 0 else 0.0

    meaningful = f72[f72["rain"] >= 0.5]
    first_meaningful_rain_time = str(meaningful.iloc[0]["datetime"]) if len(meaningful) > 0 else None

    # Identify most significant forecast rainfall period (highest 6h rolling window in next 72h)
    most_significant_window = None
    if len(f72) >= 6:
        r6 = f72["rain"].rolling(6).sum()
        max_6h_idx = r6.idxmax()
        max_6h_val = round(float(r6.loc[max_6h_idx]), 2)
        if max_6h_val > 0:
            start_pos = max(0, f72.index.get_loc(max_6h_idx) - 5)
            s_dt = f72.iloc[start_pos]["datetime"]
            e_dt = f72.loc[max_6h_idx]["datetime"]
            most_significant_window = {
                "period": f"{s_dt.strftime('%H:%M')}–{e_dt.strftime('%H:%M IST (%d %b)')}",
                "start_time": str(s_dt),
                "end_time": str(e_dt),
                "total_rain_mm": max_6h_val,
            }

    # Evaluate deterministic policy-defined advisory thresholds
    next_24h_val = forecast_windows.get("next_24h", {}).get("value", 0.0)
    outlook_severity, outlook_text, outlook_source = evaluate_future_rainfall_advisory(next_24h_val)

    future_outlook = {
        "summary": outlook_text,
        "severity": outlook_severity,
        "source_type": outlook_source,
        "max_hourly_rain_mm": max_hourly_rain,
        "max_hourly_time": max_hourly_time,
        "total_cumulative_72h_mm": total_forecast_72h,
        "first_meaningful_rain_time": first_meaningful_rain_time,
        "most_significant_window": most_significant_window,
    }

    return {
        "features": live_features.iloc[0].to_dict(),
        "source_time": str(latest["datetime"]),
        "analysis_time_ist": str(now_india),
        "state": state,
        "district": district,
        "latitude": float(lat),
        "longitude": float(lon),
        "rainfall_summary_mm": {
            "rainfall_1h": float(latest["rainfall_1h"]),
            "rainfall_3h": float(latest["rainfall_3h"]),
            "rainfall_6h": float(latest["rainfall_6h"]),
            "rainfall_12h": float(latest["rainfall_12h"]),
            "rainfall_24h": float(latest["rainfall_24h"]),
            "rainfall_3d": float(latest["rainfall_3d"]),
            "rainfall_7d": float(latest["rainfall_7d"]),
        },
        "recent_rainfall": recent_windows,
        "forecast_rainfall": forecast_windows,
        "future_outlook": future_outlook,
        "terrain": terrain_res,
    }


# Standard platform alias for consistent multi-module integration
build_live_waterlogging_features = build_live_features
