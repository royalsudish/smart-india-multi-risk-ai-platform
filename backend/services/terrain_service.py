from math import cos, radians, atan, sqrt
from backend.data_sources.open_meteo import elevations

def terrain_features(lat, lon, sample_distance_m=100):
    """
    Estimates local slope from center + N/S/E/W elevation samples.
    This is an approximation from 90 m DEM point samples.
    """
    dlat = sample_distance_m / 111320.0
    dlon = sample_distance_m / (111320.0 * max(0.01, cos(radians(lat))))

    pts = [
        (lat, lon),
        (lat+dlat, lon),
        (lat-dlat, lon),
        (lat, lon+dlon),
        (lat, lon-dlon),
    ]
    z = elevations(pts)
    if len(z) != 5:
        raise RuntimeError("Elevation API did not return all terrain samples")

    center, north, south, east, west = map(float, z)
    dzdy = (north - south) / (2 * sample_distance_m)
    dzdx = (east - west) / (2 * sample_distance_m)
    gradient = sqrt(dzdx*dzdx + dzdy*dzdy)
    slope_deg = atan(gradient) * 180.0 / 3.141592653589793

    return {
        "elevation_m": center,
        "slope_deg_approx": slope_deg,
        "sample_distance_m": sample_distance_m
    }
