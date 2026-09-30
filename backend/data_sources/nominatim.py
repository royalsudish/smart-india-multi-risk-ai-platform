import time
import threading
from backend.config.settings import (
    NOMINATIM_URL, NOMINATIM_REVERSE_URL, APP_USER_AGENT, APP_CONTACT_EMAIL, REQUEST_TIMEOUT_SECONDS
)
from backend.data_sources.http_client import session

_lock = threading.Lock()
_last_call = 0.0

# In-memory lightweight geocode cache
_GEOCODE_CACHE = {}
_GEOCODE_CACHE_LOCK = threading.Lock()

def _valid_contact_email(value: str) -> bool:
    """
    Validate that contact email is non-empty and does not contain placeholder patterns.
    """
    if not value:
        return False
    val = value.strip().lower()
    invalid_patterns = [
        "example.com",
        "your-email",
        "your_email",
        "test@test",
    ]
    return not any(pattern in val for pattern in invalid_patterns)

def search_place(query: str, limit: int = 5):
    """
    Public Nominatim usage requires a descriptive User-Agent and light usage.
    This client throttles calls to <= 1 request/second inside one app process,
    restricts results to India (countrycodes=in), and caches responses in memory.
    """
    global _last_call

    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")

    cache_key = f"search:{query.lower()}:{limit}"
    with _GEOCODE_CACHE_LOCK:
        if cache_key in _GEOCODE_CACHE:
            return _GEOCODE_CACHE[cache_key]

    with _lock:
        # Re-check cache under lock
        with _GEOCODE_CACHE_LOCK:
            if cache_key in _GEOCODE_CACHE:
                return _GEOCODE_CACHE[cache_key]

        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)

        params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "countrycodes": "in",
            "limit": max(1, min(int(limit), 10)),
        }
        if _valid_contact_email(APP_CONTACT_EMAIL):
            params["email"] = APP_CONTACT_EMAIL

        s = session()
        s.headers["User-Agent"] = APP_USER_AGENT
        r = s.get(NOMINATIM_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        _last_call = time.monotonic()
        r.raise_for_status()
        data = r.json()

        with _GEOCODE_CACHE_LOCK:
            _GEOCODE_CACHE[cache_key] = data
        return data

def reverse_geocode(lat: float, lon: float):
    """
    Reverse geocodes coordinates to a place name using Nominatim.
    Respects rate limiting, caching, User-Agent, and validation.
    """
    global _last_call

    cache_key = f"rev:{lat:.6f}:{lon:.6f}"
    with _GEOCODE_CACHE_LOCK:
        if cache_key in _GEOCODE_CACHE:
            return _GEOCODE_CACHE[cache_key]

    with _lock:
        with _GEOCODE_CACHE_LOCK:
            if cache_key in _GEOCODE_CACHE:
                return _GEOCODE_CACHE[cache_key]

        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)

        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
        }
        if _valid_contact_email(APP_CONTACT_EMAIL):
            params["email"] = APP_CONTACT_EMAIL

        s = session()
        s.headers["User-Agent"] = APP_USER_AGENT
        r = s.get(NOMINATIM_REVERSE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        _last_call = time.monotonic()
        r.raise_for_status()
        data = r.json()

        with _GEOCODE_CACHE_LOCK:
            _GEOCODE_CACHE[cache_key] = data
        return data
