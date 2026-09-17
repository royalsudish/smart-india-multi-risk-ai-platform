import time
import threading
from backend.config.settings import (
    NOMINATIM_URL, APP_USER_AGENT, APP_CONTACT_EMAIL, REQUEST_TIMEOUT_SECONDS
)
from backend.data_sources.http_client import session

_lock = threading.Lock()
_last_call = 0.0

def search_place(query: str, limit: int = 5):
    """
    Public Nominatim usage requires a descriptive User-Agent and light usage.
    This client throttles calls to <= 1 request/second inside one app process.
    """
    global _last_call

    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")

    with _lock:
        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)

        params = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": max(1, min(int(limit), 10)),
        }
        if APP_CONTACT_EMAIL:
            params["email"] = APP_CONTACT_EMAIL

        s = session()
        s.headers["User-Agent"] = APP_USER_AGENT
        r = s.get(NOMINATIM_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        _last_call = time.monotonic()
        r.raise_for_status()
        return r.json()
