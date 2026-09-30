import time
import json
import hashlib
import threading
from typing import Any, Optional, Dict, Tuple

class TTLCache:
    """
    Thread-safe in-memory cache with per-item TTL expiration.
    """
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            val, expire_at = self._store[key]
            if time.time() > expire_at:
                del self._store[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            expire_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            self._store[key] = (value, expire_at)

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def count(self) -> int:
        with self._lock:
            now = time.time()
            return sum(1 for _, exp in self._store.values() if exp > now)

provider_cache = TTLCache(default_ttl=300.0)

# Scientific TTL defaults (in seconds)
TTL_TERRAIN = 3600.0    # 1 hour (static physical topography)
TTL_FLOOD = 300.0       # 5 minutes (daily hydrology model)
TTL_USGS = 300.0        # 5 minutes (recent earthquake events)
TTL_AIR = 300.0         # 5 minutes (hourly air quality)
TTL_WEATHER = 300.0     # 5 minutes (hourly atmospheric forecast)
TTL_AG_DAILY = 300.0    # 5 minutes (14-day daily forecast)
TTL_WL_HOURLY = 300.0   # 5 minutes (7-day hourly forecast)
TTL_GDACS = 300.0       # 5 minutes (active cyclone/volcano feed)
TTL_SPATIAL_RAINFALL = 1800.0       # 30 minutes (Open-Meteo spatial rainfall grid & windows)

def make_cache_key(provider: str, lat: float, lon: float, params: Optional[dict] = None) -> str:
    """
    Constructs a deterministic cache key preserving exact floating-point coordinate precision.
    Uses repr(float(coord)) to avoid any decimal rounding/truncation.
    Canonicalizes provider parameters with sorted keys and SHA256 hashing.
    """
    lat_key = repr(float(lat))
    lon_key = repr(float(lon))
    param_str = json.dumps(params or {}, sort_keys=True, default=str)
    param_hash = hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{lat_key}:{lon_key}:{param_hash}"
