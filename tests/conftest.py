import pytest
from backend.utils.cache import provider_cache

@pytest.fixture(autouse=True)
def clear_provider_cache_between_tests():
    """Ensure in-memory provider cache does not leak state across independent tests."""
    provider_cache.clear()
    yield
    provider_cache.clear()
