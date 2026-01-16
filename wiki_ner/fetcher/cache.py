"""Disk-based caching for Wikipedia API responses."""

import os
from typing import Any, Callable, Optional

import diskcache


class WikiCache:
    """Persistent disk cache for Wikipedia data."""

    def __init__(self, cache_dir: str = "./cache", ttl_days: int = 7):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
            ttl_days: Time-to-live for cached items in days.
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)
        self.ttl = ttl_days * 86400  # Convert to seconds

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a cached value.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        self.cache.set(key, value, expire=self.ttl)

    def get_or_fetch(self, key: str, fetch_fn: Callable[[], Any]) -> Any:
        """Get cached value or fetch and cache it.

        Args:
            key: Cache key.
            fetch_fn: Function to call if value not in cache.

        Returns:
            Cached or freshly fetched value.
        """
        value = self.get(key)
        if value is not None:
            return value

        value = fetch_fn()
        if value is not None:
            self.set(key, value)
        return value

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

    def close(self) -> None:
        """Close the cache."""
        self.cache.close()
