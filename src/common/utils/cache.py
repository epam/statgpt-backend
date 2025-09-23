import time
from typing import Generic, NamedTuple, TypeVar

T = TypeVar('T')


class CacheItem(NamedTuple, Generic[T]):
    value: T
    expiry: float


class Cache(Generic[T]):
    def __init__(self, ttl: int = 3600):
        self._cache: dict[str, CacheItem[T]] = {}
        self._ttl = ttl

    def set(self, key: str, value: T) -> None:
        expiry = time.time() + self._ttl
        self._cache[key] = CacheItem(value=value, expiry=expiry)

    def get(self, key: str, default: T | None = None) -> T | None:
        if key in self._cache:
            item = self._cache[key]
            if time.time() < item.expiry:
                return item.value
            else:
                self._remove_expired_item(key)
        return default

    def clear(self) -> None:
        """Clear all items from the cache"""
        self._cache.clear()

    def cleanup(self) -> None:
        """Remove all expired items from the cache"""
        current_time = time.time()
        expired_keys = [key for key, item in self._cache.items() if current_time >= item.expiry]
        for key in expired_keys:
            self._remove_expired_item(key)

    def _remove_expired_item(self, key: str) -> None:
        self._cache.pop(key, None)
