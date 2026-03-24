import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, NamedTuple, TypeVar

T = TypeVar('T')


class _CacheEntry(NamedTuple, Generic[T]):
    value: T
    expires_at: float


class AsyncLoadingCache(Generic[T]):
    """A cache that loads values asynchronously on cache miss,
    with optional validation and TTL-based expiration.

    Uses per-key locks to deduplicate concurrent loads:
    only one coroutine runs the loader while others wait
    on the lock and then read the cached result.
    """

    def __init__(self, ttl: int | None = None) -> None:
        self._cache: dict[str, _CacheEntry[T]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._ttl = ttl

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _is_valid(self, key: str, validator: Callable[[T], bool] | None) -> bool:
        if key not in self._cache:
            return False
        entry = self._cache[key]
        if time.time() >= entry.expires_at:
            self._cache.pop(key, None)
            return False
        if validator is not None and not validator(entry.value):
            self._cache.pop(key, None)
            return False
        return True

    async def get(
        self,
        key: str,
        loader: Callable[[], Awaitable[T]],
        validator: Callable[[T], bool] | None = None,
    ) -> T:
        # Fast path: return cached value without acquiring lock
        if self._is_valid(key, validator):
            return self._cache[key].value

        # Slow path: acquire per-key lock, then load if still missing
        lock = self._get_lock(key)
        async with lock:
            # Double-check after acquiring the lock
            if self._is_valid(key, validator):
                return self._cache[key].value

            value = await loader()
            self._cache[key] = _CacheEntry(
                value=value,
                expires_at=time.time() + self._ttl if self._ttl is not None else float('inf'),
            )
            return value

    def remove(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
