import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, NamedTuple, TypeVar

T = TypeVar('T')


class _CacheEntry(NamedTuple, Generic[T]):
    future: asyncio.Future[T]
    expires_at: float


class AsyncLoadingCache(Generic[T]):
    """A cache that loads values asynchronously on cache miss,
    with optional validation and TTL-based expiration.

    Concurrent requests for the same key are deduplicated: only one
    load runs while other callers await its result. In-flight futures
    are never evicted by TTL.
    """

    def __init__(self, ttl: int | None = None) -> None:
        self._cache: dict[str, _CacheEntry[T]] = {}
        self._ttl = ttl

    async def get(
        self,
        key: str,
        loader: Callable[[], Awaitable[T]],
        validator: Callable[[T], bool] | None = None,
    ) -> T:
        if key in self._cache:
            entry = self._cache[key]
            if entry.future.done() and time.time() >= entry.expires_at:
                self._cache.pop(key, None)
            else:
                value = await entry.future
                if validator is None or validator(value):
                    return value
                self._cache.pop(key, None)

        future = asyncio.ensure_future(loader())
        self._cache[key] = _CacheEntry(
            future=future,
            expires_at=time.time() + self._ttl if self._ttl is not None else float('inf'),
        )
        try:
            return await future
        except BaseException:  # includes CancelledError to avoid caching canceled futures
            self._cache.pop(key, None)
            raise

    def remove(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
