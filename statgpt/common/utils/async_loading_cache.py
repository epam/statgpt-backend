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
        while True:
            if key in self._cache:
                entry = self._cache[key]

                # TTL check (only for completed futures — in-flight ones are never evicted)
                if entry.future.done() and time.time() >= entry.expires_at:
                    if self._cache.get(key) is entry:
                        self._cache.pop(key, None)
                    continue

                value = await entry.future
                if validator is None or validator(value):
                    return value

                # Validator failed — evict only if nobody replaced the entry while we awaited
                if self._cache.get(key) is entry:
                    self._cache.pop(key, None)
                continue

            # No entry — create one
            future = asyncio.ensure_future(loader())
            entry = _CacheEntry(
                future=future,
                expires_at=time.time() + self._ttl if self._ttl is not None else float('inf'),
            )
            self._cache[key] = entry
            try:
                return await future
            except BaseException:  # includes CancelledError
                if self._cache.get(key) is entry:
                    self._cache.pop(key, None)
                raise

    def remove(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
