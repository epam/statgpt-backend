import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar('T')


class AsyncLoadingCache(Generic[T]):
    """A cache that loads values asynchronously on cache miss,
    with optional validation of cached entries.

    Concurrent requests for the same key are deduplicated: only one
    load runs while other callers await its result.
    """

    def __init__(self) -> None:
        self._cache: dict[str, asyncio.Future[T]] = {}

    async def get(
        self,
        key: str,
        loader: Callable[[], Awaitable[T]],
        validator: Callable[[T], bool] | None = None,
    ) -> T:
        if key in self._cache:
            value = await self._cache[key]
            if validator is None or validator(value):
                return value
            self._cache.pop(key, None)

        self._cache[key] = asyncio.ensure_future(loader())
        try:
            return await self._cache[key]
        except BaseException:  # includes CancelledError to avoid caching canceled futures
            self._cache.pop(key, None)
            raise

    def remove(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
