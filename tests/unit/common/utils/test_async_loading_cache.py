"""Unit tests for the AsyncLoadingCache utility class."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from statgpt.common.utils.async_loading_cache import AsyncLoadingCache


class TestAsyncLoadingCacheGet:

    @pytest.mark.asyncio
    async def test_get_loads_on_miss(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(return_value="value")

        result = await cache.get("k", loader)

        assert result == "value"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_returns_cached_on_hit(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(return_value="value")

        await cache.get("k", loader)
        result = await cache.get("k", loader)

        assert result == "value"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_reloads_when_validator_fails(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(side_effect=["old", "new"])

        await cache.get("k", loader)
        result = await cache.get("k", loader, validator=lambda v: v == "new")

        assert result == "new"
        assert loader.await_count == 2

    @pytest.mark.asyncio
    async def test_get_without_validator_always_hits(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(return_value="value")

        await cache.get("k", loader)
        result = await cache.get("k", loader)

        assert result == "value"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_get_deduplicates(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(return_value="value")

        results = await asyncio.gather(
            cache.get("k", loader),
            cache.get("k", loader),
        )

        assert results == ["value", "value"]
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_failure_not_cached(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(side_effect=[ValueError("fail"), "value"])

        with pytest.raises(ValueError, match="fail"):
            await cache.get("k", loader)

        result = await cache.get("k", loader)
        assert result == "value"
        assert loader.await_count == 2


class TestAsyncLoadingCacheRemove:

    @pytest.mark.asyncio
    async def test_remove_triggers_reload(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(side_effect=["first", "second"])

        await cache.get("k", loader)
        cache.remove("k")
        result = await cache.get("k", loader)

        assert result == "second"
        assert loader.await_count == 2

    @pytest.mark.asyncio
    async def test_remove_nonexistent_key(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        cache.remove("nonexistent")  # should not raise


class TestAsyncLoadingCacheClear:

    @pytest.mark.asyncio
    async def test_clear_removes_all(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(side_effect=["a1", "b1", "a2", "b2"])

        await cache.get("a", loader)
        await cache.get("b", loader)
        cache.clear()
        await cache.get("a", loader)
        await cache.get("b", loader)

        assert loader.await_count == 4
