"""Unit tests for the AsyncLoadingCache utility class."""

import asyncio
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

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


class TestAsyncLoadingCacheRefresh:

    @pytest.mark.asyncio
    async def test_refresh_replaces_live_entry(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        await cache.get("k", AsyncMock(return_value="old"))

        result = await cache.refresh("k", AsyncMock(return_value="new"))

        assert result == "new"
        loader = AsyncMock(return_value="wrong")
        assert await cache.get("k", loader) == "new"
        loader.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_extends_expiry(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache(ttl=60)
        start = time.monotonic()
        await cache.get("k", AsyncMock(return_value="old"))

        with patch.object(time, "monotonic", return_value=start + 50):
            await cache.refresh("k", AsyncMock(return_value="new"))

        # Past the original expiry (start + 60) but before the refreshed one (start + 110)
        loader = AsyncMock(return_value="reloaded")
        with patch.object(time, "monotonic", return_value=start + 100):
            result = await cache.get("k", loader)

        assert result == "new"
        loader.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_get_during_refresh_waits_and_sees_new_value(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        release = asyncio.Event()

        async def slow_refresh_loader() -> str:
            await release.wait()
            return "refreshed"

        refresh_task = asyncio.ensure_future(cache.refresh("k", slow_refresh_loader))
        await asyncio.sleep(0)  # let refresh acquire the per-key lock

        get_loader = AsyncMock(return_value="from-get")
        get_task = asyncio.ensure_future(cache.get("k", get_loader))
        await asyncio.sleep(0)  # get misses the cache and blocks on the lock

        release.set()

        assert await refresh_task == "refreshed"
        assert await get_task == "refreshed"
        get_loader.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_failure_keeps_existing_entry(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        await cache.get("k", AsyncMock(return_value="old"))

        with pytest.raises(ValueError, match="boom"):
            await cache.refresh("k", AsyncMock(side_effect=ValueError("boom")))

        result = await cache.get("k", AsyncMock(return_value="wrong"))
        assert result == "old"


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


class TestAsyncLoadingCacheTtl:

    @pytest.mark.asyncio
    async def test_ttl_expiry_triggers_reload(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache(ttl=60)
        loader = AsyncMock(side_effect=["old", "new"])

        await cache.get("k", loader)
        with patch.object(time, "monotonic", return_value=time.monotonic() + 120):
            result = await cache.get("k", loader)

        assert result == "new"
        assert loader.await_count == 2

    @pytest.mark.asyncio
    async def test_no_ttl_means_no_expiry(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        loader = AsyncMock(return_value="value")

        await cache.get("k", loader)
        with patch.object(time, "monotonic", return_value=time.monotonic() + 999999):
            result = await cache.get("k", loader)

        assert result == "value"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ttl_not_expired_returns_cached(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache(ttl=60)
        loader = AsyncMock(return_value="value")

        await cache.get("k", loader)
        with patch.object(time, "monotonic", return_value=time.monotonic() + 30):
            result = await cache.get("k", loader)

        assert result == "value"
        loader.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ttl_and_validator_both_checked(self) -> None:
        cache: AsyncLoadingCache[str] = AsyncLoadingCache(ttl=60)
        loader = AsyncMock(side_effect=["old", "new"])

        await cache.get("k", loader)
        result = await cache.get("k", loader, validator=lambda v: v == "new")

        assert result == "new"
        assert loader.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_load_failure_propagates_to_both(self) -> None:
        """When two tasks piggyback on the same future and it raises,
        both should see the exception and the entry should be cleaned up for retry."""
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        event = asyncio.Event()

        async def failing_loader() -> str:
            await event.wait()
            raise ValueError("boom")

        task1 = asyncio.ensure_future(cache.get("k", failing_loader))
        task2 = asyncio.ensure_future(cache.get("k", failing_loader))
        await asyncio.sleep(0)  # let both tasks start and piggyback

        event.set()

        with pytest.raises(ValueError, match="boom"):
            await task1
        with pytest.raises(ValueError, match="boom"):
            await task2

        # Entry should be cleaned up — next load succeeds
        loader = AsyncMock(return_value="recovered")
        result = await cache.get("k", loader)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_concurrent_validator_failure_deduplicates_reload(self) -> None:
        """When two tasks await the same cached future and both validators reject,
        only one reload should happen (the second task piggybacks on the first)."""
        cache: AsyncLoadingCache[str] = AsyncLoadingCache()
        event = asyncio.Event()
        load_count = 0

        async def counting_loader() -> str:
            nonlocal load_count
            load_count += 1
            if load_count == 1:
                return "old"
            await event.wait()
            return "new"

        # Seed the cache with "old"
        result = await cache.get("k", counting_loader)
        assert result == "old"
        assert load_count == 1

        # Both tasks reject "old" via validator
        reject_old: Callable[[str], bool] = lambda v: v != "old"
        task1 = asyncio.ensure_future(cache.get("k", counting_loader, validator=reject_old))
        task2 = asyncio.ensure_future(cache.get("k", counting_loader, validator=reject_old))
        await asyncio.sleep(0)  # let both tasks discover "old" and reject it

        event.set()
        results = await asyncio.gather(task1, task2)

        assert results == ["new", "new"]
        assert load_count == 2  # 1 initial + 1 reload (not 3)
