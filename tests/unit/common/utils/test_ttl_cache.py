"""Unit tests for the TtlCache utility class."""

import time
from unittest.mock import patch

from statgpt.common.utils.ttl_cache import TtlCache


class TestTtlCacheRemove:
    """Tests for the TtlCache.remove() method."""

    def test_remove_existing_key(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        cache.set("key1", "value1")

        cache.remove("key1")

        assert cache.get("key1") is None

    def test_remove_nonexistent_key(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)

        cache.remove("nonexistent")  # should not raise

    def test_remove_does_not_affect_other_keys(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.remove("key1")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"


class TestTtlCacheGetSetBasics:
    """Basic tests for TtlCache get/set behavior."""

    def test_get_returns_set_value(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_get_returns_default_for_missing_key(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        assert cache.get("missing") is None
        assert cache.get("missing", "fallback") == "fallback"

    def test_get_returns_none_for_expired_key(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        cache.set("k", "v")

        with patch.object(time, "monotonic", return_value=time.monotonic() + 120):
            assert cache.get("k") is None

    def test_set_overwrites_existing(self) -> None:
        cache: TtlCache[str] = TtlCache(ttl=60)
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"
