"""Unit tests for the shared HTTP connection pools."""

import httpx
import pytest

from statgpt.common.utils import http_pool


@pytest.fixture(autouse=True)
async def _reset_pools():
    await http_pool.close_shared_http_clients()
    yield
    await http_pool.close_shared_http_clients()


class TestSharedLlmHttpClient:
    async def test_returns_same_client(self) -> None:
        assert http_pool.get_shared_llm_http_client() is http_pool.get_shared_llm_http_client()

    async def test_follows_redirects_like_openai_sdk_default(self) -> None:
        assert http_pool.get_shared_llm_http_client().follow_redirects is True

    async def test_recreates_after_close(self) -> None:
        client = http_pool.get_shared_llm_http_client()
        await client.aclose()

        new_client = http_pool.get_shared_llm_http_client()

        assert new_client is not client
        assert not new_client.is_closed


class TestSharedSdmxHttpClient:
    async def test_returns_same_client_per_key(self) -> None:
        client_a = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "k"})
        client_b = http_pool.get_shared_sdmx_http_client("source-b")

        assert client_a is not client_b
        assert (
            http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "k"}) is client_a
        )
        assert http_pool.get_shared_sdmx_http_client("source-b") is client_b

    async def test_none_and_empty_headers_are_equivalent(self) -> None:
        client = http_pool.get_shared_sdmx_http_client("source-a")
        assert http_pool.get_shared_sdmx_http_client("source-a", headers={}) is client

    async def test_distinct_header_sets_coexist(self) -> None:
        old_client = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "old"})

        new_client = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "new"})

        assert new_client is not old_client
        # The old client may still serve in-flight requests; it is closed on shutdown only.
        assert not old_client.is_closed
        assert (
            http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "new"})
            is new_client
        )
        # Both header sets stay pooled under their own keys — no create/retire thrashing.
        assert (
            http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "old"})
            is old_client
        )

    async def test_recreates_after_close(self) -> None:
        client = http_pool.get_shared_sdmx_http_client("source-a")
        await client.aclose()

        new_client = http_pool.get_shared_sdmx_http_client("source-a")

        assert new_client is not client
        assert not new_client.is_closed

    async def test_client_static_headers_and_timeout(self) -> None:
        client = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "secret"})

        assert client.headers["api-key"] == "secret"
        assert client.timeout == httpx.Timeout(90.0, connect=45.0)


class TestCloseSharedHttpClients:
    async def test_closes_all_clients_including_superseded_header_sets(self) -> None:
        llm_client = http_pool.get_shared_llm_http_client()
        superseded = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "old"})
        current = http_pool.get_shared_sdmx_http_client("source-a", headers={"api-key": "new"})
        other = http_pool.get_shared_sdmx_http_client("source-b")

        await http_pool.close_shared_http_clients()

        assert llm_client.is_closed
        assert superseded.is_closed
        assert current.is_closed
        assert other.is_closed

    async def test_pools_repopulate_after_close(self) -> None:
        llm_client = http_pool.get_shared_llm_http_client()
        sdmx_client = http_pool.get_shared_sdmx_http_client("source-a")

        await http_pool.close_shared_http_clients()

        assert http_pool.get_shared_llm_http_client() is not llm_client
        assert http_pool.get_shared_sdmx_http_client("source-a") is not sdmx_client

    async def test_close_is_idempotent(self) -> None:
        http_pool.get_shared_llm_http_client()
        http_pool.get_shared_sdmx_http_client("source-a")

        await http_pool.close_shared_http_clients()
        await http_pool.close_shared_http_clients()  # must not raise
