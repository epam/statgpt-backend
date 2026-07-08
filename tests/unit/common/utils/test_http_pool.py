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
    async def test_returns_same_client_per_source(self) -> None:
        client_a = http_pool.get_shared_sdmx_http_client("source-a")
        client_b = http_pool.get_shared_sdmx_http_client("source-b")

        assert client_a is not client_b
        assert http_pool.get_shared_sdmx_http_client("source-a") is client_a
        assert http_pool.get_shared_sdmx_http_client("source-b") is client_b

    async def test_recreates_after_close(self) -> None:
        client = http_pool.get_shared_sdmx_http_client("source-a")
        await client.aclose()

        new_client = http_pool.get_shared_sdmx_http_client("source-a")

        assert new_client is not client
        assert not new_client.is_closed

    async def test_client_timeout(self) -> None:
        client = http_pool.get_shared_sdmx_http_client("source-a")

        assert client.timeout == httpx.Timeout(90.0, connect=45.0)


class TestCloseSharedHttpClients:
    async def test_closes_all_clients(self) -> None:
        llm_client = http_pool.get_shared_llm_http_client()
        sdmx_a = http_pool.get_shared_sdmx_http_client("source-a")
        sdmx_b = http_pool.get_shared_sdmx_http_client("source-b")

        await http_pool.close_shared_http_clients()

        assert llm_client.is_closed
        assert sdmx_a.is_closed
        assert sdmx_b.is_closed

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


class TestSharedHttpClientsContext:
    async def test_closes_clients_on_exit(self) -> None:
        async with http_pool.shared_http_clients_context():
            llm_client = http_pool.get_shared_llm_http_client()
            sdmx_client = http_pool.get_shared_sdmx_http_client("source-a")

        assert llm_client.is_closed
        assert sdmx_client.is_closed

    async def test_closes_clients_on_error(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with http_pool.shared_http_clients_context():
                client = http_pool.get_shared_llm_http_client()
                raise RuntimeError("boom")

        assert client.is_closed
