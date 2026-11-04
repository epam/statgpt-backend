"""Unit tests for the shared HTTP connection pools."""

import httpx
import openai
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


class TestRateLimitPolicy:
    """The shared LLM client lets the openai SDK retry a 429 only for a short `Retry-After`."""

    @staticmethod
    def _client(*responses: httpx.Response) -> tuple[openai.AsyncOpenAI, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return responses[min(len(requests), len(responses)) - 1]

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"response": [http_pool._enforce_rate_limit_policy]},
        )
        client = openai.AsyncOpenAI(
            api_key="test", base_url="http://dial.test", max_retries=3, http_client=http_client
        )
        return client, requests

    @staticmethod
    async def _create(client: openai.AsyncOpenAI) -> None:
        await client.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "Hello!"}]
        )

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(openai._base_client.anyio, "sleep", fake_sleep)
        return sleeps

    async def test_shared_client_has_the_hook(self) -> None:
        hooks = http_pool.get_shared_llm_http_client().event_hooks["response"]

        assert http_pool._enforce_rate_limit_policy in hooks

    @pytest.mark.parametrize(
        "headers",
        [{"retry-after": "53932"}, {"retry-after": "60"}, {}, {"retry-after": "soon"}],
        ids=["long", "60s", "missing", "unparseable"],
    )
    async def test_fails_fast_without_a_short_retry_after(self, headers, _no_sleep) -> None:
        client, requests = self._client(httpx.Response(429, headers=headers, json=_ERROR))

        with pytest.raises(openai.RateLimitError):
            await self._create(client)

        assert len(requests) == 1
        assert _no_sleep == []

    @pytest.mark.parametrize(
        "headers, expected_sleep",
        [({"retry-after": "1"}, 1.0), ({"retry-after-ms": "500"}, 0.5)],
        ids=["retry-after", "retry-after-ms"],
    )
    async def test_waits_and_retries_a_short_retry_after(
        self, headers, expected_sleep, _no_sleep
    ) -> None:
        client, requests = self._client(
            httpx.Response(429, headers=headers, json=_ERROR), httpx.Response(200, json=_COMPLETION)
        )

        await self._create(client)

        assert len(requests) == 2
        assert _no_sleep == [expected_sleep]

    async def test_still_retries_server_errors(self) -> None:
        client, requests = self._client(
            httpx.Response(500, json=_ERROR), httpx.Response(200, json=_COMPLETION)
        )

        await self._create(client)

        assert len(requests) == 2


_ERROR = {"error": {"message": "Hit cost rate limit.", "code": "429"}}
_COMPLETION = {
    "id": "1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}
    ],
}
