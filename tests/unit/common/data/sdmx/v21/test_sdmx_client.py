"""Tests for AsyncSdmxClient._perform_request's timeout retry/error behavior."""

import httpx
import pytest
import requests

from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient, SdmxRequestTimeoutError


def _prepared_request() -> requests.PreparedRequest:
    return requests.Request(method="GET", url="https://example.invalid/data").prepare()


def _client(handler) -> AsyncSdmxClient:
    transport = httpx.MockTransport(handler)
    httpx_client = httpx.AsyncClient(transport=transport)
    return AsyncSdmxClient(
        sync_client=None,  # type: ignore[arg-type]
        httpx_client=httpx_client,
        authorizer=None,
        rate_limiter=None,  # type: ignore[arg-type]
    )


async def test_perform_request_raises_sdmx_timeout_error_after_exhausting_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ReadTimeout("simulated read timeout", request=request)

    client = _client(handler)

    with pytest.raises(SdmxRequestTimeoutError, match="timed out after 2 attempt"):
        await client._perform_request(_prepared_request(), max_retries=2, delay=0)

    assert calls["count"] == 2


async def test_perform_request_retries_connect_timeout_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectTimeout("simulated connect timeout", request=request)
        return httpx.Response(200, content=b"{}")

    client = _client(handler)

    response = await client._perform_request(_prepared_request(), max_retries=3, delay=0)

    assert response.status_code == 200
    assert calls["count"] == 2
