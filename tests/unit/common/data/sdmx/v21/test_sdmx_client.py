"""Tests for AsyncSdmxClient._perform_request's timeout retry/error behavior."""

from pathlib import Path

import httpx
import pytest
import requests
from sdmx.message import DataMessage

from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient, SdmxRequestTimeoutError

SDMX_JSON_FIXTURE = Path(__file__).parents[1] / "common" / "sdmx_json_1_0_attribute_levels.json"


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


@pytest.mark.parametrize(
    "error_cls",
    [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError],
)
async def test_perform_request_retries_transient_transport_error_then_succeeds(error_cls) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise error_cls("simulated transport error", request=request)
        return httpx.Response(200, content=b"{}")

    client = _client(handler)

    response = await client._perform_request(_prepared_request(), max_retries=3, delay=0)

    assert response.status_code == 200
    assert calls["count"] == 2


async def test_perform_request_reraises_transport_error_after_exhausting_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("simulated connect error", request=request)

    client = _client(handler)

    with pytest.raises(httpx.ConnectError):
        await client._perform_request(_prepared_request(), max_retries=2, delay=0)

    assert calls["count"] == 2


async def test_perform_request_does_not_retry_non_transient_transport_error() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.UnsupportedProtocol("simulated bad scheme", request=request)

    client = _client(handler)

    with pytest.raises(httpx.UnsupportedProtocol):
        await client._perform_request(_prepared_request(), max_retries=3, delay=0)

    assert calls["count"] == 1


def test_describe_exception_chain_includes_empty_message_causes() -> None:
    try:
        try:
            raise ConnectionResetError("peer reset")
        except ConnectionResetError as root:
            raise httpx.ConnectError("") from root
    except httpx.ConnectError as e:
        description = AsyncSdmxClient._describe_exception_chain(e)

    assert description == "httpx.ConnectError <- ConnectionResetError: peer reset"


def test_parse_response_resolves_attributes_of_each_sdmx_json_series() -> None:
    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.invalid/data"
    response.headers["content-type"] = "text/json"
    response._content = SDMX_JSON_FIXTURE.read_bytes()

    msg = AsyncSdmxClient._parse_response(response)
    assert isinstance(msg, DataMessage)

    assert {sk.attrib["SCALE"].value.id for sk in msg.data[0].series} == {"9", "6"}
