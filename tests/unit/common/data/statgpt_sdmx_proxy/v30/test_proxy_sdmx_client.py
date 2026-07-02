"""Tests for the async StatGPT SDMX proxy client."""

import asyncio
import time
from unittest import mock

import httpx
import requests
from sdmx.message import DataMessage

from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import AsyncStatGptSdmxProxyClient


def _make_client() -> AsyncStatGptSdmxProxyClient:
    return AsyncStatGptSdmxProxyClient(
        sync_client=mock.Mock(),
        httpx_client=mock.Mock(spec=httpx.AsyncClient),
        authorizer=None,
        rate_limiter=mock.Mock(),
    )


async def test_proxy_data_parses_off_the_event_loop() -> None:
    """A slow (CPU-bound) data parse must not block concurrent coroutines on the event loop."""
    client = _make_client()
    message = DataMessage()
    parse_seconds = 0.3

    def _slow_convert(response_content, dsd) -> DataMessage:
        time.sleep(parse_seconds)  # simulates CPU-bound JSON parsing in the worker thread
        return message

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    # ResponseIO reads `.headers` and `.content` from the wrapped response.
    http_response = mock.Mock(headers={}, content=b"{}")

    with (
        mock.patch.object(
            client, "_perform_get", mock.AsyncMock(return_value=(http_response, mock.Mock()))
        ),
        mock.patch.object(
            client, "_convert_response", mock.Mock(return_value=mock.Mock(spec=requests.Response))
        ),
        mock.patch.object(client, "_convert_proxy_data", _slow_convert),
    ):
        ticker_task = asyncio.create_task(_ticker())
        try:
            result = await client._proxy_data(
                agency_id="IMF",
                resource_id="DF",
                version="1.0",
                key=None,
                params=None,
                dsd=None,
            )
        finally:
            ticker_task.cancel()

    assert result is message
    # With the parse on the event loop the ticker would not tick at all during the sleep.
    assert ticks >= 5
