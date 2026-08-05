"""Tests for the async SDMX client."""

import asyncio
import time
from unittest import mock

import httpx
import requests
from sdmx.message import Message

from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient


def _make_client() -> AsyncSdmxClient:
    return AsyncSdmxClient(
        sync_client=mock.Mock(),
        httpx_client=mock.Mock(spec=httpx.AsyncClient),
        authorizer=None,
        rate_limiter=mock.Mock(),
    )


async def test_fetch_parses_off_the_event_loop() -> None:
    """A slow (CPU-bound) parse must not block concurrent coroutines on the event loop."""
    client = _make_client()
    message = Message()
    parse_seconds = 0.3

    def _slow_parse(response, tofile=None, dsd=None) -> Message:
        time.sleep(parse_seconds)  # simulates CPU-bound XML parsing in the worker thread
        return message

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    with (
        mock.patch.object(client, "_perform_request", mock.AsyncMock(return_value=mock.Mock())),
        mock.patch.object(
            client, "_convert_response", mock.Mock(return_value=mock.Mock(spec=requests.Response))
        ),
        mock.patch.object(client, "_parse_response", _slow_parse),
    ):
        ticker_task = asyncio.create_task(_ticker())
        try:
            result = await client._fetch(mock.Mock())
        finally:
            ticker_task.cancel()

    assert result is message
    # With the parse on the event loop the ticker would not tick at all during the sleep.
    assert ticks >= 5
