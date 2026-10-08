"""Tests for the StatGPT SDMX proxy client data path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import requests
from sdmx.message import DataMessage

from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import AsyncStatGptSdmxProxyClient

DATA_CONTENT_TYPE = "application/vnd.sdmx.data+json;version=2.0.0"


def _client(response: httpx.Response) -> AsyncStatGptSdmxProxyClient:
    sync_client = SimpleNamespace(source=SimpleNamespace(url="https://proxy.invalid"))
    client = AsyncStatGptSdmxProxyClient(sync_client, httpx.AsyncClient(), None, None)  # type: ignore[arg-type]
    req = requests.Request(method="GET", url="https://proxy.invalid/data").prepare()
    client._perform_get = AsyncMock(return_value=(response, req))  # type: ignore[method-assign]
    return client


@pytest.mark.parametrize("body", [b"", b"  \n"])
async def test_proxy_data_returns_empty_message_for_empty_body(body: bytes) -> None:
    response = httpx.Response(200, headers={"content-type": DATA_CONTENT_TYPE}, content=body)
    client = _client(response)

    msg = await client._proxy_data(
        agency_id="IMF.STA", resource_id="MFS_FMP", version="3.0.0", key=None, params=None, dsd=None
    )

    assert isinstance(msg, DataMessage)
    assert msg.data == []
