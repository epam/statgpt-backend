"""Tests for the StatGPT SDMX proxy configuration server client."""

from collections.abc import Callable

import httpx
import pytest

from statgpt.common.data.statgpt_sdmx_proxy import config_client
from statgpt.common.data.statgpt_sdmx_proxy.config_client import (
    ProxyConfigServerError,
    ProxyConfigValidationError,
    fetch_proxy_config,
    push_proxy_config,
)

_URL = "http://config-server:8060/statgpt/sdmx-proxy-config-server/api/v0/config"
_CONFIG = {"configs": [], "agencies": [{"name": "IMF"}], "structureFanOutEnabled": False}

Handler = Callable[[httpx.Request], httpx.Response]


class _StubHttpClient:
    """Stands in for the module's `ManagedHttpClient`, serving a mocked transport."""

    def __init__(self, handler: Handler) -> None:
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def mock_transport(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    """Route the module's shared client through a caller-provided request handler."""

    def install(handler: Handler) -> None:
        monkeypatch.setattr(
            config_client, "proxy_config_http_client", _StubHttpClient(handler), raising=True
        )

    return install


async def test_fetch_returns_the_stored_configuration(mock_transport) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CONFIG)

    mock_transport(handler)

    assert await fetch_proxy_config(_URL) == _CONFIG
    assert [(r.method, str(r.url)) for r in requests] == [("GET", _URL)]


async def test_fetch_returns_none_on_cold_start(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(404))

    assert await fetch_proxy_config(_URL) is None


async def test_fetch_raises_when_storage_is_unavailable(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(503))

    with pytest.raises(ProxyConfigServerError, match="503"):
        await fetch_proxy_config(_URL)


async def test_fetch_raises_when_the_server_is_unreachable(mock_transport) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_transport(handler)

    with pytest.raises(ProxyConfigServerError, match="connection refused"):
        await fetch_proxy_config(_URL)


async def test_fetch_rejects_an_unresolved_env_placeholder(mock_transport) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be sent")

    mock_transport(handler)

    with pytest.raises(ProxyConfigServerError, match="unresolved environment variable"):
        await fetch_proxy_config("$env:{SDMX_PROXY_CONFIG_SERVER_HOST}/api/v0/config")


async def test_push_sends_the_configuration_and_returns_what_was_stored(mock_transport) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CONFIG)

    mock_transport(handler)

    assert await push_proxy_config(_URL, _CONFIG) == _CONFIG
    assert [(r.method, str(r.url)) for r in requests] == [("POST", _URL)]
    assert requests[0].read() == httpx.Request("POST", _URL, json=_CONFIG).read()


@pytest.mark.parametrize("status", [400, 422])
async def test_push_reports_a_rejected_configuration(mock_transport, status: int) -> None:
    mock_transport(
        lambda request: httpx.Response(status, json={"message": "agency IMF has no registry"})
    )

    with pytest.raises(ProxyConfigValidationError, match="agency IMF has no registry"):
        await push_proxy_config(_URL, _CONFIG)


async def test_push_falls_back_to_the_raw_body_when_it_is_not_json(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(400, text="Bad Request"))

    with pytest.raises(ProxyConfigValidationError, match="Bad Request"):
        await push_proxy_config(_URL, _CONFIG)


async def test_push_raises_on_an_unexpected_status(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(500))

    with pytest.raises(ProxyConfigServerError, match="500"):
        await push_proxy_config(_URL, _CONFIG)
