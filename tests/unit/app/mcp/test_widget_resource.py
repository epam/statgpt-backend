from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from statgpt.app.mcp import widget_resource as wr
from statgpt.app.mcp.provider import ChannelToolProvider
from statgpt.app.mcp.widget_resource import WidgetResource, WidgetResourceError
from statgpt.common.schemas import ProxiedResourceConfig
from statgpt.common.utils.media_types import MediaTypes

_URI = "ui://statgpt/data-widget.html"


def _config(**kwargs) -> ProxiedResourceConfig:
    base = {
        "uri": _URI,
        "origin": "https://widget.example",
        "html_url": "https://widget-internal.svc/index.html",
    }
    base.update(kwargs)
    return ProxiedResourceConfig.model_validate(base)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clear_caches():
    wr._caches.clear()
    yield
    wr._caches.clear()


def _patch_client(monkeypatch, get) -> None:
    # Seed the managed client's lazy slot with a fake so the `client` property returns it.
    monkeypatch.setattr(wr.widget_http_client, "_client", SimpleNamespace(get=get))


class TestFromConfig:
    def test_sets_meta_mime_and_uri(self):
        resource = WidgetResource.from_config(
            _config(origin="https://widget.example/", mime_type=MediaTypes.HTML_MCP_APP)
        )
        assert resource.mime_type == MediaTypes.HTML_MCP_APP
        assert resource.meta == {"ui": {"csp": {"resourceDomains": ["https://widget.example"]}}}
        assert str(resource.uri) == _URI


class TestRead:
    async def test_fetches_and_caches(self, monkeypatch):
        calls: list[str] = []

        async def fake_get(url: str, **kwargs) -> _FakeResponse:
            calls.append(url)
            assert kwargs.get("follow_redirects") is True
            return _FakeResponse("<html>hi</html>")

        _patch_client(monkeypatch, fake_get)
        resource = WidgetResource.from_config(_config())

        assert await resource.read() == "<html>hi</html>"
        # Second read within TTL is served from cache: no new fetch.
        assert await resource.read() == "<html>hi</html>"
        assert calls == ["https://widget-internal.svc/index.html"]

    async def test_raises_on_upstream_error(self, monkeypatch):
        async def fake_get(url: str, **kwargs):
            raise httpx.ConnectError("down")

        _patch_client(monkeypatch, fake_get)
        resource = WidgetResource.from_config(_config())

        with pytest.raises(WidgetResourceError):
            await resource.read()

    async def test_failure_is_not_cached(self, monkeypatch):
        outcomes: list = [httpx.ConnectError("down"), _FakeResponse("<html>ok</html>")]

        async def fake_get(url: str, **kwargs) -> _FakeResponse:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        _patch_client(monkeypatch, fake_get)
        resource = WidgetResource.from_config(_config())

        with pytest.raises(WidgetResourceError):
            await resource.read()
        # The failed load was not cached, so the next read retries and succeeds.
        assert await resource.read() == "<html>ok</html>"


def _build_provider(channel_config, monkeypatch) -> ChannelToolProvider:
    provider = ChannelToolProvider()
    channel_service = SimpleNamespace(channel_config=channel_config)
    monkeypatch.setattr("statgpt.app.mcp.provider.get_http_request", lambda: None)
    monkeypatch.setattr(
        provider, "_resolve_context", AsyncMock(return_value=(SimpleNamespace(), channel_service))
    )
    return provider


class TestProviderResources:
    async def test_list_resources_returns_widgets(self, monkeypatch):
        channel_config = SimpleNamespace(mcp=SimpleNamespace(resources=[_config()]))
        provider = _build_provider(channel_config, monkeypatch)

        resources = await provider._list_resources()

        assert len(resources) == 1
        assert isinstance(resources[0], WidgetResource)

    async def test_list_resources_empty_when_none_configured(self, monkeypatch):
        channel_config = SimpleNamespace(mcp=SimpleNamespace(resources=[]))
        provider = _build_provider(channel_config, monkeypatch)

        assert await provider._list_resources() == []

    async def test_get_resource_matches_uri(self, monkeypatch):
        channel_config = SimpleNamespace(mcp=SimpleNamespace(resources=[_config()]))
        provider = _build_provider(channel_config, monkeypatch)

        assert isinstance(await provider._get_resource(_URI), WidgetResource)
        assert await provider._get_resource("ui://statgpt/other.html") is None

    async def test_list_resources_empty_when_context_unresolved(self, monkeypatch):
        provider = ChannelToolProvider()
        monkeypatch.setattr("statgpt.app.mcp.provider.get_http_request", lambda: None)
        monkeypatch.setattr(
            provider, "_resolve_context", AsyncMock(side_effect=ValueError("no deployment"))
        )

        assert await provider._list_resources() == []
