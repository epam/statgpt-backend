import pytest
from pydantic import ValidationError

from statgpt.common.schemas import McpConfig, ProxiedResourceConfig
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.enums import McpResourceTypes
from statgpt.common.schemas.tools import AvailableDatasetsTool
from statgpt.common.utils.media_types import MediaTypes

_URI = "ui://statgpt/data-widget.html"


def _proxied(**kwargs) -> ProxiedResourceConfig:
    base = {
        "uri": _URI,
        "origin": "https://widget.example",
        "html_url": "https://widget-internal.svc/index.html",
    }
    base.update(kwargs)
    return ProxiedResourceConfig.model_validate(base)


class TestProxiedResourceConfig:
    def test_defaults(self):
        cfg = _proxied()
        assert cfg.type is McpResourceTypes.PROXIED
        assert cfg.cache_ttl_seconds == 60
        assert cfg.mime_type == MediaTypes.HTML_MCP_APP
        assert cfg.get_origin() == "https://widget.example"
        assert cfg.get_html_url() == "https://widget-internal.svc/index.html"

    def test_uri_must_use_ui_scheme(self):
        with pytest.raises(ValidationError, match="ui://"):
            _proxied(uri="https://widget.example/index.html")

    @pytest.mark.parametrize("field", ["origin", "html_url"])
    def test_url_must_be_http(self, field: str):
        with pytest.raises(ValidationError, match="http"):
            _proxied(**{field: "ftp://nope"})

    def test_env_interpolation(self, monkeypatch):
        monkeypatch.setenv("WIDGET_HTML_URL", "https://from-env.svc/index.html")
        cfg = _proxied(html_url="$env:{WIDGET_HTML_URL}")
        assert cfg.get_html_url() == "https://from-env.svc/index.html"

    def test_origin_trailing_slash_trimmed(self):
        assert _proxied(origin="https://widget.example/").get_origin() == "https://widget.example"

    @pytest.mark.parametrize(
        "origin",
        [
            "https://widget.example/app",  # path
            "https://widget.example?x=1",  # query
            "https://widget.example#frag",  # fragment
        ],
    )
    def test_origin_must_be_bare(self, origin: str):
        with pytest.raises(ValidationError, match="bare origin"):
            _proxied(origin=origin)

    def test_camel_alias_accepted(self):
        cfg = ProxiedResourceConfig.model_validate(
            {"uri": _URI, "origin": "https://w.example", "htmlUrl": "https://w-internal.svc/i.html"}
        )
        assert cfg.get_html_url() == "https://w-internal.svc/i.html"


class TestMcpConfig:
    def test_empty_by_default(self):
        assert McpConfig().resources == []

    def test_duplicate_uri_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            McpConfig(resources=[_proxied(), _proxied()])

    def test_resource_uris_property(self):
        assert McpConfig(resources=[_proxied()]).resource_uris == {_URI}


def _channel(**kwargs) -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT",
            domain="official statistics",
            terminology_domain="official statistics",
        ),
        **kwargs,
    )


class TestChannelMcpBinding:
    def test_binding_to_declared_resource_ok(self):
        channel = _channel(
            mcp=McpConfig(resources=[_proxied()]),
            available_datasets=AvailableDatasetsTool(
                name="data_query", description="Query.", mcp_app_resource_uri=_URI
            ),
        )
        assert channel.available_datasets.mcp_app_resource_uri == _URI

    def test_binding_to_undeclared_resource_rejected(self):
        with pytest.raises(ValidationError, match="not declared"):
            _channel(
                available_datasets=AvailableDatasetsTool(
                    name="data_query",
                    description="Query.",
                    mcp_app_resource_uri="ui://statgpt/missing.html",
                ),
            )


class TestMcpMetaResourceUri:
    def test_resource_uri_only(self):
        tool = AvailableDatasetsTool(name="t", description="d", mcp_app_resource_uri=_URI)
        assert tool.mcp_meta == {"ui": {"resourceUri": _URI}}

    def test_merges_visibility_and_resource_uri(self):
        tool = AvailableDatasetsTool(
            name="t", description="d", mcp_visibility=["app"], mcp_app_resource_uri=_URI
        )
        assert tool.mcp_meta == {"ui": {"visibility": ["app"], "resourceUri": _URI}}
