from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.apps import AppConfig

from statgpt.app.mcp.provider import ChannelToolProvider
from statgpt.app.mcp.tools import StatGptMcpTool, tool_app_config
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.schemas.tools import (
    AvailableDatasetsTool,
    DatasetsMetadataAppTool,
    SdmxQueryAppTool,
)


def _tool_config(**kwargs) -> AvailableDatasetsTool:
    return AvailableDatasetsTool(name="query_data", description="Query data tool.", **kwargs)


def _channel_config(tool_config, prefix: str) -> SimpleNamespace:
    return SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=prefix), tools=[tool_config])


def _create_mcp_tool(tool_config, channel_config) -> StatGptMcpTool:
    return StatGptMcpTool.from_config(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )


def _build_provider(channel_config, monkeypatch) -> ChannelToolProvider:
    provider = ChannelToolProvider()
    channel_service = SimpleNamespace(channel_config=channel_config, deployment_id="dep")
    monkeypatch.setattr("statgpt.app.mcp.provider.get_http_request", lambda: None)
    monkeypatch.setattr(
        provider,
        "_resolve_context",
        AsyncMock(return_value=(SimpleNamespace(), channel_service)),
    )
    return provider


def test_create_mcp_tool_applies_prefix():
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.name == "statgpt__query_data"
    assert mcp_tool.description == "Query data tool."


def test_create_mcp_tool_empty_prefix_keeps_name():
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.name == "query_data"


def test_create_mcp_tool_uses_mcp_overrides():
    tool_config = _tool_config(mcp_name="data", mcp_description="MCP description.")
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.name == "statgpt__data"
    assert mcp_tool.description == "MCP description."


def test_create_mcp_tool_app_visibility():
    tool_config = _tool_config(mcp_visibility=["app"])
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.meta == {"ui": {"visibility": ["app"]}}


def test_create_mcp_tool_model_visibility():
    tool_config = _tool_config(mcp_visibility=["model"])
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.meta == {"ui": {"visibility": ["model"]}}


def test_create_mcp_tool_omits_meta_when_visibility_unset():
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = _create_mcp_tool(tool_config, channel_config)

    assert mcp_tool.meta is None


async def test_get_tool_resolves_prefixed_name(monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    mcp_tool = await provider._get_tool("statgpt__query_data")

    assert mcp_tool is not None
    assert mcp_tool.name == "statgpt__query_data"


async def test_get_tool_rejects_unprefixed_name_when_prefix_set(monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    assert await provider._get_tool("query_data") is None


async def test_get_tool_without_prefix_resolves_plain_name(monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="")
    provider = _build_provider(channel_config, monkeypatch)

    mcp_tool = await provider._get_tool("query_data")

    assert mcp_tool is not None
    assert mcp_tool.name == "query_data"


async def test_get_tool_matches_mcp_name_override(monkeypatch):
    tool_config = _tool_config(mcp_name="data")
    channel_config = _channel_config(tool_config, prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    assert await provider._get_tool("statgpt__data") is not None
    assert await provider._get_tool("statgpt__query_data") is None


async def test_list_tools_serves_every_channel_tool(monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    tools = await provider._list_tools()

    assert [tool.name for tool in tools] == ["statgpt__query_data"]


def test_effective_mcp_fields_fall_back_to_base_fields():
    tool_config = _tool_config()

    assert tool_config.effective_mcp_name == "query_data"
    assert tool_config.effective_mcp_description == "Query data tool."


def test_tool_app_config_none_when_nothing_set():
    assert tool_app_config(_tool_config()) is None


@pytest.mark.parametrize("visibility", [["app"], ["model"], ["model", "app"]])
def test_tool_app_config_carries_visibility(visibility):
    tool_config = _tool_config(mcp_visibility=visibility)

    assert tool_app_config(tool_config) == AppConfig(visibility=visibility)


def test_tool_app_config_carries_resource_uri():
    tool_config = _tool_config(mcp_app_resource_uri="ui://statgpt/data-widget.html")

    assert tool_app_config(tool_config) == AppConfig(resource_uri="ui://statgpt/data-widget.html")


async def test_get_tool_serializes_ui_meta(monkeypatch):
    tool_config = _tool_config(
        mcp_visibility=["app"], mcp_app_resource_uri="ui://statgpt/data-widget.html"
    )
    channel_config = _channel_config(tool_config, prefix="")
    provider = _build_provider(channel_config, monkeypatch)

    mcp_tool = await provider._get_tool("query_data")

    assert mcp_tool is not None
    assert mcp_tool.meta == {
        "ui": {"visibility": ["app"], "resourceUri": "ui://statgpt/data-widget.html"}
    }


def _sdmx_details() -> SdmxQueryAppDetails:
    return SdmxQueryAppDetails.model_validate({"base_url": "https://example.test"})


def test_sdmx_query_app_defaults_to_app_only():
    tool_config = SdmxQueryAppTool(
        name="sdmx", description="SDMX passthrough.", details=_sdmx_details()
    )

    assert tool_config.mcp_visibility == ["app"]
    assert tool_app_config(tool_config) == AppConfig(visibility=["app"])


def test_sdmx_query_app_visibility_is_overridable():
    tool_config = SdmxQueryAppTool(
        name="sdmx",
        description="SDMX passthrough.",
        details=_sdmx_details(),
        mcp_visibility=["model", "app"],
    )

    assert tool_app_config(tool_config) == AppConfig(visibility=["model", "app"])


def test_datasets_metadata_app_defaults_to_app_only():
    tool_config = DatasetsMetadataAppTool(
        name="datasets_metadata", description="Datasets metadata."
    )

    assert tool_config.mcp_only is True
    assert tool_config.mcp_visibility == ["app"]
    assert tool_app_config(tool_config) == AppConfig(visibility=["app"])
