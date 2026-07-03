import json
from types import SimpleNamespace

import pytest

from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.supreme_agent import ToolCaller
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.schemas.tools import AvailableDatasetsTool, SdmxQueryAppTool


@pytest.fixture
def channel_config() -> ChannelConfig:
    """A channel with one agent-facing tool and one `mcp_only` tool (SDMX query app,
    which is pinned to `mcp_only=True`)."""
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT",
            domain="official statistics",
            terminology_domain="official statistics",
        ),
        available_datasets=AvailableDatasetsTool(
            name="available_datasets",
            description="List available datasets.",
        ),
        sdmx_query_app=SdmxQueryAppTool(
            name="sdmx_query_app",
            description="Forward an SDMX request.",
            details=SdmxQueryAppDetails(base_url_raw="https://sdmx.example.org/api"),
        ),
    )


def test_agent_tools_property_excludes_mcp_only(channel_config: ChannelConfig):
    all_names = {tool.name for tool in channel_config.tools}
    agent_names = {tool.name for tool in channel_config.agent_tools}

    assert all_names == {"available_datasets", "sdmx_query_app"}
    assert agent_names == {"available_datasets"}


def test_get_tools_from_config_excludes_mcp_only(channel_config: ChannelConfig, monkeypatch):
    # Build cheap stand-ins instead of real tool objects: we only assert which configs
    # `get_tools_from_config` forwards, not how each tool is constructed.
    monkeypatch.setattr(
        "statgpt.app.chains.supreme_agent.StatGptTool",
        SimpleNamespace(
            from_config=lambda tool_cfg, channel_config: SimpleNamespace(name=tool_cfg.name)
        ),
    )

    tools = ToolCaller.get_tools_from_config(channel_config)

    assert {tool.name for tool in tools} == {"available_datasets"}


def test_out_of_scope_checker_excludes_mcp_only(channel_config: ChannelConfig):
    checker = OutOfScopeChecker(channel_config)

    description = json.loads(checker._get_tool_description())

    assert "available_datasets" in description
    assert "sdmx_query_app" not in description
