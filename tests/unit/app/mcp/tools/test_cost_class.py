"""Cost-class resolution that selects a model-facing MCP tool's per-caller rate-limit allowance
(#600): dedicated tools declare their own, and the LangChain catch-all maps by tool type so the
expensive/open-world tools it serves are not left on the moderate default."""

from types import SimpleNamespace

import pytest

from statgpt.app.mcp.rate_limit import McpToolCostClass
from statgpt.app.mcp.tools.base import LangChainMcpTool
from statgpt.app.mcp.tools.data_query import DataQueryMcpTool
from statgpt.app.mcp.tools.glossary import AvailableTermsMcpTool, TermDefinitionsMcpTool
from statgpt.common.schemas import ToolTypes


def _config(tool_type: ToolTypes) -> SimpleNamespace:
    # get_cost_class only reads `tool_config.type`, so a stand-in with that attribute suffices.
    return SimpleNamespace(type=tool_type)


def test_data_query_is_expensive():
    assert (
        DataQueryMcpTool.get_cost_class(_config(ToolTypes.DATA_QUERY)) is McpToolCostClass.EXPENSIVE
    )


def test_glossary_tools_are_cheap():
    assert (
        AvailableTermsMcpTool.get_cost_class(_config(ToolTypes.AVAILABLE_TERMS))
        is McpToolCostClass.CHEAP
    )
    assert (
        TermDefinitionsMcpTool.get_cost_class(_config(ToolTypes.TERM_DEFINITIONS))
        is McpToolCostClass.CHEAP
    )


@pytest.mark.parametrize(
    "tool_type",
    [
        ToolTypes.WEB_SEARCH,
        ToolTypes.WEB_SEARCH_AGENT,
        ToolTypes.DEEP_RESEARCH,
        ToolTypes.FILE_RAG,
    ],
)
def test_catch_all_maps_expensive_tool_types(tool_type: ToolTypes):
    assert LangChainMcpTool.get_cost_class(_config(tool_type)) is McpToolCostClass.EXPENSIVE


@pytest.mark.parametrize(
    "tool_type",
    [
        ToolTypes.AVAILABLE_DATASETS,
        ToolTypes.AVAILABLE_PUBLICATIONS,
        ToolTypes.DATASETS_METADATA,
        ToolTypes.PLAIN_CONTENT,
    ],
)
def test_catch_all_defaults_to_moderate(tool_type: ToolTypes):
    assert LangChainMcpTool.get_cost_class(_config(tool_type)) is McpToolCostClass.MODERATE
