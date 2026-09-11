"""The cross-cutting behaviour every MCP tool shares, exercised through the LangChain-backed default:
argument validation, the input guardrail, error mapping, and the tool-type registry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from mcp.types import TextContent

from statgpt.app.chains.out_of_scope_checker import OutOfScopeCheckerResponse
from statgpt.app.chains.tools import StatGptTool, ToolUpstreamError
from statgpt.app.mcp.tools import LangChainMcpTool, StatGptMcpTool, mcp_tool_class_for
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.tools import (
    AvailablePublicationsTool,
    BaseToolConfig,
    DatasetsMetadataTool,
)

# The tools with an MCP interface of their own; every other type is served by the default wrapper.
SPLIT_TOOL_TYPES = {
    ToolTypes.DATA_QUERY,
    ToolTypes.AVAILABLE_TERMS,
    ToolTypes.TERM_DEFINITIONS,
    ToolTypes.AVAILABLE_DATASETS,
    ToolTypes.DATASET_STRUCTURE,
    ToolTypes.SDMX_QUERY_APP,
    ToolTypes.DATASETS_METADATA_APP,
}


def _channel_config(**overrides) -> SimpleNamespace:
    # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
    attrs: dict = {"mcp": SimpleNamespace(tool_name_prefix=""), "out_of_scope": None}
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _fake_langchain_tool(monkeypatch, *, result=None, error: Exception | None = None):
    ainvoke = AsyncMock(return_value=result, side_effect=error)
    fake = SimpleNamespace(name="fake_tool", ainvoke=ainvoke)
    monkeypatch.setattr(
        "statgpt.app.mcp.tools.base.StatGptTool",
        SimpleNamespace(
            from_config=lambda tool_config, channel_config: fake,
            implementation_for=StatGptTool.implementation_for,
        ),
    )
    return fake


def _build(
    tool_config: BaseToolConfig | None = None, channel_config=None, auth_context=None
) -> StatGptMcpTool:
    return StatGptMcpTool.from_config(
        tool_config or AvailablePublicationsTool(name="fake_tool", description="Publications."),
        channel_config or _channel_config(),
        inputs={},
        auth_context=auth_context or SimpleNamespace(),
    )


# ~~~~~~~~~~~~~ registry ~~~~~~~~~~~~~


def test_registry_holds_exactly_the_split_tools():
    registered = {t for t in ToolTypes if mcp_tool_class_for(t) is not LangChainMcpTool}
    assert registered == SPLIT_TOOL_TYPES


def test_unregistered_tool_types_fall_back_to_the_langchain_wrapper():
    assert mcp_tool_class_for(ToolTypes.WEB_SEARCH) is LangChainMcpTool
    assert mcp_tool_class_for(ToolTypes.AVAILABLE_PUBLICATIONS) is LangChainMcpTool


def test_default_annotations_are_read_only_and_closed_world(monkeypatch):
    _fake_langchain_tool(monkeypatch)
    tool = _build()
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False


@pytest.mark.parametrize("tool_type", [ToolTypes.WEB_SEARCH, ToolTypes.WEB_SEARCH_AGENT])
def test_web_search_tools_are_flagged_open_world(tool_type: ToolTypes):
    annotations = LangChainMcpTool.get_annotations(SimpleNamespace(type=tool_type))  # type: ignore[arg-type]
    assert annotations.readOnlyHint is True
    assert annotations.openWorldHint is True


def test_langchain_backed_tool_declares_no_output_schema(monkeypatch):
    _fake_langchain_tool(monkeypatch)
    assert _build().output_schema is None


# ~~~~~~~~~~~~~ running the LangChain tool ~~~~~~~~~~~~~


async def test_text_result_becomes_a_single_text_block(monkeypatch):
    fake = _fake_langchain_tool(monkeypatch, result=SimpleNamespace(content="hello", artifact=None))

    tool_result = await _build().run({})

    assert tool_result.content == [TextContent(type="text", text="hello")]
    assert tool_result.structured_content is None
    tool_call = fake.ainvoke.call_args.args[0]
    assert tool_call["name"] == "fake_tool" and tool_call["type"] == "tool_call"
    assert tool_call["args"] == {"inputs": {}}


async def test_non_string_content_is_stringified(monkeypatch):
    _fake_langchain_tool(monkeypatch, result=SimpleNamespace(content=[{"a": 1}], artifact=None))

    tool_result = await _build().run({})

    assert tool_result.content == [TextContent(type="text", text="[{'a': 1}]")]


async def test_empty_text_yields_no_content_block(monkeypatch):
    _fake_langchain_tool(monkeypatch, result=SimpleNamespace(content="", artifact=None))

    assert (await _build().run({})).content == []


# ~~~~~~~~~~~~~ error mapping ~~~~~~~~~~~~~


async def test_tool_failure_raises_tool_error(monkeypatch):
    _fake_langchain_tool(monkeypatch, error=RuntimeError("boom"))

    with pytest.raises(ToolError, match="failed to execute"):
        await _build().run({})


async def test_upstream_error_surfaces_its_message(monkeypatch):
    _fake_langchain_tool(monkeypatch, error=ToolUpstreamError("The backend timed out."))

    with pytest.raises(ToolError, match="The backend timed out."):
        await _build().run({})


async def test_invalid_arguments_raise_tool_error_naming_the_field(monkeypatch):
    # The datasets-metadata tool requires a `query`; leaving it out is an argument-schema failure
    # that must surface a concise ToolError naming the field so the caller can correct the request.
    fake = _fake_langchain_tool(monkeypatch)

    with pytest.raises(ToolError, match="Invalid arguments") as exc_info:
        await _build(DatasetsMetadataTool(name="meta", description="Metadata.")).run({})

    assert "query" in str(exc_info.value)
    fake.ainvoke.assert_not_called()


async def test_invalid_arguments_are_rejected_before_the_guardrail_runs(monkeypatch):
    # Validation comes first: a malformed request must not cost a guardrail LLM call.
    _fake_langchain_tool(monkeypatch)
    guardrail = AsyncMock()
    monkeypatch.setattr("statgpt.app.mcp.tools.base.enforce_input_guardrail", guardrail)

    with pytest.raises(ToolError, match="Invalid arguments"):
        await _build(DatasetsMetadataTool(name="meta", description="Metadata.")).run({"query": 42})

    guardrail.assert_not_called()


# ~~~~~~~~~~~~~ guardrail ~~~~~~~~~~~~~


class _FakeChatModel(Runnable):
    """Stands in for the guardrail LLM: structured checker verdict + plain response."""

    def __init__(self, decision: OutOfScopeCheckerResponse, message: str):
        self._decision = decision
        self._message = message

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda _: self._decision)

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content=self._message)


def _guardrail_channel_config() -> SimpleNamespace:
    return _channel_config(
        out_of_scope=SimpleNamespace(
            domain="official statistics",
            use_general_topics_blacklist=False,
            custom_blacklist=None,
            llm_model_config=SimpleNamespace(),
        ),
        supreme_agent=SimpleNamespace(language_instructions=["Answer in English"]),
        agent_tools=[SimpleNamespace(name="meta", out_of_scope_description="Metadata.")],
    )


async def test_run_blocks_out_of_scope_query(monkeypatch):
    decision = OutOfScopeCheckerResponse(reasoning="off-domain weather request", out_of_scope=True)
    fake_model = _FakeChatModel(decision, "I can only help with official statistics.")
    monkeypatch.setattr(
        "statgpt.app.chains.out_of_scope_checker.get_chat_model",
        lambda api_key, model_config: fake_model,
    )
    monkeypatch.setattr(
        "statgpt.app.mcp.guardrails.dial_app_settings.skip_out_of_scope_check", False
    )
    fake = _fake_langchain_tool(monkeypatch)
    tool = _build(
        DatasetsMetadataTool(name="meta", description="Metadata."),
        channel_config=_guardrail_channel_config(),
        auth_context=SimpleNamespace(api_key="key"),
    )

    with pytest.raises(ToolError, match="I can only help with official statistics."):
        await tool.run({"query": "weather in London"})

    fake.ainvoke.assert_not_called()
