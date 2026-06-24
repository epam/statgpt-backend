from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from mcp.types import EmbeddedResource, TextContent

from statgpt.app.chains.out_of_scope_checker import OutOfScopeCheckerResponse
from statgpt.app.mcp.provider import APP_ONLY_TOOL_META, ChannelToolProvider, _McpToolAdapter
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.schemas.tools import AvailableDatasetsTool


def _build_adapter(result) -> _McpToolAdapter:
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(return_value=result))
    return _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
        channel_config=SimpleNamespace(out_of_scope=None),  # type: ignore[arg-type]
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
    )


async def test_data_query_artifact_adds_csv_resources():
    df = pd.DataFrame({"x": [1, 2]})
    response = SimpleNamespace(
        resource_path="IMF:CPI(1.0.0)",
        visual_dataframe=df,
        csv_dataframe=df,
        created_at=datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc),
    )
    artifact = DataQueryArtifact.model_construct(data_responses={"ds1": response})
    result = SimpleNamespace(content="answer", artifact=artifact)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "answer"
    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert len(resources) == 1
    assert resources[0].resource.mimeType == "text/csv"


async def test_non_data_query_result_returns_text_only():
    result = SimpleNamespace(content="hello", artifact=None)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert len(tool_result.content) == 1
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "hello"


async def test_non_string_content_is_stringified():
    result = SimpleNamespace(content=[{"a": 1}], artifact=None)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert len(tool_result.content) == 1
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "[{'a': 1}]"


async def test_tool_failure_raises_tool_error():
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(side_effect=RuntimeError("boom")))
    adapter = _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        channel_config=SimpleNamespace(out_of_scope=None),  # type: ignore[arg-type]
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
    )

    with pytest.raises(ToolError):
        await adapter.run({})


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
    return SimpleNamespace(
        out_of_scope=SimpleNamespace(
            domain="official statistics",
            use_general_topics_blacklist=False,
            custom_blacklist=None,
            llm_model_config=SimpleNamespace(),
        ),
        supreme_agent=SimpleNamespace(language_instructions=["Answer in English"]),
        agent_tools=[SimpleNamespace(name="data_query", out_of_scope_description="Query data")],
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

    tool = SimpleNamespace(
        name="data_query",
        get_guardrail_input=lambda arguments: arguments.get("query"),
        ainvoke=AsyncMock(),
    )
    adapter = _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        channel_config=_guardrail_channel_config(),  # type: ignore[arg-type]
        auth_context=SimpleNamespace(api_key="key"),  # type: ignore[arg-type]
        name="data_query",
        parameters={},
    )

    with pytest.raises(ToolError, match="I can only help with official statistics."):
        await adapter.run({"query": "weather in London"})

    tool.ainvoke.assert_not_called()


def _tool_config(**kwargs) -> AvailableDatasetsTool:
    return AvailableDatasetsTool(name="query_data", description="Query data tool.", **kwargs)


def _channel_config(tool_config, prefix: str) -> SimpleNamespace:
    return SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=prefix), tools=[tool_config])


@pytest.fixture
def fake_statgpt_tool(monkeypatch) -> SimpleNamespace:
    langchain_tool = SimpleNamespace(
        name="query_data",
        get_public_args_schema=lambda: {},
        get_mcp_annotations=lambda: None,
    )
    monkeypatch.setattr(
        "statgpt.app.mcp.provider.StatGptTool",
        SimpleNamespace(from_config=lambda tool_config, channel_config: langchain_tool),
    )
    return langchain_tool


def _build_provider(channel_config, monkeypatch) -> ChannelToolProvider:
    provider = ChannelToolProvider()
    channel_service = SimpleNamespace(channel_config=channel_config)
    monkeypatch.setattr("statgpt.app.mcp.provider.get_http_request", lambda: None)
    monkeypatch.setattr(
        provider,
        "_resolve_context",
        AsyncMock(return_value=(SimpleNamespace(), channel_service)),
    )
    return provider


def test_create_mcp_tool_applies_prefix(fake_statgpt_tool):
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.name == "statgpt__query_data"
    assert mcp_tool.description == "Query data tool."


def test_create_mcp_tool_empty_prefix_keeps_name(fake_statgpt_tool):
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.name == "query_data"


def test_create_mcp_tool_uses_mcp_overrides(fake_statgpt_tool):
    tool_config = _tool_config(mcp_name="data", mcp_description="MCP description.")
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.name == "statgpt__data"
    assert mcp_tool.description == "MCP description."


def test_create_mcp_tool_marks_mcp_only_as_app_visibility(fake_statgpt_tool):
    tool_config = _tool_config(mcp_only=True)
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.meta == APP_ONLY_TOOL_META
    assert mcp_tool.meta == {"ui": {"visibility": ["app"]}}


def test_create_mcp_tool_omits_meta_for_agent_tool(fake_statgpt_tool):
    tool_config = _tool_config()
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.meta is None


async def test_get_tool_resolves_prefixed_name(fake_statgpt_tool, monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    mcp_tool = await provider._get_tool("statgpt__query_data")

    assert mcp_tool is not None
    assert mcp_tool.name == "statgpt__query_data"


async def test_get_tool_rejects_unprefixed_name_when_prefix_set(fake_statgpt_tool, monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    assert await provider._get_tool("query_data") is None


async def test_get_tool_without_prefix_resolves_plain_name(fake_statgpt_tool, monkeypatch):
    channel_config = _channel_config(_tool_config(), prefix="")
    provider = _build_provider(channel_config, monkeypatch)

    mcp_tool = await provider._get_tool("query_data")

    assert mcp_tool is not None
    assert mcp_tool.name == "query_data"


async def test_get_tool_matches_mcp_name_override(fake_statgpt_tool, monkeypatch):
    tool_config = _tool_config(mcp_name="data")
    channel_config = _channel_config(tool_config, prefix="statgpt__")
    provider = _build_provider(channel_config, monkeypatch)

    assert await provider._get_tool("statgpt__data") is not None
    assert await provider._get_tool("statgpt__query_data") is None


def test_effective_mcp_fields_fall_back_to_base_fields():
    tool_config = _tool_config()

    assert tool_config.effective_mcp_name == "query_data"
    assert tool_config.effective_mcp_description == "Query data tool."
