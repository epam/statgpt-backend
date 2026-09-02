from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from mcp.types import EmbeddedResource, TextContent
from pydantic import BaseModel, ValidationError

from statgpt.app.chains.out_of_scope_checker import OutOfScopeCheckerResponse
from statgpt.app.mcp.provider import ChannelToolProvider, _McpToolAdapter, _tool_app_config
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.service import ChannelDatasetsMetadataResponse
from statgpt.app.schemas.tool_artifact import (
    DataQueryArtifact,
    DatasetsMetadataAppArtifact,
    SdmxQueryAppArtifact,
)
from statgpt.app.schemas.tool_states import ToolMessageState
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.data_query_tool import DataQueryMcpResources, McpResource
from statgpt.common.schemas.query import JsonQueryMetadata, JsonQueryWithMetadata
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.schemas.tools import (
    AvailableDatasetsTool,
    BaseToolConfig,
    DataQueryTool,
    DatasetsMetadataAppTool,
    SdmxQueryAppTool,
)


def _data_query_config(**mcp_resources) -> DataQueryTool:
    config = DataQueryTool(name="fake_tool", description="Query data")
    if mcp_resources:
        config.details.mcp_resources = DataQueryMcpResources(
            **{key: McpResource(enabled_str=str(value)) for key, value in mcp_resources.items()}
        )
    return config


def _build_adapter(
    result,
    sdmx_query_app=SimpleNamespace(name="sdmx_query_app"),
    tool_config: BaseToolConfig | None = None,
) -> _McpToolAdapter:
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(return_value=result))
    return _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
        channel_config=SimpleNamespace(  # type: ignore[arg-type]
            out_of_scope=None, sdmx_query_app=sdmx_query_app
        ),
        tool_config=tool_config or _data_query_config(),
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
    )


def _json_query(urn: str) -> JsonQueryWithMetadata:
    return JsonQueryWithMetadata(
        urn=urn,
        filters=[],
        metadata=JsonQueryMetadata(
            country_dimension="REF_AREA",
            indicator_dimensions=["INDICATOR"],
            time_period_dimension="TIME_PERIOD",
        ),
    )


def _state(status: DataQueryStatus = DataQueryStatus.DATA_AVAILABLE) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def _data_response(df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        resource_path="IMF:CPI(1.0.0)",
        dataset_name="CPI [IMF:CPI]",
        visual_dataframe=df,
        csv_dataframe=df,
        component_names={},
        is_empty=df.empty,
        created_at=datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc),
        json_query=_json_query("IMF:CPI(1.0.0)"),
    )


async def test_data_query_artifact_adds_csv_resources():
    response = _data_response(pd.DataFrame({"x": [1, 2]}))
    artifact = DataQueryArtifact.model_construct(data_responses={"ds1": response}, state=_state())
    result = SimpleNamespace(content="answer", artifact=artifact)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "answer"
    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert len(resources) == 1
    assert resources[0].resource.mimeType == "text/csv"
    structured = tool_result.structured_content
    assert structured is not None
    assert structured["status"] == DataQueryStatus.DATA_AVAILABLE
    assert [q["urn"] for q in structured["queries"]] == ["IMF:CPI(1.0.0)"]
    assert structured["tools"] == {"sdmxProxy": "sdmx_query_app"}
    assert structured["version"] == 2
    assert "import sdmx" in structured["pythonCode"]


async def test_data_query_no_data_returns_status_and_message():
    artifact = DataQueryArtifact.model_construct(
        data_responses={}, state=_state(DataQueryStatus.NO_DATA)
    )
    adapter = _build_adapter(SimpleNamespace(content="No relevant data found.", artifact=artifact))

    tool_result = await adapter.run({})

    structured = tool_result.structured_content
    assert structured is not None
    assert structured["status"] == DataQueryStatus.NO_DATA
    assert structured["message"] == "No relevant data found."
    assert structured["queries"] == []


async def test_data_query_markdown_resource_is_added_when_configured():
    response = _data_response(pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]}))
    artifact = DataQueryArtifact.model_construct(data_responses={"ds1": response}, state=_state())
    adapter = _build_adapter(
        SimpleNamespace(content="answer", artifact=artifact),
        tool_config=_data_query_config(csv=False, markdown_table=True),
    )

    tool_result = await adapter.run({})

    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert [r.resource.mimeType for r in resources] == ["text/markdown"]


async def test_data_query_structured_content_omits_sdmx_proxy_when_unconfigured():
    response = _data_response(pd.DataFrame({"x": [1]}))
    artifact = DataQueryArtifact.model_construct(data_responses={"ds1": response}, state=_state())
    adapter = _build_adapter(
        SimpleNamespace(content="answer", artifact=artifact), sdmx_query_app=None
    )

    tool_result = await adapter.run({})

    assert tool_result.structured_content is not None
    assert tool_result.structured_content["tools"] == {"sdmxProxy": None}


async def test_sdmx_query_app_artifact_exposes_http_metadata():
    artifact = SdmxQueryAppArtifact.model_construct(
        status_code=200, content_type="application/json"
    )
    adapter = _build_adapter(SimpleNamespace(content="<xml/>", artifact=artifact))

    tool_result = await adapter.run({})

    assert tool_result.structured_content == {
        "statusCode": 200,
        "contentType": "application/json",
    }


async def test_datasets_metadata_app_artifact_exposes_payload():
    response = ChannelDatasetsMetadataResponse(
        deployment_id="dep", title="Channel", n_datasets=0, datasets=[]
    )
    artifact = DatasetsMetadataAppArtifact(
        state=ToolMessageState(type=ToolTypes.DATASETS_METADATA_APP),
        response=response,
    )
    adapter = _build_adapter(SimpleNamespace(content=response.model_dump_json(), artifact=artifact))

    tool_result = await adapter.run({})

    assert tool_result.structured_content == {
        "deployment_id": "dep",
        "title": "Channel",
        "n_datasets": 0,
        "datasets": [],
    }


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
        tool_config=_data_query_config(),
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
    )

    with pytest.raises(ToolError):
        await adapter.run({})


async def test_invalid_arguments_raise_tool_error():
    # Argument-schema validation failures surface a concise ToolError that names the
    # offending field so the caller can correct the request.
    class _Args(BaseModel):
        limit: int

    try:
        _Args(limit="not-an-int")
    except ValidationError as exc:
        validation_error = exc

    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(side_effect=validation_error))
    adapter = _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        channel_config=SimpleNamespace(out_of_scope=None),  # type: ignore[arg-type]
        tool_config=_data_query_config(),
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
    )

    with pytest.raises(ToolError, match="Invalid arguments") as exc_info:
        await adapter.run({"limit": "not-an-int"})

    assert "limit" in str(exc_info.value)


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
        tool_config=_data_query_config(),
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
        get_mcp_output_schema=lambda: None,
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


def test_create_mcp_tool_app_visibility(fake_statgpt_tool):
    tool_config = _tool_config(mcp_visibility=["app"])
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.meta == {"ui": {"visibility": ["app"]}}


def test_create_mcp_tool_model_visibility(fake_statgpt_tool):
    tool_config = _tool_config(mcp_visibility=["model"])
    channel_config = _channel_config(tool_config, prefix="statgpt__")

    mcp_tool = ChannelToolProvider()._create_mcp_tool(
        tool_config, channel_config, inputs={}, auth_context=SimpleNamespace()
    )

    assert mcp_tool.meta == {"ui": {"visibility": ["model"]}}


def test_create_mcp_tool_omits_meta_when_visibility_unset(fake_statgpt_tool):
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


def test_tool_app_config_none_when_nothing_set():
    assert _tool_app_config(_tool_config()) is None


@pytest.mark.parametrize("visibility", [["app"], ["model"], ["model", "app"]])
def test_tool_app_config_carries_visibility(visibility):
    tool_config = _tool_config(mcp_visibility=visibility)

    assert _tool_app_config(tool_config) == AppConfig(visibility=visibility)


def test_tool_app_config_carries_resource_uri():
    tool_config = _tool_config(mcp_app_resource_uri="ui://statgpt/data-widget.html")

    assert _tool_app_config(tool_config) == AppConfig(resource_uri="ui://statgpt/data-widget.html")


async def test_get_tool_serializes_ui_meta(fake_statgpt_tool, monkeypatch):
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
    assert _tool_app_config(tool_config) == AppConfig(visibility=["app"])


def test_sdmx_query_app_visibility_is_overridable():
    tool_config = SdmxQueryAppTool(
        name="sdmx",
        description="SDMX passthrough.",
        details=_sdmx_details(),
        mcp_visibility=["model", "app"],
    )

    assert _tool_app_config(tool_config) == AppConfig(visibility=["model", "app"])


def test_datasets_metadata_app_defaults_to_app_only():
    tool_config = DatasetsMetadataAppTool(
        name="datasets_metadata", description="Datasets metadata."
    )

    assert tool_config.mcp_only is True
    assert tool_config.mcp_visibility == ["app"]
    assert _tool_app_config(tool_config) == AppConfig(visibility=["app"])
