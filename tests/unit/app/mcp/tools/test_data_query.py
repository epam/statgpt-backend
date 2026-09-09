from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
from mcp.types import EmbeddedResource, TextContent

from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.app.mcp.tools.data_query import DataQueryMcpTool
from statgpt.app.schemas.data_query_outcome import DataQueryMcpPayload, DataQueryStatus
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryDatasetsEvalAttachment,
    DiscoveryDatasetsOutcome,
)
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.common.schemas.data_query_tool import DataQueryMcpResources, McpResource
from statgpt.common.schemas.query import JsonQueryMetadata, JsonQueryWithMetadata
from statgpt.common.schemas.tools import DataQueryTool


def _tool_config(**mcp_resources) -> DataQueryTool:
    config = DataQueryTool(name="data_query", description="Query data")
    if mcp_resources:
        config.details.mcp_resources = DataQueryMcpResources(
            **{key: McpResource(enabled_str=str(value)) for key, value in mcp_resources.items()}
        )
    return config


def _build(
    outcome: DataQueryOutcome,
    tool_config: DataQueryTool | None = None,
    sdmx_query_app=SimpleNamespace(name="sdmx_query_app"),
) -> DataQueryMcpTool:
    tool = StatGptMcpTool.from_config(
        tool_config or _tool_config(),
        # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
        SimpleNamespace(  # type: ignore[arg-type]
            mcp=SimpleNamespace(tool_name_prefix=""),
            out_of_scope=None,
            sdmx_query_app=sdmx_query_app,
        ),
        inputs={},
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert isinstance(tool, DataQueryMcpTool)
    tool._runner.run = AsyncMock(return_value=outcome)  # type: ignore[method-assign]
    return tool


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


def _outcome(
    response: str = "answer",
    data_responses: dict | None = None,
    status: DataQueryStatus = DataQueryStatus.DATA_AVAILABLE,
    discovery: DiscoveryDatasetsOutcome | None = None,
) -> DataQueryOutcome:
    # Bypass pydantic validation: the MCP tool only reads data_responses, state.status, mcp_payload
    # and discovery off the outcome.
    return DataQueryOutcome.model_construct(
        response=response,
        data_responses=data_responses or {},
        state=SimpleNamespace(status=status),
        mcp_payload=DataQueryMcpPayload(),
        discovery=discovery,
    )


async def test_data_available_returns_text_csv_and_structured_content():
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1, 2]}))})

    tool_result = await _build(outcome).run({"query": "cpi"})

    assert tool_result.content[0] == TextContent(type="text", text="answer")
    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert [r.resource.mimeType for r in resources] == ["text/csv"]
    structured = tool_result.structured_content
    assert structured is not None
    assert structured["status"] == DataQueryStatus.DATA_AVAILABLE
    assert [q["urn"] for q in structured["queries"]] == ["IMF:CPI(1.0.0)"]
    assert structured["tools"] == {"sdmxProxy": "sdmx_query_app"}
    assert structured["version"] == 2
    assert "import sdmx" in structured["pythonCode"]


async def test_no_data_returns_status_and_message():
    outcome = _outcome(response="No relevant data found.", status=DataQueryStatus.NO_DATA)

    tool_result = await _build(outcome).run({"query": "cpi"})

    structured = tool_result.structured_content
    assert structured is not None
    assert structured["status"] == DataQueryStatus.NO_DATA
    assert structured["message"] == "No relevant data found."
    assert structured["queries"] == []


async def test_markdown_resource_is_added_when_configured():
    outcome = _outcome(
        data_responses={"ds1": _data_response(pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]}))}
    )

    tool_result = await _build(outcome, _tool_config(csv=False, markdown_table=True)).run(
        {"query": "cpi"}
    )

    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert [r.resource.mimeType for r in resources] == ["text/markdown"]


async def test_structured_content_omits_sdmx_proxy_when_unconfigured():
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1]}))})

    tool_result = await _build(outcome, sdmx_query_app=None).run({"query": "cpi"})

    assert tool_result.structured_content is not None
    assert tool_result.structured_content["tools"] == {"sdmxProxy": None}


async def test_discovery_block_is_a_content_block_of_its_own():
    # Not folded into the response text: it would duplicate there and leave markdown in the
    # `message` field a client parses rather than reads.
    discovery = DiscoveryDatasetsOutcome(
        rendered="### Datasets\n\n- Alpha",
        eval_attachment=DiscoveryDatasetsEvalAttachment(query="gdp", rendered="### Datasets"),
    )

    tool_result = await _build(_outcome(discovery=discovery)).run({"query": "gdp"})

    texts = [c.text for c in tool_result.content if isinstance(c, TextContent)]
    assert texts == ["answer", "### Datasets\n\n- Alpha"]


async def test_the_runner_receives_the_validated_query():
    tool = _build(_outcome())

    await tool.run({"query": "cpi in France"})

    _, query = tool._runner.run.call_args.args  # type: ignore[attr-defined]
    assert query == "cpi in France"
