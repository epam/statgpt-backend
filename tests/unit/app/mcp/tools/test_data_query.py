import json
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
from statgpt.common.data.base import DataResponseStatus
from statgpt.common.schemas.data_query_tool import (
    DataQueryMcpMeta,
    DataQueryMcpResources,
    McpResource,
    ToggleableConfig,
)
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus
from statgpt.common.schemas.query import JsonQueryMetadata, JsonQueryWithMetadata
from statgpt.common.schemas.tools import DataQueryTool

_WIDGET_URI = "ui://statgpt/data-widget.html"


def _tool_config(
    mcp_app_resource_uri: str | None = _WIDGET_URI,
    client_meta: bool = True,
    **mcp_resources,
) -> DataQueryTool:
    config = DataQueryTool(
        name="data_query",
        description="Query data",
        mcp_app_resource_uri=mcp_app_resource_uri,
    )
    config.details.mcp_meta = DataQueryMcpMeta(
        client=ToggleableConfig(enabled_str=str(client_meta))
    )
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
        time_period=("2020", "2024"),
        url_query="https://explorer.example/query",
        get_display_series_count=lambda: 2,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
    )


def _outcome(
    response: str = "answer",
    data_responses: dict | None = None,
    status: DataQueryStatus = DataQueryStatus.DATA_AVAILABLE,
    discovery: DiscoveryDatasetsOutcome | None = None,
    message: str | None = None,
) -> DataQueryOutcome:
    # Bypass pydantic validation: the MCP tool only reads data_responses, state.status, mcp_payload
    # and discovery off the outcome.
    return DataQueryOutcome.model_construct(
        response=response,
        data_responses=data_responses or {},
        state=SimpleNamespace(status=status, dimension_id_to_name={}),
        mcp_payload=DataQueryMcpPayload(message=message),
        discovery=discovery,
    )


async def test_data_available_returns_structured_content_its_json_text_and_csv():
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1, 2]}))})

    tool_result = await _build(outcome).run({"query": "cpi"})

    structured = tool_result.structured_content
    assert structured is not None
    # The rendered response is not sent: a client reading only `content` gets the same payload.
    first = tool_result.content[0]
    assert isinstance(first, TextContent)
    assert json.loads(first.text) == structured
    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert [r.resource.mimeType for r in resources] == ["text/csv"]
    assert structured["status"] == DataQueryStatus.DATA_AVAILABLE
    assert "version" not in structured
    assert [q["datasetUrn"] for q in structured["queries"]] == ["IMF:CPI(1.0.0)"]
    assert structured["queries"][0]["executed"] is True
    # Null fields are dropped from what the model reads.
    assert "missingDimensions" not in structured


async def test_data_available_carries_both_meta_audiences():
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1, 2]}))})

    tool_result = await _build(outcome).run({"query": "cpi"})

    meta = tool_result.meta
    assert meta is not None
    assert set(meta) == {"statgpt.dialx.ai/mcp-app", "statgpt.dialx.ai/client"}
    mcp_app = meta["statgpt.dialx.ai/mcp-app"]
    assert mcp_app["status"] == DataQueryStatus.DATA_AVAILABLE
    assert mcp_app["version"] == 3
    assert mcp_app["tools"] == {"sdmxProxy": "sdmx_query_app"}
    assert "import sdmx" in mcp_app["pythonCode"]
    assert [q["urn"] for q in mcp_app["queries"]] == ["IMF:CPI(1.0.0)"]
    client = meta["statgpt.dialx.ai/client"]
    assert client["queries"][0]["dataExplorerUrl"] == "https://explorer.example/query"
    assert client["queries"][0]["resourceUris"] == [
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{client['queries'][0]['queryId']}.csv"
    ]


async def test_no_data_reports_the_status_and_the_message():
    outcome = _outcome(
        response="No relevant data found.",
        status=DataQueryStatus.NO_DATA,
        message="No relevant data found.",
    )

    tool_result = await _build(outcome).run({"query": "cpi"})

    structured = tool_result.structured_content
    assert structured is not None
    assert structured["queries"] == []
    assert structured["status"] == DataQueryStatus.NO_DATA
    assert structured["message"] == "No relevant data found."
    assert tool_result.meta is not None
    for payload in tool_result.meta.values():
        assert payload["status"] == DataQueryStatus.NO_DATA
    # Of the `_meta` payloads, only the widget's carries the message.
    assert tool_result.meta["statgpt.dialx.ai/mcp-app"]["message"] == "No relevant data found."
    assert "message" not in tool_result.meta["statgpt.dialx.ai/client"]


async def test_markdown_resource_is_added_when_configured():
    outcome = _outcome(
        data_responses={"ds1": _data_response(pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]}))}
    )

    tool_result = await _build(outcome, _tool_config(csv=False, markdown_table=True)).run(
        {"query": "cpi"}
    )

    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert [r.resource.mimeType for r in resources] == ["text/markdown"]


async def test_mcp_app_meta_omits_sdmx_proxy_when_unconfigured():
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1]}))})

    tool_result = await _build(outcome, sdmx_query_app=None).run({"query": "cpi"})

    assert tool_result.meta is not None
    assert tool_result.meta["statgpt.dialx.ai/mcp-app"]["tools"] == {"sdmxProxy": None}


async def test_mcp_app_meta_is_omitted_when_the_tool_binds_no_widget():
    # Nothing renders the widget payload, so it is not built at all.
    config = _tool_config(mcp_app_resource_uri=None)
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1]}))})

    tool_result = await _build(outcome, config).run({"query": "cpi"})

    assert tool_result.meta is not None
    assert set(tool_result.meta) == {"statgpt.dialx.ai/client"}


async def test_meta_is_omitted_when_no_audience_has_a_reader():
    config = _tool_config(mcp_app_resource_uri=None, client_meta=False)
    outcome = _outcome(data_responses={"ds1": _data_response(pd.DataFrame({"x": [1]}))})

    tool_result = await _build(outcome, config).run({"query": "cpi"})

    assert tool_result.meta is None
    assert tool_result.structured_content is not None


async def test_discovery_block_is_a_content_block_of_its_own():
    # It comes from a lookup beside the query, so it is not part of the structured content.
    discovery = DiscoveryDatasetsOutcome(
        rendered="### Datasets\n\n- Alpha",
        eval_attachment=DiscoveryDatasetsEvalAttachment(query="gdp", rendered="### Datasets"),
    )

    tool_result = await _build(_outcome(discovery=discovery)).run({"query": "gdp"})

    texts = [c.text for c in tool_result.content if isinstance(c, TextContent)]
    assert texts == [json.dumps(tool_result.structured_content), "### Datasets\n\n- Alpha"]


async def test_the_runner_receives_the_validated_query():
    tool = _build(_outcome())

    await tool.run({"query": "cpi in France"})

    _, query = tool._runner.run.call_args.args  # type: ignore[attr-defined]
    assert query == "cpi in France"
