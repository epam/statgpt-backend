from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace

import pandas as pd

from statgpt.app.mcp.attachments import (
    data_query_outcome_to_meta,
    data_query_outcome_to_resources,
    data_query_outcome_to_structured_content,
)
from statgpt.app.schemas.data_query_outcome import (
    DataQueryMcpPayload,
    DataQueryStatus,
    DataSetChoice,
    DimensionValueInfo,
    InvalidPeriodInfo,
    MissingDimensionInfo,
    MissingDimensionsInfo,
    QueryDetails,
)
from statgpt.app.schemas.mcp import ExecutionResult
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.common.data.base import DataResponseStatus
from statgpt.common.schemas.data_query_tool import (
    DataQueryExplorerLink,
    DataQueryMcpMeta,
    DataQueryMcpResources,
    DataQueryMcpStructuredContent,
    McpResource,
    ToggleableConfig,
)
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus, ExplorerLinkPolicy
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)

_FIXED_TS = datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc)
_FIXED_TS_STR = _FIXED_TS.strftime("%Y%m%dT%H%M%SZ")


def _state(
    status: DataQueryStatus = DataQueryStatus.DATA_AVAILABLE,
    dimension_id_to_name: dict | None = None,
) -> SimpleNamespace:
    # The converters only read `status` and `dimension_id_to_name` off the outcome's state.
    return SimpleNamespace(status=status, dimension_id_to_name=dimension_id_to_name or {})


def _mcp_payload(
    *,
    constructed_queries: list | None = None,
    candidate_datasets: list | None = None,
    missing_dimensions: MissingDimensionsInfo | None = None,
    query_details: dict[str, QueryDetails] | None = None,
    message: str | None = None,
    executed_at: str | None = None,
) -> DataQueryMcpPayload:
    return DataQueryMcpPayload(
        constructed_queries=constructed_queries or [],
        candidate_datasets=candidate_datasets or [],
        missing_dimensions=missing_dimensions,
        query_details=query_details or {},
        message=message,
        executed_at=executed_at,
    )


def _make_outcome(
    data_responses: dict,
    state: SimpleNamespace | None = None,
    mcp_payload: DataQueryMcpPayload | None = None,
) -> DataQueryOutcome:
    # Bypass pydantic validation — the converter only reads data_responses, state and mcp_payload.
    return DataQueryOutcome.model_construct(
        data_responses=data_responses,
        state=state or _state(),
        mcp_payload=mcp_payload or _mcp_payload(),
    )


def _channel_config(sdmx_proxy_name: str | None = "sdmx_query_app") -> SimpleNamespace:
    # The converter only reads channel_config.sdmx_query_app(.name).
    sdmx_query_app = SimpleNamespace(name=sdmx_proxy_name) if sdmx_proxy_name is not None else None
    return SimpleNamespace(sdmx_query_app=sdmx_query_app)


def _json_query(urn: str) -> JsonQueryWithMetadata:
    return JsonQueryWithMetadata(
        urn=urn,
        filters=[
            JsonComponentQuery(
                component_code="REF_AREA", operator=JsonQueryOperator.IN, values=["FR", "DE"]
            ),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="REF_AREA",
            indicator_dimensions=["INDICATOR"],
            time_period_dimension="TIME_PERIOD",
            key_dimension_ids_in_dsd_order=["REF_AREA", "INDICATOR"],
        ),
        sdmx1_source="IMF_DATA",
    )


def _response(
    resource_path: str,
    df: pd.DataFrame,
    created_at: datetime = _FIXED_TS,
    json_query: JsonQueryWithMetadata | None = None,
    component_names: dict[str, str] | None = None,
    time_period: tuple[str, str] | None = ("2020", "2024"),
    url_query: str | None = "https://explorer.example/query",
    series_count: int = 3,
    request_status: DataRequestStatus = DataRequestStatus.SUCCESS,
    parsing_status: DataParsingStatus = DataParsingStatus.SUCCESS,
    reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        resource_path=resource_path,
        dataset_name=f"CPI [{resource_path}]",
        visual_dataframe=df,
        csv_dataframe=df,
        component_names=component_names or {},
        is_empty=df.empty,
        created_at=created_at,
        json_query=json_query,
        time_period=time_period,
        url_query=url_query,
        get_display_series_count=lambda: series_count,
        status=DataResponseStatus(
            request_status=request_status, parsing_status=parsing_status, reason=reason
        ),
    )


def _resources(outcome: DataQueryOutcome, csv: bool = True, markdown: bool = False):
    config = DataQueryMcpResources(
        csv=McpResource(enabled_str=str(csv)),
        markdown_table=McpResource(enabled_str=str(markdown)),
    )
    return data_query_outcome_to_resources(outcome, config)


def _markdown_text(outcome: DataQueryOutcome) -> str:
    resources = _resources(outcome, csv=False, markdown=True)
    assert len(resources) == 1
    return resources[0].resource.text


def test_single_dataset_produces_one_csv_resource():
    df = pd.DataFrame({"country": ["FR", "DE"], "value": [1, 2]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(2.0.0)", df)})

    resources = _resources(outcome)

    assert len(resources) == 1
    resource = resources[0].resource
    assert resource.mimeType == "text/csv"
    assert str(resource.uri) == f"statgpt://data_query/IMF%3ACPI%282.0.0%29/{_FIXED_TS_STR}.csv"
    parsed = pd.read_csv(StringIO(resource.text))
    assert list(parsed.columns) == ["country", "value"]
    assert parsed.shape == (2, 2)


def test_multiple_datasets_preserve_insertion_order():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df1),
            "ds2": _response("BIS:IR(2.1.0)", df2),
        }
    )

    resources = _resources(outcome)

    assert [str(r.resource.uri) for r in resources] == [
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.csv",
        f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.csv",
    ]


def test_each_response_uses_its_own_created_at():
    ts1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"a": [1]}), created_at=ts1),
            "ds2": _response("BIS:IR(2.1.0)", pd.DataFrame({"b": [2]}), created_at=ts2),
        }
    )

    resources = _resources(outcome)

    assert [str(r.resource.uri) for r in resources] == [
        "statgpt://data_query/IMF%3ACPI%281.0.0%29/20260101T000000Z.csv",
        "statgpt://data_query/BIS%3AIR%282.1.0%29/20260102T000000Z.csv",
    ]


def test_empty_dataframes_are_skipped():
    outcome = _make_outcome(
        {
            "empty": _response("IMF:EMPTY(1.0.0)", pd.DataFrame()),
            "ok": _response("IMF:OK(1.0.0)", pd.DataFrame({"x": [1]})),
        }
    )

    resources = _resources(outcome)

    assert len(resources) == 1
    assert (
        str(resources[0].resource.uri)
        == f"statgpt://data_query/IMF%3AOK%281.0.0%29/{_FIXED_TS_STR}.csv"
    )


def test_no_responses_returns_empty_list():
    outcome = _make_outcome({})

    assert _resources(outcome) == []


def test_csv_disabled_produces_no_resources():
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}))})

    assert _resources(outcome, csv=False) == []


def test_both_payloads_enabled_are_emitted_per_response():
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]})),
            "ds2": _response("BIS:IR(2.1.0)", pd.DataFrame({"y": [2]})),
        }
    )

    resources = _resources(outcome, csv=True, markdown=True)

    assert [(r.resource.mimeType, str(r.resource.uri)) for r in resources] == [
        ("text/csv", f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.csv"),
        ("text/markdown", f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.md"),
        ("text/csv", f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.csv"),
        ("text/markdown", f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.md"),
    ]


def test_markdown_resource_is_annotated_for_the_user():
    df = pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    resources = _resources(outcome, csv=False, markdown=True)

    assert len(resources) == 1
    assert resources[0].annotations is not None
    assert resources[0].annotations.audience == ["user"]
    assert resources[0].resource.mimeType == "text/markdown"
    assert str(resources[0].resource.uri).endswith(f"{_FIXED_TS_STR}.md")


def test_markdown_table_uses_display_names_and_drops_codes():
    df = pd.DataFrame(
        {
            "REF_AREA": ["FR", "DE"],
            "REF_AREA_Name": ["France", "Germany"],
            "INDICATOR": ["CPI", "CPI"],
            "INDICATOR_Name": ["Consumer price index", "Consumer price index"],
            "2024": [1.5, 2.5],
        }
    )
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)",
                df,
                component_names={"REF_AREA": "Reference area", "INDICATOR": "Indicator"},
            )
        }
    )

    text = _markdown_text(outcome)

    assert text.startswith("### CPI [IMF:CPI(1.0.0)]\n\n")
    header = text.splitlines()[2]
    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "Reference area",
        "Indicator",
        "2024",
    ]
    assert "France" in text
    assert "| FR " not in text


def test_markdown_table_keeps_ids_without_display_names():
    df = pd.DataFrame({"REF_AREA": ["FR"], "REF_AREA_Name": ["France"], "2024": [1.0]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    header = _markdown_text(outcome).splitlines()[2]

    assert [cell.strip() for cell in header.strip("|").split("|")] == ["REF_AREA", "2024"]


def test_markdown_table_keeps_id_when_display_name_collides():
    df = pd.DataFrame({"FREQ": ["A"], "REF_AREA": ["FR"], "2024": [1.0]})
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)",
                df,
                # Both components claim the same display name: the first one gets it.
                component_names={"FREQ": "Frequency", "REF_AREA": "Frequency"},
            )
        }
    )

    header = _markdown_text(outcome).splitlines()[2]

    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "Frequency",
        "REF_AREA",
        "2024",
    ]


def test_markdown_table_is_not_padded():
    df = pd.DataFrame({"REF_AREA": ["FR"], "REF_AREA_Name": ["France"], "2024": [1.5]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, component_names={"REF_AREA": "Reference area"})}
    )

    lines = _markdown_text(outcome).splitlines()

    assert lines[2] == "| Reference area | 2024 |"
    # Numeric columns are right-aligned, everything else left-aligned, at minimum rule width.
    assert lines[3] == "|:---|---:|"
    assert lines[4] == "| France | 1.5 |"


def test_markdown_table_escapes_pipes_in_values():
    df = pd.DataFrame({"INDICATOR": ["GDP | current prices"], "2024": [1.5]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    row = _markdown_text(outcome).splitlines()[4]

    assert row == "| GDP \\| current prices | 1.5 |"


def test_markdown_table_drops_trailing_zero_of_whole_floats():
    df = pd.DataFrame({"REF_AREA": ["FR", "DE", "IT"], "2024": [123.0, 4.5, 1e14]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    rows = _markdown_text(outcome).splitlines()[4:]

    assert rows == ["| FR | 123 |", "| DE | 4.5 |", "| IT | 100000000000000 |"]


def test_markdown_table_keeps_textual_values_verbatim():
    # A `.0` in a value that arrived as text may be part of a code, so it is left alone.
    df = pd.DataFrame({"VERSION": ["1.0"], "2024": ["7.0"]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert _markdown_text(outcome).splitlines()[4] == "| 1.0 | 7.0 |"


def test_markdown_table_preserves_full_float_precision():
    df = pd.DataFrame({"REF_AREA": ["FR"], "2024": [112.345678901234]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert "112.345678901234" in _markdown_text(outcome)


def test_markdown_table_renders_missing_values_as_na():
    df = pd.DataFrame({"REF_AREA": ["FR", "DE"], "2024": [1.5, None]})
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert "NA" in _markdown_text(outcome)


def test_markdown_table_skips_empty_responses():
    outcome = _make_outcome(
        {
            "empty": _response("IMF:EMPTY(1.0.0)", pd.DataFrame()),
            "ok": _response("IMF:OK(1.0.0)", pd.DataFrame({"x": [1]})),
        }
    )

    resources = _resources(outcome, csv=False, markdown=True)

    assert len(resources) == 1
    assert str(resources[0].resource.uri).startswith("statgpt://data_query/IMF%3AOK%281.0.0%29/")


def _app_query(urn: str) -> AppJsonQueryWithMetadata:
    return AppJsonQueryWithMetadata.from_common(_json_query(urn))


def _timed_query(urn: str, values: list[str], operator=JsonQueryOperator.BETWEEN):
    query = _json_query(urn)
    query.filters.append(
        JsonComponentQuery(component_code="TIME_PERIOD", operator=operator, values=values)
    )
    return query


def _tool_config(
    mcp_app: bool = True,
    client: bool = True,
    namespace: str = "statgpt.dialx.ai",
    csv: bool = True,
    markdown: bool = False,
    explorer_link: ExplorerLinkPolicy = ExplorerLinkPolicy.always,
    **structured_content_fields: bool,
) -> SimpleNamespace:
    # The converters only read `mcp_app_resource_uri` off the tool config, and `mcp_meta` /
    # `mcp_resources` / `mcp_structured_content` / `explorer_link` off its details.
    return SimpleNamespace(
        mcp_app_resource_uri="ui://statgpt/data-widget.html" if mcp_app else None,
        details=SimpleNamespace(
            mcp_meta=DataQueryMcpMeta(
                namespace=namespace, client=ToggleableConfig(enabled_str=str(client))
            ),
            mcp_resources=DataQueryMcpResources(
                csv=McpResource(enabled_str=str(csv)),
                markdown_table=McpResource(enabled_str=str(markdown)),
            ),
            mcp_structured_content=DataQueryMcpStructuredContent(**structured_content_fields),
            explorer_link=DataQueryExplorerLink(mcp=explorer_link),
        ),
    )


def _structured(outcome: DataQueryOutcome, **tool_config):
    return data_query_outcome_to_structured_content(outcome, _tool_config(**tool_config))


def _details(**fields) -> QueryDetails:
    return QueryDetails(
        dataset_urn=fields.pop("dataset_urn", "IMF:CPI(1.0.0)"),
        dataset_name=fields.pop("dataset_name", "Consumer prices"),
        **fields,
    )


# ~~~~~~~~~~~~~ structured content: what the model reads ~~~~~~~~~~~~~


def test_structured_content_data_available_describes_the_executed_query():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)",
                df,
                json_query=_timed_query("IMF:CPI(1.0.0)", ["2020-01-01", "2024-12-31"]),
                component_names={"REF_AREA": "Reference area"},
            )
        },
        state=_state(dimension_id_to_name={"ds1": {"REF_AREA": {"FR": "France"}}}),
    )

    data = _structured(outcome).model_dump(by_alias=True, exclude_none=True)

    # The python code, the companion tools and the version are client-facing: `_meta` carries them.
    assert set(data) == {"status", "queries", "candidateDatasets"}
    assert data["status"] == DataQueryStatus.DATA_AVAILABLE
    assert len(data["queries"]) == 1
    query = data["queries"][0]
    assert query["datasetUrn"] == "IMF:CPI(1.0.0)"
    assert query["datasetName"] == "CPI [IMF:CPI(1.0.0)]"
    assert query["executed"] is True
    assert query["queryId"].startswith("dq_")
    assert query["seriesCount"] == 3
    assert query["requestedPeriod"] == {
        "startPeriod": "2020-01-01",
        "endPeriod": "2024-12-31",
        "isDefault": False,
    }
    assert query["execution"] == {"result": ExecutionResult.DATA_RECEIVED}
    assert query["dataExplorerUrl"] == "https://explorer.example/query"
    assert query["factualPeriod"] == {"startPeriod": "2020", "endPeriod": "2024"}
    # The time period is reported once, as requestedPeriod - not as another filter.
    assert [f["dimensionId"] for f in query["filters"]] == ["REF_AREA"]
    assert query["filters"][0]["dimensionName"] == "Reference area"
    assert query["filters"][0]["operator"] == JsonQueryOperator.IN
    # The name of a value is reported when the pipeline resolved it, the id alone otherwise.
    assert query["filters"][0]["values"] == [{"id": "FR", "name": "France"}, {"id": "DE"}]
    assert query["filters"][0]["isIndicator"] is False
    assert query["filters"][0]["isDefault"] is False


def test_structured_content_reports_the_query_id_the_resources_use():
    # The id joins the query to its CSV/Markdown resources, so both must agree.
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    query_id = _structured(outcome).queries[0].query_id

    assert str(_resources(outcome)[0].resource.uri).endswith(f"/{query_id}.csv")


def test_query_id_is_stable_for_the_same_query_on_the_same_date():
    df = pd.DataFrame({"x": [1]})
    same_day = datetime(2026, 4, 20, 23, 59, 0, tzinfo=timezone.utc)
    next_day = datetime(2026, 4, 21, 0, 1, 0, tzinfo=timezone.utc)

    def query_id(created_at: datetime, urn: str = "IMF:CPI(1.0.0)") -> str:
        outcome = _make_outcome(
            {"ds1": _response(urn, df, created_at=created_at, json_query=_json_query(urn))}
        )
        return _structured(outcome).queries[0].query_id

    assert query_id(_FIXED_TS) == query_id(same_day)
    # The same query a month (or a day) later may well return different data.
    assert query_id(_FIXED_TS) != query_id(next_day)
    assert query_id(_FIXED_TS) != query_id(_FIXED_TS, urn="BIS:IR(2.1.0)")


def test_structured_content_reports_a_long_value_list_in_full():
    values = [f"C{i:02d}" for i in range(25)]
    query = _json_query("IMF:CPI(1.0.0)")
    query.filters = [
        JsonComponentQuery(component_code="INDICATOR", operator=JsonQueryOperator.IN, values=values)
    ]
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}), json_query=query)}
    )

    reported = _structured(outcome).queries[0].filters[0]

    assert reported.value_count == 25
    assert [value.id for value in reported.values] == values


def test_structured_content_preserves_insertion_order():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)")),
            "ds2": _response("BIS:IR(2.1.0)", df, json_query=_json_query("BIS:IR(2.1.0)")),
        }
    )

    structured = _structured(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)", "BIS:IR(2.1.0)"]


def test_structured_content_skips_responses_without_query():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "no_query": _response("IMF:NOQ(1.0.0)", df, json_query=None),
            "ok": _response("IMF:OK(1.0.0)", df, json_query=_json_query("IMF:OK(1.0.0)")),
        }
    )

    structured = _structured(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:OK(1.0.0)"]


def test_structured_content_executed_no_data_still_reports_the_queries():
    # The query ran and returned nothing: the model should still see what was asked.
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)", pd.DataFrame(), json_query=_json_query("IMF:CPI(1.0.0)")
            )
        },
        state=_state(DataQueryStatus.EXECUTED_NO_DATA),
    )

    structured = _structured(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert structured.queries[0].executed is True
    assert structured.queries[0].series_count is None


def test_structured_content_failed_reports_the_queries_that_errored():
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)", pd.DataFrame(), json_query=_json_query("IMF:CPI(1.0.0)")
            )
        },
        state=_state(DataQueryStatus.FAILED),
    )

    structured = _structured(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]


def test_structured_content_failed_without_responses_has_no_queries():
    # `failed` is also the default status, reached when the pipeline errored before executing
    # anything - there is nothing to report.
    outcome = _make_outcome({}, state=_state(DataQueryStatus.FAILED))

    assert _structured(outcome).queries == []


def test_structured_content_not_executed_flags_the_constructed_queries():
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.NOT_EXECUTED),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    structured = _structured(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert structured.queries[0].executed is False
    assert structured.queries[0].series_count is None
    # Nothing ran, so there is no response to read the display names off.
    assert structured.queries[0].dataset_name is None
    assert structured.queries[0].filters[0].dimension_name is None


def test_structured_content_dataset_selection_carries_candidates():
    candidates = [
        DataSetChoice(id="IMF:CPI", name="CPI", description="Prices", is_official=True),
        DataSetChoice(id="BIS:IR", name="Rates"),
    ]
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.DATASET_SELECTION_REQUIRED),
        mcp_payload=_mcp_payload(candidate_datasets=candidates),
    )

    data = _structured(outcome).model_dump(by_alias=True, exclude_none=True)

    # The official mark is reported only for channels that enable it.
    assert data["candidateDatasets"] == [
        {"id": "IMF:CPI", "name": "CPI"},
        {"id": "BIS:IR", "name": "Rates"},
    ]
    assert data["queries"] == []


def test_structured_content_dataset_selection_carries_each_candidates_query():
    candidates = [
        DataSetChoice(id="IMF:CPI(1.0.0)", name="CPI", is_official=True),
        DataSetChoice(id="BIS:IR(2.1.0)", name="Rates"),
    ]
    details = {
        "ds1": _details(
            json_query=_app_query("IMF:CPI(1.0.0)"),
            summary="Consumer prices in France and Germany.",
            is_official=True,
            dimension_names={"REF_AREA": "Reference area"},
        ),
        "ds2": _details(
            dataset_urn="BIS:IR(2.1.0)",
            dataset_name="Rates",
            json_query=_app_query("BIS:IR(2.1.0)"),
        ),
    }
    outcome = _make_outcome(
        {},
        state=_state(
            DataQueryStatus.DATASET_SELECTION_REQUIRED,
            dimension_id_to_name={"ds1": {"REF_AREA": {"FR": "France"}}},
        ),
        mcp_payload=_mcp_payload(
            candidate_datasets=candidates, query_details=details, message="Pick a dataset."
        ),
    )

    structured = _structured(outcome, is_official=True)

    assert structured.message == "Pick a dataset."
    cpi, rates = structured.candidate_datasets
    assert (cpi.is_official, rates.is_official) == (True, False)
    assert cpi.query is not None and rates.query is not None
    assert cpi.query.executed is False
    assert cpi.query.is_official is True
    assert cpi.query.query_summary == "Consumer prices in France and Germany."
    assert cpi.query.dataset_name == "Consumer prices"
    assert cpi.query.filters[0].dimension_name == "Reference area"
    assert cpi.query.filters[0].values[0].name == "France"
    assert rates.query.dataset_urn == "BIS:IR(2.1.0)"


def test_structured_content_missing_dimensions_samples_the_values():
    values = [DimensionValueInfo(id=f"C{i:02d}", name=f"Country {i}") for i in range(14)]
    missing = MissingDimensionsInfo(
        dataset_id="ds1",
        dataset_urn="IMF:NSDP(7.0.0)",
        dimensions=[
            MissingDimensionInfo(dimension_id="COUNTRY", name="Country", available_values=values)
        ],
    )
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.MISSING_DIMENSIONS),
        mcp_payload=_mcp_payload(missing_dimensions=missing),
    )

    reported = _structured(outcome).missing_dimensions

    assert reported is not None
    assert reported.dataset_urn == "IMF:NSDP(7.0.0)"
    dimension = reported.dimensions[0]
    assert dimension.dimension_id == "COUNTRY"
    assert dimension.total_values == 14
    assert [value.id for value in dimension.sample_values] == [f"C{i:02d}" for i in range(10)]


def test_structured_content_no_data_carries_the_message_alone():
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.NO_DATA),
        mcp_payload=_mcp_payload(message="No relevant data found."),
    )

    data = _structured(outcome).model_dump(by_alias=True, exclude_none=True)

    assert data == {
        "status": DataQueryStatus.NO_DATA,
        "message": "No relevant data found.",
        "queries": [],
        "candidateDatasets": [],
    }


def test_structured_content_invalid_time_period_reports_why_the_period_was_rejected():
    details = _details(
        json_query=_app_query("IMF:CPI(1.0.0)"),
        invalid_period=InvalidPeriodInfo(
            rejected_bound="end",
            requested_value="2030",
            available_start="2000",
            available_end="2024",
        ),
    )
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.INVALID_TIME_PERIOD),
        mcp_payload=_mcp_payload(query_details={"ds1": details}, message="Adjust the period."),
    )

    structured = _structured(outcome)

    assert structured.message == "Adjust the period."
    [query] = structured.queries
    assert query.executed is False
    # The rejected period was never applied, so the query carries the reason instead.
    assert query.requested_period is None
    assert query.invalid_period is not None
    assert query.invalid_period.model_dump(by_alias=True) == {
        "rejectedBound": "endPeriod",
        "requestedValue": "2030",
        "availablePeriod": {"startPeriod": "2000", "endPeriod": "2024"},
    }


def test_structured_content_reports_the_query_details():
    query = _timed_query("IMF:CPI(1.0.0)", ["2020-01-01", "2024-12-31"])
    query.metadata.dataset_url = "https://data.example/IMF:CPI"
    query.filters.append(
        JsonComponentQuery(component_code="INDICATOR", operator=JsonQueryOperator.IN, values=["X"])
    )
    details = _details(
        summary="Consumer prices in France and Germany.",
        last_updated="2026-09-01",
        provider="IMF",
        is_official=True,
        default_dimension_ids=["REF_AREA"],
        default_time_period=True,
    )
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}), json_query=query)},
        mcp_payload=_mcp_payload(
            query_details={"ds1": details},
            message="Mention the source.",
            executed_at="2026-09-23T10:00:00+00:00",
        ),
    )

    structured = _structured(outcome, is_official=True)

    assert structured.message == "Mention the source."
    assert structured.executed_at == "2026-09-23T10:00:00+00:00"
    [record] = structured.queries
    assert record.query_summary == "Consumer prices in France and Germany."
    assert record.last_updated == "2026-09-01"
    assert record.provider == "IMF"
    assert record.is_official is True
    assert record.dataset_url == "https://data.example/IMF:CPI"
    assert record.requested_period is not None and record.requested_period.is_default is True
    flags = {f.dimension_id: (f.is_indicator, f.is_default) for f in record.filters}
    assert flags == {"REF_AREA": (False, True), "INDICATOR": (True, False)}


def test_structured_content_omits_the_fields_the_config_disables():
    query = _json_query("IMF:CPI(1.0.0)")
    query.metadata.dataset_url = "https://data.example/IMF:CPI"
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}), json_query=query)},
        mcp_payload=_mcp_payload(
            query_details={"ds1": _details(provider="IMF", is_official=True)},
            executed_at="2026-09-23T10:00:00+00:00",
        ),
    )

    structured = _structured(
        outcome,
        executed_at=False,
        provider=False,
        dataset_url=False,
        explorer_link=ExplorerLinkPolicy.never,
    )

    assert structured.executed_at is None
    [record] = structured.queries
    assert record.provider is None
    assert record.dataset_url is None
    assert record.data_explorer_url is None
    # Off by default: a channel that does not mark official datasets must not advertise it.
    assert record.is_official is None


def test_structured_content_explorer_link_only_when_no_data():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)")),
            "ds2": _response(
                "BIS:IR(2.1.0)", pd.DataFrame(), json_query=_json_query("BIS:IR(2.1.0)")
            ),
        }
    )

    structured = _structured(outcome, explorer_link=ExplorerLinkPolicy.only_when_no_data)

    assert [q.data_explorer_url for q in structured.queries] == [
        None,
        "https://explorer.example/query",
    ]


def _execution(mcp_app: bool = True, df: pd.DataFrame | None = None, **status):
    frame = pd.DataFrame() if df is None else df
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)", frame, json_query=_json_query("IMF:CPI(1.0.0)"), **status
            )
        }
    )
    execution = _structured(outcome, mcp_app=mcp_app).queries[0].execution
    assert execution is not None
    return execution


def test_execution_reports_a_failed_request_with_its_reason():
    execution = _execution(request_status=DataRequestStatus.FAILED, reason="Timed out.")

    assert execution.result is ExecutionResult.REQUEST_FAILED
    assert execution.reason == "Timed out."
    assert execution.advice is not None and "retry" in execution.advice


def test_execution_reports_no_data():
    execution = _execution()

    assert execution.result is ExecutionResult.NO_DATA
    assert execution.advice is not None and "time period" in execution.advice


def test_execution_parsing_failure_points_to_the_widget_only_when_one_is_bound():
    with_widget = _execution(parsing_status=DataParsingStatus.FAILED)
    without_widget = _execution(mcp_app=False, parsing_status=DataParsingStatus.FAILED)

    assert with_widget.result is without_widget.result is ExecutionResult.PARSING_FAILED
    assert with_widget.advice is not None and "widget" in with_widget.advice
    assert without_widget.advice is not None and "widget" not in without_widget.advice


def test_execution_reports_data_that_was_only_partly_parsed():
    with_widget = _execution(
        df=pd.DataFrame({"x": [1]}), parsing_status=DataParsingStatus.PARTIALLY_FAILED
    )
    without_widget = _execution(
        mcp_app=False,
        df=pd.DataFrame({"x": [1]}),
        parsing_status=DataParsingStatus.PARTIALLY_FAILED,
    )

    assert with_widget.result is without_widget.result is ExecutionResult.PARTIALLY_PARSED
    assert with_widget.advice is not None and "widget" in with_widget.advice
    assert without_widget.advice is not None and "widget" not in without_widget.advice


# ~~~~~~~~~~~~~ `_meta`: what the clients read ~~~~~~~~~~~~~


def test_meta_carries_one_payload_per_audience():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    meta = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())

    assert meta is not None
    assert set(meta) == {"statgpt.dialx.ai/mcp-app", "statgpt.dialx.ai/client"}


def test_meta_namespace_is_configurable():
    outcome = _make_outcome({})

    meta = data_query_outcome_to_meta(
        outcome, _channel_config(), _tool_config(namespace="data.example.org")
    )

    assert meta is not None
    assert set(meta) == {"data.example.org/mcp-app", "data.example.org/client"}


def test_meta_omits_the_mcp_app_payload_when_the_tool_binds_no_widget():
    outcome = _make_outcome({})

    meta = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config(mcp_app=False))

    assert meta is not None
    assert set(meta) == {"statgpt.dialx.ai/client"}


def test_meta_omits_a_disabled_client_payload():
    outcome = _make_outcome({})

    meta = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config(client=False))

    assert meta is not None
    assert set(meta) == {"statgpt.dialx.ai/mcp-app"}


def test_meta_is_omitted_when_no_audience_has_a_reader():
    outcome = _make_outcome({})

    meta = data_query_outcome_to_meta(
        outcome, _channel_config(), _tool_config(mcp_app=False, client=False)
    )

    assert meta is None


def test_mcp_app_meta_carries_the_sdmx_query_model():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    payload = data_query_outcome_to_meta(
        outcome, _channel_config(), _tool_config(), message="answer"
    )["statgpt.dialx.ai/mcp-app"]

    assert payload["status"] == DataQueryStatus.DATA_AVAILABLE
    assert payload["message"] == "answer"
    assert payload["version"] == 3
    assert payload["tools"] == {"sdmxProxy": "sdmx_query_app"}
    assert "import sdmx" in payload["pythonCode"]
    query = payload["queries"][0]
    assert query["urn"] == "IMF:CPI(1.0.0)"
    assert query["queryId"].startswith("dq_")
    assert query["disabled"] is False
    assert query["sdmx1Source"] == "IMF_DATA"
    assert query["filters"][0]["componentCode"] == "REF_AREA"
    assert query["metadata"]["keyDimensionIdsInDsdOrder"] == ["REF_AREA", "INDICATOR"]


def test_mcp_app_meta_keeps_null_fields_so_its_shape_never_changes():
    outcome = _make_outcome({}, state=_state(DataQueryStatus.NO_DATA))

    payload = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())[
        "statgpt.dialx.ai/mcp-app"
    ]

    assert payload["message"] is None
    assert payload["missingDimensions"] is None
    assert payload["pythonCode"] is None
    assert payload["queries"] == []
    assert payload["candidateDatasets"] == []


def test_mcp_app_meta_omits_sdmx_proxy_when_unconfigured():
    outcome = _make_outcome({})

    payload = data_query_outcome_to_meta(
        outcome, _channel_config(sdmx_proxy_name=None), _tool_config()
    )["statgpt.dialx.ai/mcp-app"]

    assert payload["tools"] == {"sdmxProxy": None}


def test_mcp_app_meta_carries_every_available_value():
    # The full lists live here: only what the model reads is truncated.
    values = [DimensionValueInfo(id=f"C{i:02d}", name=f"Country {i}") for i in range(14)]
    missing = MissingDimensionsInfo(
        dataset_id="ds1",
        dataset_urn="IMF:NSDP(7.0.0)",
        dimensions=[
            MissingDimensionInfo(dimension_id="COUNTRY", name="Country", available_values=values)
        ],
    )
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.MISSING_DIMENSIONS),
        mcp_payload=_mcp_payload(missing_dimensions=missing),
    )

    payload = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())[
        "statgpt.dialx.ai/mcp-app"
    ]

    assert len(payload["missingDimensions"]["dimensions"][0]["availableValues"]) == 14
    assert payload["missingDimensions"]["datasetId"] == "ds1"


def test_client_meta_carries_links_and_resource_uris():
    df = pd.DataFrame({"x": [1]})
    query = _json_query("IMF:CPI(1.0.0)")
    query.metadata.dataset_url = "https://data.example.org/datasets/IMF:CPI"
    outcome = _make_outcome({"ds1": _response("IMF:CPI(1.0.0)", df, json_query=query)})

    payload = data_query_outcome_to_meta(
        outcome, _channel_config(), _tool_config(csv=True, markdown=True), message="answer"
    )["statgpt.dialx.ai/client"]

    assert payload["status"] == DataQueryStatus.DATA_AVAILABLE
    assert "message" not in payload
    assert payload["version"] == 3
    record = payload["queries"][0]
    assert record["urn"] == "IMF:CPI(1.0.0)"
    assert record["datasetName"] == "CPI [IMF:CPI(1.0.0)]"
    assert record["dataExplorerUrl"] == "https://explorer.example/query"
    assert record["datasetUrl"] == "https://data.example.org/datasets/IMF:CPI"
    assert record["seriesCount"] == 3
    assert record["resourceUris"] == [
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{record['queryId']}.csv",
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{record['queryId']}.md",
    ]


def test_client_meta_reports_the_same_query_id_as_the_structured_content():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    meta = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())
    structured = _structured(outcome)

    assert meta is not None
    query_id = structured.queries[0].query_id
    assert meta["statgpt.dialx.ai/client"]["queries"][0]["queryId"] == query_id
    assert meta["statgpt.dialx.ai/mcp-app"]["queries"][0]["queryId"] == query_id


def test_client_meta_omits_resources_for_an_empty_response():
    # An empty response contributes no resource, so it has no URI to report either.
    outcome = _make_outcome(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)", pd.DataFrame(), json_query=_json_query("IMF:CPI(1.0.0)")
            )
        },
        state=_state(DataQueryStatus.EXECUTED_NO_DATA),
    )

    payload = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())[
        "statgpt.dialx.ai/client"
    ]

    assert payload["queries"][0]["resourceUris"] == []
    assert "seriesCount" not in payload["queries"][0]


def test_client_meta_reports_constructed_queries_without_links():
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.NOT_EXECUTED),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    payload = data_query_outcome_to_meta(outcome, _channel_config(), _tool_config())[
        "statgpt.dialx.ai/client"
    ]

    record = payload["queries"][0]
    assert record["urn"] == "IMF:CPI(1.0.0)"
    assert record["resourceUris"] == []
    assert "dataExplorerUrl" not in record
