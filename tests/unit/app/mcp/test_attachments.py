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
    MissingDimensionInfo,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.common.schemas.data_query_tool import (
    DataQueryMcpMeta,
    DataQueryMcpResources,
    McpResource,
    ToggleableConfig,
)
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
) -> DataQueryMcpPayload:
    return DataQueryMcpPayload(
        constructed_queries=constructed_queries or [],
        candidate_datasets=candidate_datasets or [],
        missing_dimensions=missing_dimensions,
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
) -> SimpleNamespace:
    # The converter only reads `mcp_app_resource_uri` off the tool config, and `mcp_meta` /
    # `mcp_resources` off its details.
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
        ),
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

    data = data_query_outcome_to_structured_content(outcome).model_dump(
        by_alias=True, exclude_none=True
    )

    # The status, the python code, the companion tools and the version are client-facing: `_meta`
    # carries them.
    assert set(data) == {"queries", "candidateDatasets"}
    assert len(data["queries"]) == 1
    query = data["queries"][0]
    assert query["datasetUrn"] == "IMF:CPI(1.0.0)"
    assert query["datasetName"] == "CPI [IMF:CPI(1.0.0)]"
    assert query["executed"] is True
    assert query["queryId"].startswith("dq_")
    assert query["seriesCount"] == 3
    assert query["requestedPeriod"] == {"startPeriod": "2020-01-01", "endPeriod": "2024-12-31"}
    assert query["factualPeriod"] == {"startPeriod": "2020", "endPeriod": "2024"}
    # The time period is reported once, as requestedPeriod - not as another filter.
    assert [f["dimensionId"] for f in query["filters"]] == ["REF_AREA"]
    assert query["filters"][0]["dimensionName"] == "Reference area"
    assert query["filters"][0]["operator"] == JsonQueryOperator.IN
    # The name of a value is reported when the pipeline resolved it, the id alone otherwise.
    assert query["filters"][0]["values"] == [{"id": "FR", "name": "France"}, {"id": "DE"}]


def test_structured_content_reports_the_query_id_the_resources_use():
    # The id joins the query to its CSV/Markdown resources, so both must agree.
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    query_id = data_query_outcome_to_structured_content(outcome).queries[0].query_id

    assert str(_resources(outcome)[0].resource.uri).endswith(f"/{query_id}.csv")


def test_query_id_is_stable_for_the_same_query_on_the_same_date():
    df = pd.DataFrame({"x": [1]})
    same_day = datetime(2026, 4, 20, 23, 59, 0, tzinfo=timezone.utc)
    next_day = datetime(2026, 4, 21, 0, 1, 0, tzinfo=timezone.utc)

    def query_id(created_at: datetime, urn: str = "IMF:CPI(1.0.0)") -> str:
        outcome = _make_outcome(
            {"ds1": _response(urn, df, created_at=created_at, json_query=_json_query(urn))}
        )
        return data_query_outcome_to_structured_content(outcome).queries[0].query_id

    assert query_id(_FIXED_TS) == query_id(same_day)
    # The same query a month (or a day) later may well return different data.
    assert query_id(_FIXED_TS) != query_id(next_day)
    assert query_id(_FIXED_TS) != query_id(_FIXED_TS, urn="BIS:IR(2.1.0)")


def test_structured_content_truncates_a_long_value_list():
    values = [f"C{i:02d}" for i in range(25)]
    query = _json_query("IMF:CPI(1.0.0)")
    query.filters = [
        JsonComponentQuery(component_code="INDICATOR", operator=JsonQueryOperator.IN, values=values)
    ]
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}), json_query=query)}
    )

    reported = data_query_outcome_to_structured_content(outcome).queries[0].filters[0]

    assert reported.total_values == 25
    assert reported.returned_values == 10
    assert [value.id for value in reported.values] == values[:10]


def test_structured_content_omits_the_total_when_nothing_is_truncated():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    assert (
        data_query_outcome_to_structured_content(outcome).queries[0].filters[0].total_values is None
    )


def test_structured_content_preserves_insertion_order():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)")),
            "ds2": _response("BIS:IR(2.1.0)", df, json_query=_json_query("BIS:IR(2.1.0)")),
        }
    )

    structured = data_query_outcome_to_structured_content(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)", "BIS:IR(2.1.0)"]


def test_structured_content_skips_responses_without_query():
    df = pd.DataFrame({"x": [1]})
    outcome = _make_outcome(
        {
            "no_query": _response("IMF:NOQ(1.0.0)", df, json_query=None),
            "ok": _response("IMF:OK(1.0.0)", df, json_query=_json_query("IMF:OK(1.0.0)")),
        }
    )

    structured = data_query_outcome_to_structured_content(outcome)

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

    structured = data_query_outcome_to_structured_content(outcome)

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

    structured = data_query_outcome_to_structured_content(outcome)

    assert [q.dataset_urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]


def test_structured_content_failed_without_responses_has_no_queries():
    # `failed` is also the default status, reached when the pipeline errored before executing
    # anything - there is nothing to report, and the text block explains it.
    outcome = _make_outcome({}, state=_state(DataQueryStatus.FAILED))

    assert data_query_outcome_to_structured_content(outcome).queries == []


def test_structured_content_not_executed_flags_the_constructed_queries():
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.NOT_EXECUTED),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    structured = data_query_outcome_to_structured_content(outcome)

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

    data = data_query_outcome_to_structured_content(outcome).model_dump(by_alias=True)

    assert data["candidateDatasets"] == [
        {"id": "IMF:CPI", "name": "CPI", "isOfficial": True},
        {"id": "BIS:IR", "name": "Rates", "isOfficial": False},
    ]
    assert data["queries"] == []


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

    reported = data_query_outcome_to_structured_content(outcome).missing_dimensions

    assert reported is not None
    assert reported.dataset_urn == "IMF:NSDP(7.0.0)"
    dimension = reported.dimensions[0]
    assert dimension.dimension_id == "COUNTRY"
    assert dimension.total_values == 14
    assert [value.id for value in dimension.sample_values] == [f"C{i:02d}" for i in range(10)]


def test_structured_content_no_data_is_empty():
    # The text content block explains the outcome; there is no query to report.
    outcome = _make_outcome({}, state=_state(DataQueryStatus.NO_DATA))

    data = data_query_outcome_to_structured_content(outcome).model_dump(
        by_alias=True, exclude_none=True
    )

    assert data == {"queries": [], "candidateDatasets": []}


def test_structured_content_invalid_time_period_is_empty():
    # The rejected time period was never applied to the constructed queries, so reporting them
    # would describe a query the user did not ask for.
    outcome = _make_outcome(
        {},
        state=_state(DataQueryStatus.INVALID_TIME_PERIOD),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    assert data_query_outcome_to_structured_content(outcome).queries == []


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
    structured = data_query_outcome_to_structured_content(outcome)

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
