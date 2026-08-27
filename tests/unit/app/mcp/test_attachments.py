from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace

import pandas as pd

from statgpt.app.mcp.attachments import (
    data_query_artifact_to_resources,
    data_query_artifact_to_structured_content,
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
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.schemas.data_query_tool import DataQueryMcpResources, McpResource
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)

_FIXED_TS = datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc)
_FIXED_TS_STR = _FIXED_TS.strftime("%Y%m%dT%H%M%SZ")


def _state(status: DataQueryStatus = DataQueryStatus.DATA_AVAILABLE) -> SimpleNamespace:
    # The converter only reads `status` off the artifact's state.
    return SimpleNamespace(status=status)


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


def _make_artifact(
    data_responses: dict,
    state: SimpleNamespace | None = None,
    mcp_payload: DataQueryMcpPayload | None = None,
) -> DataQueryArtifact:
    # Bypass pydantic validation — the converter only reads data_responses, state and mcp_payload.
    return DataQueryArtifact.model_construct(
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
    )


def _resources(artifact: DataQueryArtifact, csv: bool = True, markdown: bool = False):
    config = DataQueryMcpResources(
        csv=McpResource(enabled_str=str(csv)),
        markdown_table=McpResource(enabled_str=str(markdown)),
    )
    return data_query_artifact_to_resources(artifact, config)


def _markdown_text(artifact: DataQueryArtifact) -> str:
    resources = _resources(artifact, csv=False, markdown=True)
    assert len(resources) == 1
    return resources[0].resource.text


def test_single_dataset_produces_one_csv_resource():
    df = pd.DataFrame({"country": ["FR", "DE"], "value": [1, 2]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(2.0.0)", df)})

    resources = _resources(artifact)

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
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df1),
            "ds2": _response("BIS:IR(2.1.0)", df2),
        }
    )

    resources = _resources(artifact)

    assert [str(r.resource.uri) for r in resources] == [
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.csv",
        f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.csv",
    ]


def test_each_response_uses_its_own_created_at():
    ts1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"a": [1]}), created_at=ts1),
            "ds2": _response("BIS:IR(2.1.0)", pd.DataFrame({"b": [2]}), created_at=ts2),
        }
    )

    resources = _resources(artifact)

    assert [str(r.resource.uri) for r in resources] == [
        "statgpt://data_query/IMF%3ACPI%281.0.0%29/20260101T000000Z.csv",
        "statgpt://data_query/BIS%3AIR%282.1.0%29/20260102T000000Z.csv",
    ]


def test_empty_dataframes_are_skipped():
    artifact = _make_artifact(
        {
            "empty": _response("IMF:EMPTY(1.0.0)", pd.DataFrame()),
            "ok": _response("IMF:OK(1.0.0)", pd.DataFrame({"x": [1]})),
        }
    )

    resources = _resources(artifact)

    assert len(resources) == 1
    assert (
        str(resources[0].resource.uri)
        == f"statgpt://data_query/IMF%3AOK%281.0.0%29/{_FIXED_TS_STR}.csv"
    )


def test_no_responses_returns_empty_list():
    artifact = _make_artifact({})

    assert _resources(artifact) == []


def test_csv_disabled_produces_no_resources():
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]}))})

    assert _resources(artifact, csv=False) == []


def test_both_payloads_enabled_are_emitted_per_response():
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"x": [1]})),
            "ds2": _response("BIS:IR(2.1.0)", pd.DataFrame({"y": [2]})),
        }
    )

    resources = _resources(artifact, csv=True, markdown=True)

    assert [(r.resource.mimeType, str(r.resource.uri)) for r in resources] == [
        ("text/csv", f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.csv"),
        ("text/markdown", f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.md"),
        ("text/csv", f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.csv"),
        ("text/markdown", f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.md"),
    ]


def test_markdown_resource_is_annotated_for_the_user():
    df = pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    resources = _resources(artifact, csv=False, markdown=True)

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
    artifact = _make_artifact(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)",
                df,
                component_names={"REF_AREA": "Reference area", "INDICATOR": "Indicator"},
            )
        }
    )

    text = _markdown_text(artifact)

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
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    header = _markdown_text(artifact).splitlines()[2]

    assert [cell.strip() for cell in header.strip("|").split("|")] == ["REF_AREA", "2024"]


def test_markdown_table_keeps_id_when_display_name_collides():
    df = pd.DataFrame({"FREQ": ["A"], "REF_AREA": ["FR"], "2024": [1.0]})
    artifact = _make_artifact(
        {
            "ds1": _response(
                "IMF:CPI(1.0.0)",
                df,
                # Both components claim the same display name: the first one gets it.
                component_names={"FREQ": "Frequency", "REF_AREA": "Frequency"},
            )
        }
    )

    header = _markdown_text(artifact).splitlines()[2]

    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "Frequency",
        "REF_AREA",
        "2024",
    ]


def test_markdown_table_is_not_padded():
    df = pd.DataFrame({"REF_AREA": ["FR"], "REF_AREA_Name": ["France"], "2024": [1.5]})
    artifact = _make_artifact(
        {"ds1": _response("IMF:CPI(1.0.0)", df, component_names={"REF_AREA": "Reference area"})}
    )

    lines = _markdown_text(artifact).splitlines()

    assert lines[2] == "| Reference area | 2024 |"
    # Numeric columns are right-aligned, everything else left-aligned, at minimum rule width.
    assert lines[3] == "|:---|---:|"
    assert lines[4] == "| France | 1.5 |"


def test_markdown_table_escapes_pipes_in_values():
    df = pd.DataFrame({"INDICATOR": ["GDP | current prices"], "2024": [1.5]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    row = _markdown_text(artifact).splitlines()[4]

    assert row == "| GDP \\| current prices | 1.5 |"


def test_markdown_table_drops_trailing_zero_of_whole_floats():
    df = pd.DataFrame({"REF_AREA": ["FR", "DE", "IT"], "2024": [123.0, 4.5, 1e14]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    rows = _markdown_text(artifact).splitlines()[4:]

    assert rows == ["| FR | 123 |", "| DE | 4.5 |", "| IT | 100000000000000 |"]


def test_markdown_table_keeps_textual_values_verbatim():
    # A `.0` in a value that arrived as text may be part of a code, so it is left alone.
    df = pd.DataFrame({"VERSION": ["1.0"], "2024": ["7.0"]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert _markdown_text(artifact).splitlines()[4] == "| 1.0 | 7.0 |"


def test_markdown_table_preserves_full_float_precision():
    df = pd.DataFrame({"REF_AREA": ["FR"], "2024": [112.345678901234]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert "112.345678901234" in _markdown_text(artifact)


def test_markdown_table_renders_missing_values_as_na():
    df = pd.DataFrame({"REF_AREA": ["FR", "DE"], "2024": [1.5, None]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(1.0.0)", df)})

    assert "NA" in _markdown_text(artifact)


def test_markdown_table_skips_empty_responses():
    artifact = _make_artifact(
        {
            "empty": _response("IMF:EMPTY(1.0.0)", pd.DataFrame()),
            "ok": _response("IMF:OK(1.0.0)", pd.DataFrame({"x": [1]})),
        }
    )

    resources = _resources(artifact, csv=False, markdown=True)

    assert len(resources) == 1
    assert str(resources[0].resource.uri).startswith("statgpt://data_query/IMF%3AOK%281.0.0%29/")


def _app_query(urn: str) -> AppJsonQueryWithMetadata:
    return AppJsonQueryWithMetadata.from_common(_json_query(urn))


def test_structured_content_data_available_serializes_single_query():
    df = pd.DataFrame({"x": [1]})
    artifact = _make_artifact(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    data = structured.model_dump(by_alias=True)
    assert data["status"] == DataQueryStatus.DATA_AVAILABLE
    assert data["tools"] == {"sdmxProxy": "sdmx_query_app"}
    assert data["version"] == 2
    assert "import sdmx" in data["pythonCode"]
    assert len(data["queries"]) == 1
    query = data["queries"][0]
    assert query["urn"] == "IMF:CPI(1.0.0)"
    assert query["disabled"] is False
    assert query["sdmx1Source"] == "IMF_DATA"
    assert query["filters"][0]["componentCode"] == "REF_AREA"
    assert query["metadata"]["keyDimensionIdsInDsdOrder"] == ["REF_AREA", "INDICATOR"]


def test_structured_content_omits_sdmx_proxy_when_unconfigured():
    df = pd.DataFrame({"x": [1]})
    artifact = _make_artifact(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))}
    )

    structured = data_query_artifact_to_structured_content(
        artifact, _channel_config(sdmx_proxy_name=None)
    )

    assert structured.tools.sdmx_proxy is None


def test_structured_content_data_available_preserves_insertion_order():
    df = pd.DataFrame({"x": [1]})
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)")),
            "ds2": _response("BIS:IR(2.1.0)", df, json_query=_json_query("BIS:IR(2.1.0)")),
        }
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert [q.urn for q in structured.queries] == ["IMF:CPI(1.0.0)", "BIS:IR(2.1.0)"]


def test_structured_content_data_available_skips_responses_without_query():
    df = pd.DataFrame({"x": [1]})
    artifact = _make_artifact(
        {
            "no_query": _response("IMF:NOQ(1.0.0)", df, json_query=None),
            "ok": _response("IMF:OK(1.0.0)", df, json_query=_json_query("IMF:OK(1.0.0)")),
        }
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert [q.urn for q in structured.queries] == ["IMF:OK(1.0.0)"]


def test_structured_content_no_data_carries_message():
    artifact = _make_artifact({}, state=_state(DataQueryStatus.NO_DATA))

    structured = data_query_artifact_to_structured_content(
        artifact, _channel_config(), message="No relevant data found."
    )

    assert structured.status is DataQueryStatus.NO_DATA
    assert structured.message == "No relevant data found."
    assert structured.queries == []
    assert structured.python_code is None
    assert structured.tools.sdmx_proxy == "sdmx_query_app"


def test_structured_content_dataset_selection_carries_candidates():
    candidates = [
        DataSetChoice(id="IMF:CPI", name="CPI", description="Prices", is_official=True),
        DataSetChoice(id="BIS:IR", name="Rates"),
    ]
    artifact = _make_artifact(
        {},
        state=_state(DataQueryStatus.DATASET_SELECTION_REQUIRED),
        mcp_payload=_mcp_payload(candidate_datasets=candidates),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.DATASET_SELECTION_REQUIRED
    data = structured.model_dump(by_alias=True)
    assert [c["id"] for c in data["candidateDatasets"]] == ["IMF:CPI", "BIS:IR"]
    assert data["candidateDatasets"][0]["isOfficial"] is True
    assert data["queries"] == []


def test_structured_content_missing_dimensions_carries_payload():
    missing = MissingDimensionsInfo(
        dataset_id="ds1",
        dimensions=[
            MissingDimensionInfo(
                dimension_id="FREQ",
                name="Frequency",
                available_values=[DimensionValueInfo(id="A", name="Annual")],
            )
        ],
    )
    artifact = _make_artifact(
        {},
        state=_state(DataQueryStatus.MISSING_DIMENSIONS),
        mcp_payload=_mcp_payload(missing_dimensions=missing),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.MISSING_DIMENSIONS
    data = structured.model_dump(by_alias=True)["missingDimensions"]
    assert data["datasetId"] == "ds1"
    assert data["dimensions"][0]["dimensionId"] == "FREQ"
    assert data["dimensions"][0]["availableValues"][0] == {
        "id": "A",
        "name": "Annual",
        "description": None,
    }


def test_structured_content_invalid_time_period_carries_message_only():
    # The rejected time period was never applied to the constructed queries, so reporting them
    # would describe a query the user did not ask for.
    artifact = _make_artifact(
        {},
        state=_state(DataQueryStatus.INVALID_TIME_PERIOD),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    structured = data_query_artifact_to_structured_content(
        artifact, _channel_config(), message="The selected end date (2030) is outside 2000-2019."
    )

    assert structured.status is DataQueryStatus.INVALID_TIME_PERIOD
    assert structured.message == "The selected end date (2030) is outside 2000-2019."
    assert structured.queries == []
    assert structured.python_code is None


def test_structured_content_not_executed_uses_constructed_queries():
    artifact = _make_artifact(
        {},
        state=_state(DataQueryStatus.NOT_EXECUTED),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.NOT_EXECUTED
    assert [q.urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert "import sdmx" in structured.python_code


def test_structured_content_not_executed_without_queries_has_no_python_code():
    artifact = _make_artifact({}, state=_state(DataQueryStatus.NOT_EXECUTED))

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.NOT_EXECUTED
    assert structured.queries == []
    assert structured.python_code is None


def test_structured_content_executed_no_data_still_reports_the_queries():
    # The queries ran and returned nothing: a client should still be able to show what was asked.
    df = pd.DataFrame()
    artifact = _make_artifact(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))},
        state=_state(DataQueryStatus.EXECUTED_NO_DATA),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.EXECUTED_NO_DATA
    assert [q.urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert "import sdmx" in structured.python_code


def test_structured_content_failed_reports_the_queries_that_errored():
    df = pd.DataFrame()
    artifact = _make_artifact(
        {"ds1": _response("IMF:CPI(1.0.0)", df, json_query=_json_query("IMF:CPI(1.0.0)"))},
        state=_state(DataQueryStatus.FAILED),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.FAILED
    assert [q.urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert "import sdmx" in structured.python_code


def test_structured_content_failed_without_responses_has_no_queries():
    # `failed` is also the default status, reached when the pipeline errored before executing
    # anything — there is nothing to report but the status.
    artifact = _make_artifact({}, state=_state(DataQueryStatus.FAILED))

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.FAILED
    assert structured.queries == []
    assert structured.python_code is None
