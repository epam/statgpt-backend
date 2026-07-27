from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace

import pandas as pd

from statgpt.app.mcp.attachments import (
    data_query_artifact_to_resources,
    data_query_artifact_to_structured_content,
)
from statgpt.app.schemas.enums import DataQueryStatus
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.query_builder import (
    DataQueryMcpPayload,
    DataSetChoice,
    DimensionValueInfo,
    MissingDimensionInfo,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
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
) -> SimpleNamespace:
    return SimpleNamespace(
        resource_path=resource_path,
        visual_dataframe=df,
        csv_dataframe=df,
        created_at=created_at,
        json_query=json_query,
    )


def test_single_dataset_produces_one_csv_resource():
    df = pd.DataFrame({"country": ["FR", "DE"], "value": [1, 2]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(2.0.0)", df)})

    resources = data_query_artifact_to_resources(artifact)

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

    resources = data_query_artifact_to_resources(artifact)

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

    resources = data_query_artifact_to_resources(artifact)

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

    resources = data_query_artifact_to_resources(artifact)

    assert len(resources) == 1
    assert (
        str(resources[0].resource.uri)
        == f"statgpt://data_query/IMF%3AOK%281.0.0%29/{_FIXED_TS_STR}.csv"
    )


def test_no_responses_returns_empty_list():
    artifact = _make_artifact({})

    assert data_query_artifact_to_resources(artifact) == []


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


def test_structured_content_invalid_time_period_uses_constructed_queries():
    artifact = _make_artifact(
        {},
        state=_state(DataQueryStatus.INVALID_TIME_PERIOD),
        mcp_payload=_mcp_payload(constructed_queries=[_app_query("IMF:CPI(1.0.0)")]),
    )

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.INVALID_TIME_PERIOD
    assert [q.urn for q in structured.queries] == ["IMF:CPI(1.0.0)"]
    assert "import sdmx" in structured.python_code


def test_structured_content_not_executed_without_queries_has_no_python_code():
    artifact = _make_artifact({}, state=_state(DataQueryStatus.NOT_EXECUTED))

    structured = data_query_artifact_to_structured_content(artifact, _channel_config())

    assert structured.status is DataQueryStatus.NOT_EXECUTED
    assert structured.queries == []
    assert structured.python_code is None
