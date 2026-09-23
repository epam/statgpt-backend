from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from statgpt.app.chains.data_query.query_builder import mcp_details
from statgpt.app.chains.data_query.query_builder.query.execute_query import ExecuteQueryChain
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.common.data.base import (
    DataResponseStatus,
    DataSetQuery,
    DateTimeDimension,
    DimensionQuery,
    QueryOperator,
)
from statgpt.common.data.base.enums import InvalidDataSetQueryReasonType
from statgpt.common.data.base.query import InvalidDataSetQueryReason
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus
from statgpt.common.schemas.query import JsonQueryMetadata, JsonQueryWithMetadata


class _TimeDimension(DateTimeDimension):
    # Only `entity_id` and `name` are read off the dimensions.
    entity_id = "TIME_PERIOD"  # type: ignore[assignment]
    name = "Time period"  # type: ignore[assignment]
    source_id = "TIME_PERIOD"  # type: ignore[assignment]
    description = None  # type: ignore[assignment]
    is_mandatory = False  # type: ignore[assignment]
    is_time_dimension = True  # type: ignore[assignment]

    def format_value(self, value):
        return value


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        source_id="IMF:CPI(1.0.0)",
        name="Consumer prices",
        config=SimpleNamespace(citation=SimpleNamespace(provider="IMF"), is_official=True),
        dimensions=lambda: [SimpleNamespace(entity_id="REF_AREA", name="Area"), _TimeDimension()],
        updated_at=AsyncMock(return_value=datetime(2026, 9, 1)),
        to_json_query=lambda query: JsonQueryWithMetadata(
            urn="IMF:CPI(1.0.0)",
            filters=[],
            metadata=JsonQueryMetadata(
                country_dimension="REF_AREA",
                indicator_dimensions=[],
                time_period_dimension="TIME_PERIOD",
            ),
        ),
    )


def _chain_state(query: DataSetQuery, dataset: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_queries={"ds1": query},
        datasets_dict={"ds1": SimpleNamespace(data=dataset or _dataset())},
        auth_context=SimpleNamespace(),
    )


async def test_collect_query_details_reads_the_query_and_its_dataset():
    query = DataSetQuery(
        dimensions_queries=[
            DimensionQuery(
                dimension_id="REF_AREA", values=["FR"], operator=QueryOperator.IN, is_default=True
            ),
            DimensionQuery(
                dimension_id="TIME_PERIOD",
                values=["2020", "2024"],
                operator=QueryOperator.BETWEEN,
                is_default=True,
            ),
        ],
        short_summary="Consumer prices in France.",
    )

    details = (await mcp_details.collect_query_details(_chain_state(query)))["ds1"]  # type: ignore[arg-type]

    assert details.dataset_urn == "IMF:CPI(1.0.0)"
    assert details.dataset_name == "Consumer prices"
    assert details.summary == "Consumer prices in France."
    assert details.last_updated == "2026-09-01"
    assert details.provider == "IMF"
    assert details.is_official is True
    assert details.dimension_names == {"REF_AREA": "Area", "TIME_PERIOD": "Time period"}
    # The default time period is reported on its own, not as a default filter.
    assert details.default_dimension_ids == ["REF_AREA"]
    assert details.default_time_period is True
    assert details.json_query is None
    assert details.invalid_period is None


async def test_collect_query_details_records_the_constructed_query_and_the_rejected_period():
    query = DataSetQuery(
        dimensions_queries=[
            DimensionQuery(dimension_id="REF_AREA", values=["FR"], operator=QueryOperator.IN)
        ],
        is_valid=False,
        invalidity_reason=InvalidDataSetQueryReason(
            type=InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD,
            details={
                "field": "selected_end",
                "value": "2030",
                "available_start": "2000",
                "available_end": "2024",
            },
        ),
    )

    details = (
        await mcp_details.collect_query_details(
            _chain_state(query), include_query=True  # type: ignore[arg-type]
        )
    )["ds1"]

    assert details.json_query is not None
    assert details.json_query.urn == "IMF:CPI(1.0.0)"
    assert details.invalid_period is not None
    assert details.invalid_period.model_dump() == {
        "rejected_bound": "end",
        "requested_value": "2030",
        "available_start": "2000",
        "available_end": "2024",
    }


async def test_collect_query_details_never_raises():
    dataset = _dataset()
    dataset.updated_at = AsyncMock(side_effect=RuntimeError("source down"))

    details = await mcp_details.collect_query_details(
        _chain_state(DataSetQuery(dimensions_queries=[]), dataset)  # type: ignore[arg-type]
    )

    assert details == {}


def _response(rows: bool, parsing_status: DataParsingStatus) -> SimpleNamespace:
    return SimpleNamespace(
        is_empty=not rows,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=parsing_status
        ),
    )


def test_a_partly_parsed_response_without_rows_is_a_failure():
    responses = {"ds1": _response(rows=False, parsing_status=DataParsingStatus.PARTIALLY_FAILED)}

    assert ExecuteQueryChain._execution_status(responses) is DataQueryStatus.FAILED  # type: ignore[arg-type]


def test_a_partly_parsed_response_with_rows_has_data():
    responses = {"ds1": _response(rows=True, parsing_status=DataParsingStatus.PARTIALLY_FAILED)}

    assert ExecuteQueryChain._execution_status(responses) is DataQueryStatus.DATA_AVAILABLE  # type: ignore[arg-type]
