from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.exceptions import ToolError

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.common.data.base import (
    Attribute,
    CategoricalDimension,
    DataSetAvailabilityQuery,
    Query,
)
from statgpt.common.data.base.enums import AttributeType, DimensionDataType, QueryOperator
from statgpt.common.schemas.availability_query_tool import AvailabilityQueryToolDetails
from statgpt.common.schemas.dataset_structure_tool import DatasetStructureToolDetails
from statgpt.common.schemas.tool_details import AvailableDatasetsDetails
from statgpt.common.schemas.tools import (
    AvailabilityQueryTool,
    AvailableDatasetsTool,
    DatasetStructureTool,
)

_AUTH = SimpleNamespace()
_AGENCIES = [
    SimpleNamespace(id="IMF", name="Intl Monetary Fund"),
    SimpleNamespace(id="WB", name="World Bank"),
]


def _dataset(
    source_id: str = "IMF:CPI(1.0.0)",
    entity_id: str = "cpi",
    provider: str | None = "IMF",
    updated_at: datetime | None = None,
    citation_last_updated: str | None = "2023-06-15",
    provider_agencies: list | None = None,
    dimensions: list | None = None,
    attributes: list | None = None,
) -> SimpleNamespace:
    citation = (
        SimpleNamespace(
            description=None,
            provider=provider,
            provider_agency_names_with_fallback_to_provider=[provider],
            provider_agencies=provider_agencies,
            last_updated=citation_last_updated,
        )
        if provider
        else None
    )
    return SimpleNamespace(
        source_id=source_id,
        entity_id=entity_id,
        name="Consumer Price Index",
        description="Prices.",
        dataset_url=None,
        config=SimpleNamespace(citation=citation),
        updated_at=AsyncMock(return_value=updated_at),
        dimensions=lambda: dimensions or [],
        attributes=lambda: attributes or [],
    )


def _categorical_dimension(n_values: int) -> MagicMock:
    # A spec'd mock passes the `isinstance` checks the record builder relies on.
    dim = MagicMock(spec=CategoricalDimension)
    dim.entity_id = "REF_AREA"
    dim.name = "Reference area"
    dim.description = None
    dim.dimension_type = DimensionDataType.CATEGORY
    dim.available_values = [
        SimpleNamespace(query_id=f"C{i}", name=f"Country {i}") for i in range(n_values)
    ]
    return dim


def _attribute() -> MagicMock:
    attr = MagicMock(spec=Attribute)
    attr.entity_id = "UNIT_MULT"
    attr.name = "Unit multiplier"
    attr.description = "Power of ten."
    attr.attribute_type = AttributeType.STRING
    return attr


def _build(tool_config, inputs: dict) -> StatGptMcpTool:
    return StatGptMcpTool.from_config(
        tool_config,
        # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
        SimpleNamespace(  # type: ignore[arg-type]
            mcp=SimpleNamespace(tool_name_prefix=""), out_of_scope=None, locale="en"
        ),
        inputs=inputs,
        auth_context=_AUTH,  # type: ignore[arg-type]
    )


# ~~~~~~~~~~~~~ available datasets ~~~~~~~~~~~~~


def _datasets_inputs(datasets: list, indicator_counts: dict[str, int] | None = None) -> dict:
    data_service = SimpleNamespace(
        list_available_datasets=AsyncMock(
            return_value=[SimpleNamespace(data=ds) for ds in datasets]
        ),
        get_indicator_counts=AsyncMock(return_value=indicator_counts),
    )
    return {
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.AUTH_CONTEXT: _AUTH,
    }


async def test_available_datasets_is_structured_only():
    inputs = _datasets_inputs([_dataset(), _dataset(source_id="WB:GDP(1.0)", provider=None)])
    tool_config = AvailableDatasetsTool(
        name="datasets",
        description="Datasets.",
        details=AvailableDatasetsDetails(include_indicator_count=False),
    )

    tool_result = await _build(tool_config, inputs).run({})

    # No text block: the complete result lives in structured content, with nulls omitted.
    assert tool_result.content == []
    assert tool_result.structured_content == {
        "providers": [{"name": "IMF", "datasetCount": 1}],
        "datasets": [
            {
                "id": "IMF:CPI(1.0.0)",
                "name": "Consumer Price Index",
                "description": "Prices.",
                "provider": "IMF",
                "lastUpdated": "2023-06-15",
            },
            {"id": "WB:GDP(1.0)", "name": "Consumer Price Index", "description": "Prices."},
        ],
        "totalDatasets": 2,
        "totalAgencies": 1,
    }


async def test_available_datasets_omits_an_unparsable_citation_date():
    # `lastUpdated` is an ISO 8601 contract: free text that cannot be parsed into a date is
    # dropped rather than passed through.
    inputs = _datasets_inputs([_dataset(citation_last_updated="Quarterly, when ready")])
    tool_config = AvailableDatasetsTool(name="datasets", description="Datasets.")

    structured = (await _build(tool_config, inputs).run({})).structured_content

    assert structured is not None
    assert "lastUpdated" not in structured["datasets"][0]


async def test_available_datasets_reports_indicator_counts_when_configured():
    inputs = _datasets_inputs([_dataset()], indicator_counts={"cpi": 42})
    tool_config = AvailableDatasetsTool(
        name="datasets",
        description="Datasets.",
        details=AvailableDatasetsDetails(include_indicator_count=True),
    )

    structured = (await _build(tool_config, inputs).run({})).structured_content

    assert structured is not None
    assert structured["datasets"][0]["numberOfIndicators"] == 42
    assert structured["totalIndicators"] == 42


# ~~~~~~~~~~~~~ dataset structure ~~~~~~~~~~~~~


def _structure_inputs(dataset) -> dict:
    data_service = SimpleNamespace(get_dataset_by_source_id=AsyncMock(return_value=dataset))
    return {
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.AUTH_CONTEXT: _AUTH,
        ChainParametersConfig.CHOICE: None,
        ChainParametersConfig.STATE: {},
    }


async def test_dataset_structure_not_found():
    tool_config = DatasetStructureTool(name="structure", description="Structure.")

    tool_result = await _build(tool_config, _structure_inputs(None)).run(
        {"dataset_id": "IMF:NOPE(1.0)"}
    )

    assert tool_result.content == []
    assert tool_result.structured_content == {
        "datasetId": "IMF:NOPE(1.0)",
        "found": False,
        "dimensions": [],
        "attributes": [],
    }


async def test_dataset_structure_found_uses_the_source_update_date():
    tool_config = DatasetStructureTool(name="structure", description="Structure.")
    dataset = _dataset(updated_at=datetime(2024, 1, 31, 12, 0, tzinfo=timezone.utc))

    tool_result = await _build(tool_config, _structure_inputs(dataset)).run(
        {"dataset_id": "IMF:CPI(1.0.0)"}
    )

    assert tool_result.content == []
    assert tool_result.structured_content == {
        "datasetId": "IMF:CPI(1.0.0)",
        "found": True,
        "name": "Consumer Price Index",
        "description": "Prices.",
        "provider": "IMF",
        "lastUpdated": "2024-01-31",
        "dimensions": [],
        "attributes": [],
    }


async def test_dataset_structure_hides_provider_agencies_by_default():
    # Mirrors the text rendering: `include_provider_agencies` defaults to False.
    tool_config = DatasetStructureTool(name="structure", description="Structure.")
    dataset = _dataset(provider_agencies=_AGENCIES)

    structured = (
        await _build(tool_config, _structure_inputs(dataset)).run({"dataset_id": "IMF:CPI(1.0.0)"})
    ).structured_content

    assert structured is not None
    assert "providerAgencies" not in structured


async def test_dataset_structure_exposes_provider_agencies_when_configured():
    tool_config = DatasetStructureTool(
        name="structure",
        description="Structure.",
        details=DatasetStructureToolDetails(include_provider_agencies=True),
    )
    dataset = _dataset(provider_agencies=_AGENCIES)

    structured = (
        await _build(tool_config, _structure_inputs(dataset)).run({"dataset_id": "IMF:CPI(1.0.0)"})
    ).structured_content

    assert structured is not None
    assert structured["providerAgencies"] == [
        {"id": "IMF", "name": "Intl Monetary Fund"},
        {"id": "WB", "name": "World Bank"},
    ]


async def test_dataset_structure_lists_every_value_of_a_small_dimension():
    tool_config = DatasetStructureTool(name="structure", description="Structure.")
    dataset = _dataset(dimensions=[_categorical_dimension(3)], attributes=[_attribute()])

    structured = (
        await _build(tool_config, _structure_inputs(dataset)).run({"dataset_id": "IMF:CPI(1.0.0)"})
    ).structured_content

    assert structured is not None
    assert structured["dimensions"] == [
        {
            "id": "REF_AREA",
            "name": "Reference area",
            "type": "category",
            "totalValues": 3,
            "sampleValues": [
                {"id": "C0", "name": "Country 0"},
                {"id": "C1", "name": "Country 1"},
                {"id": "C2", "name": "Country 2"},
            ],
        }
    ]
    assert structured["attributes"] == [
        {
            "id": "UNIT_MULT",
            "name": "Unit multiplier",
            "type": "string",
            "description": "Power of ten.",
        }
    ]


async def test_dataset_structure_samples_a_large_dimension_like_the_text_rendering():
    # Same rule as the detailed formatter: a bounded sample (random, so only its size and
    # membership are checked) once the values exceed the limit.
    tool_config = DatasetStructureTool(name="structure", description="Structure.")
    dataset = _dataset(dimensions=[_categorical_dimension(25)])

    structured = (
        await _build(tool_config, _structure_inputs(dataset)).run({"dataset_id": "IMF:CPI(1.0.0)"})
    ).structured_content

    assert structured is not None
    (dimension,) = structured["dimensions"]
    assert dimension["totalValues"] == 25
    sample_ids = [value["id"] for value in dimension["sampleValues"]]
    assert len(sample_ids) == 10
    assert len(set(sample_ids)) == 10
    assert set(sample_ids) <= {f"C{i}" for i in range(25)}


# ~~~~~~~~~~~~~ availability query ~~~~~~~~~~~~~


def _availability_dimension(entity_id: str, name: str, names_by_id: dict[str, str]) -> MagicMock:
    # A spec'd mock passes the `isinstance(..., CategoricalDimension)` check the builder relies on.
    dim = MagicMock(spec=CategoricalDimension)
    dim.entity_id = entity_id
    dim.name = name
    dim.name_by_query_id.side_effect = lambda code: names_by_id.get(code)
    return dim


def _availability_dataset(
    dimensions: list, availability_result: DataSetAvailabilityQuery, time_dimension=None
) -> SimpleNamespace:
    dims_by_id = {d.entity_id: d for d in dimensions}
    return SimpleNamespace(
        source_id="IMF:CPI(1.0.0)",
        dimensions=lambda: dimensions,
        dimension=lambda dim_id: dims_by_id[dim_id],
        get_time_dimension=lambda: time_dimension,
        availability_query=AsyncMock(return_value=availability_result),
    )


def _availability_inputs(dataset) -> dict:
    data_service = SimpleNamespace(get_dataset_by_source_id=AsyncMock(return_value=dataset))
    return {
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.AUTH_CONTEXT: _AUTH,
        ChainParametersConfig.CHOICE: None,
        ChainParametersConfig.STATE: {},
    }


def _availability_tool_config(hard_limit: int = 500) -> AvailabilityQueryTool:
    return AvailabilityQueryTool(
        name="availability",
        description="Availability.",
        details=AvailabilityQueryToolDetails(hard_limit=hard_limit),
    )


async def test_availability_query_is_structured_only():
    country = _availability_dimension(
        "COUNTRY", "Reference area", {"USA": "United States", "GBR": "United Kingdom"}
    )
    freq = _availability_dimension("FREQ", "Frequency", {"A": "Annual"})
    series = _availability_dimension("SERIES", "Series", {"51401": "Series 51401"})
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={
            "COUNTRY": Query(values=["USA", "GBR"], operator=QueryOperator.IN),
            "FREQ": Query(values=["A"], operator=QueryOperator.IN),
        }
    )
    dataset = _availability_dataset([series, country, freq], result)

    tool_result = await _build(_availability_tool_config(), _availability_inputs(dataset)).run(
        {
            "dataset_id": "IMF:CPI(1.0.0)",
            "partial_query": {"SERIES": ["51401"]},
            "codes_per_dimension": None,
            "include_dimensions": ["COUNTRY"],
        }
    )

    # No text block: the complete result lives in structured content, with nulls omitted.
    assert tool_result.content == []
    assert tool_result.structured_content == {
        "datasetId": "IMF:CPI(1.0.0)",
        "found": True,
        # include_dimensions restricts the result to COUNTRY only.
        "dimensions": [
            {
                "id": "COUNTRY",
                "name": "Reference area",
                "totalAvailable": 2,
                "returned": 2,
                "truncated": False,
                "values": [
                    {"id": "USA", "name": "United States"},
                    {"id": "GBR", "name": "United Kingdom"},
                ],
            }
        ],
    }

    # The partial key is forwarded as an IN query to the dataset.
    forwarded_query = dataset.availability_query.call_args.args[0]
    assert forwarded_query.dimensions_queries_dict["SERIES"].values == ["51401"]
    assert forwarded_query.dimensions_queries_dict["SERIES"].operator == QueryOperator.IN


async def test_availability_query_truncates_and_reports_the_total():
    codes = [f"C{i}" for i in range(5)]
    country = _availability_dimension(
        "COUNTRY", "Reference area", {c: f"Country {c}" for c in codes}
    )
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={"COUNTRY": Query(values=codes, operator=QueryOperator.IN)}
    )
    dataset = _availability_dataset([country], result)

    structured = (
        await _build(_availability_tool_config(), _availability_inputs(dataset)).run(
            {"dataset_id": "IMF:CPI(1.0.0)", "partial_query": {}, "codes_per_dimension": 2}
        )
    ).structured_content

    assert structured is not None
    (dimension,) = structured["dimensions"]
    assert dimension["totalAvailable"] == 5
    assert dimension["returned"] == 2
    assert dimension["truncated"] is True
    assert [value["id"] for value in dimension["values"]] == ["C0", "C1"]


async def test_availability_query_surfaces_time_coverage():
    country = _availability_dimension("COUNTRY", "Reference area", {"USA": "United States"})
    time_dimension = SimpleNamespace(entity_id="TIME_PERIOD", name="Time period")
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={"COUNTRY": Query(values=["USA"], operator=QueryOperator.IN)},
        time_period_start="2000",
        time_period_end="2024",
    )
    dataset = _availability_dataset([country], result, time_dimension=time_dimension)

    structured = (
        await _build(_availability_tool_config(), _availability_inputs(dataset)).run(
            {"dataset_id": "IMF:CPI(1.0.0)", "partial_query": {}}
        )
    ).structured_content

    assert structured is not None
    assert structured["timeCoverage"] == {
        "dimensionId": "TIME_PERIOD",
        "name": "Time period",
        "start": "2000",
        "end": "2024",
    }


async def test_availability_query_not_found():
    tool_result = await _build(_availability_tool_config(), _availability_inputs(None)).run(
        {"dataset_id": "IMF:NOPE(1.0)"}
    )

    assert tool_result.content == []
    assert tool_result.structured_content == {
        "datasetId": "IMF:NOPE(1.0)",
        "found": False,
        "dimensions": [],
    }


async def test_availability_query_unknown_dimension_raises():
    country = _availability_dimension("COUNTRY", "Reference area", {"USA": "United States"})
    result = DataSetAvailabilityQuery()
    dataset = _availability_dataset([country], result)

    with pytest.raises(ToolError, match="Unknown dimension"):
        await _build(_availability_tool_config(), _availability_inputs(dataset)).run(
            {"dataset_id": "IMF:CPI(1.0.0)", "partial_query": {"NOPE": ["x"]}}
        )

    dataset.availability_query.assert_not_called()
