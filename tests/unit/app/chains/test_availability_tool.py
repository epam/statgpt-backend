import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from statgpt.app.chains.datasets_meta.availability_tool import (
    AvailabilityQueryArgs,
    AvailabilityQueryTool,
)
from statgpt.app.config import ChainParametersConfig
from statgpt.common.data.base import CategoricalDimension, DataSetAvailabilityQuery, Query
from statgpt.common.data.base.enums import QueryOperator
from statgpt.common.schemas import AvailabilityQueryTool as AvailabilityQueryToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.availability_query_tool import AvailabilityQueryToolDetails

_AUTH = SimpleNamespace()


def _categorical_dimension(entity_id: str, name: str, names_by_id: dict[str, str]) -> MagicMock:
    # A spec'd mock passes the `isinstance(..., CategoricalDimension)` check the tool relies on.
    dim = MagicMock(spec=CategoricalDimension)
    dim.entity_id = entity_id
    dim.name = name
    dim.name_by_query_id.side_effect = lambda code: names_by_id.get(code)
    return dim


def _dataset(
    dimensions: list,
    availability_result: DataSetAvailabilityQuery,
    time_dimension=None,
) -> SimpleNamespace:
    dims_by_id = {d.entity_id: d for d in dimensions}
    return SimpleNamespace(
        source_id="IMF:CPI(1.0.0)",
        dimensions=lambda: dimensions,
        dimension=lambda dim_id: dims_by_id[dim_id],
        get_time_dimension=lambda: time_dimension,
        availability_query=AsyncMock(return_value=availability_result),
    )


def _inputs(dataset) -> dict:
    data_service = SimpleNamespace(get_dataset_by_source_id=AsyncMock(return_value=dataset))
    return {
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.AUTH_CONTEXT: _AUTH,
        ChainParametersConfig.CHOICE: None,
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.TARGET: None,
    }


def _tool(hard_limit: int = 500) -> AvailabilityQueryTool:
    config = AvailabilityQueryToolConfig(
        name="Availability_Query",
        description="Availability.",
        details=AvailabilityQueryToolDetails(hard_limit=hard_limit),
    )
    return AvailabilityQueryTool.from_config(config, SimpleNamespace())  # type: ignore[arg-type]


# ~~~~~~~~~~~~~ args schema ~~~~~~~~~~~~~


def test_args_apply_the_documented_defaults():
    args = AvailabilityQueryArgs(inputs={}, dataset_id="IMF:CPI(1.0.0)")
    assert args.partial_query == {}
    assert args.codes_per_dimension == 10
    assert args.include_dimensions is None


def test_public_schema_hides_the_injected_inputs_field():
    schema = AvailabilityQueryArgs.get_public_schema()
    assert "inputs" not in schema["properties"]
    assert "dataset_id" in schema["properties"]


# ~~~~~~~~~~~~~ _arun ~~~~~~~~~~~~~


async def test_partial_query_returns_named_codes_for_the_included_dimension():
    country = _categorical_dimension(
        "COUNTRY", "Reference area", {"USA": "United States", "GBR": "United Kingdom"}
    )
    freq = _categorical_dimension("FREQ", "Frequency", {"A": "Annual"})
    series = _categorical_dimension("SERIES", "Series", {"51401": "Series 51401"})
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={
            "COUNTRY": Query(values=["USA", "GBR"], operator=QueryOperator.IN),
            "FREQ": Query(values=["A"], operator=QueryOperator.IN),
        }
    )
    dataset = _dataset([series, country, freq], result)

    content, artifact = await _tool()._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={"SERIES": ["51401"]},
        codes_per_dimension=None,
        include_dimensions=["COUNTRY"],
    )

    payload = json.loads(content)
    assert payload["dataset_id"] == "IMF:CPI(1.0.0)"
    # include_dimensions restricts the result to COUNTRY only.
    assert set(payload["dimensions"]) == {"COUNTRY"}
    assert payload["dimensions"]["COUNTRY"] == {
        "name": "Reference area",
        "total_available": 2,
        "returned": 2,
        "truncated": False,
        "values": [
            {"id": "USA", "name": "United States"},
            {"id": "GBR", "name": "United Kingdom"},
        ],
    }
    assert artifact.state.type == ToolTypes.AVAILABILITY_QUERY

    # The partial key is forwarded as an IN query to the dataset.
    forwarded_query = dataset.availability_query.call_args.args[0]
    assert forwarded_query.dimensions_queries_dict["SERIES"].values == ["51401"]
    assert forwarded_query.dimensions_queries_dict["SERIES"].operator == QueryOperator.IN


async def test_codes_per_dimension_truncates_and_reports_the_total():
    codes = [f"C{i}" for i in range(5)]
    country = _categorical_dimension(
        "COUNTRY", "Reference area", {c: f"Country {c}" for c in codes}
    )
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={"COUNTRY": Query(values=codes, operator=QueryOperator.IN)}
    )
    dataset = _dataset([country], result)

    content, _ = await _tool()._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={},
        codes_per_dimension=2,
    )

    entry = json.loads(content)["dimensions"]["COUNTRY"]
    assert entry["total_available"] == 5
    assert entry["returned"] == 2
    assert entry["truncated"] is True
    assert [v["id"] for v in entry["values"]] == ["C0", "C1"]


async def test_hard_limit_caps_an_unbounded_request():
    codes = [f"C{i}" for i in range(5)]
    country = _categorical_dimension(
        "COUNTRY", "Reference area", {c: f"Country {c}" for c in codes}
    )
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={"COUNTRY": Query(values=codes, operator=QueryOperator.IN)}
    )
    dataset = _dataset([country], result)

    content, _ = await _tool(hard_limit=3)._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={},
        codes_per_dimension=None,
    )

    entry = json.loads(content)["dimensions"]["COUNTRY"]
    assert entry["returned"] == 3
    assert entry["truncated"] is True


async def test_time_coverage_is_surfaced_when_present():
    country = _categorical_dimension("COUNTRY", "Reference area", {"USA": "United States"})
    time_dimension = SimpleNamespace(entity_id="TIME_PERIOD", name="Time period")
    result = DataSetAvailabilityQuery(
        dimensions_queries_dict={"COUNTRY": Query(values=["USA"], operator=QueryOperator.IN)},
        time_period_start="2000",
        time_period_end="2024",
    )
    dataset = _dataset([country], result, time_dimension=time_dimension)

    content, _ = await _tool()._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={},
    )

    payload = json.loads(content)
    assert payload["time_coverage"] == {
        "dimension_id": "TIME_PERIOD",
        "name": "Time period",
        "start": "2000",
        "end": "2024",
    }


async def test_unknown_dimension_id_yields_a_helpful_error():
    country = _categorical_dimension("COUNTRY", "Reference area", {"USA": "United States"})
    result = DataSetAvailabilityQuery()
    dataset = _dataset([country], result)

    content, _ = await _tool()._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={"NOPE": ["x"]},
    )

    assert "Unknown dimension" in content
    assert "COUNTRY" in content
    dataset.availability_query.assert_not_called()


async def test_unknown_dimension_value_yields_a_helpful_error():
    freq = _categorical_dimension("FREQ", "Frequency", {"A": "Annual"})
    result = DataSetAvailabilityQuery()
    dataset = _dataset([freq], result)

    content, _ = await _tool()._arun(
        inputs=_inputs(dataset),
        dataset_id="IMF:CPI(1.0.0)",
        partial_query={"FREQ": ["NA"]},
    )

    assert "Unknown code value" in content
    assert "FREQ" in content
    assert "NA" in content
    dataset.availability_query.assert_not_called()


async def test_missing_dataset_returns_a_not_found_message():
    content, artifact = await _tool()._arun(
        inputs=_inputs(None),
        dataset_id="IMF:NOPE(1.0)",
    )

    assert "not found" in content
    assert artifact.state.type == ToolTypes.AVAILABILITY_QUERY
