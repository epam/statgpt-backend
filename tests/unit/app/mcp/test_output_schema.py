"""Contract tests for MCP tool output schemas.

Six tools declare an MCP output schema: the two glossary tools (available-terms,
term-definitions), the three dataset-metadata tools (available-datasets, dataset-structure,
availability-query) and the data query tool. The first five are structured-only — they carry their
whole result in `structuredContent`, repeated as its JSON text — while the data query tool also
returns resources. Every other tool opts out (no declared schema), even where it emits structured
content.
These tests fail the build when a scoped tool's runtime content drifts from its declared schema, and
lock the reduced scope so a tool cannot silently gain or lose a schema. They double as the captured
sample responses for the marketplace submission package.
"""

import json

import jsonschema
import pytest
from fastmcp.tools import ToolResult

from statgpt.app.mcp.tools import StatGptMcpTool, mcp_tool_class_for
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.mcp import (
    AvailabilityDimensionRecord,
    AvailabilityStructuredContent,
    AvailabilityValueRecord,
    AvailableDatasetsStructuredContent,
    AvailableTermsStructuredContent,
    CandidateDatasetRecord,
    DataQueryStructuredContent,
    DatasetComponentRecord,
    DatasetRecord,
    DatasetStructureStructuredContent,
    DatasetValueRecord,
    ExecutionResult,
    FilterValue,
    GlossaryDefinitionRecord,
    GlossaryTermRecord,
    InvalidPeriodRecord,
    PeriodRange,
    ProviderRecord,
    QueryExecution,
    QueryFilter,
    QueryRecord,
    RequestedPeriod,
    TermDefinitionsStructuredContent,
    TimeCoverageRecord,
)
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.query import JsonQueryOperator

# The only tools in scope for explicit MCP output schemas. Every other tool opts out.
SCOPED_TOOL_TYPES = {
    ToolTypes.AVAILABLE_TERMS,
    ToolTypes.TERM_DEFINITIONS,
    ToolTypes.AVAILABLE_DATASETS,
    ToolTypes.DATASET_STRUCTURE,
    ToolTypes.AVAILABILITY_QUERY,
    ToolTypes.DATA_QUERY,
}
# The scoped tools whose whole result is the structured content.
STRUCTURED_ONLY_TOOL_TYPES = SCOPED_TOOL_TYPES - {ToolTypes.DATA_QUERY}


def _schema(tool_type: ToolTypes) -> dict:
    schema = mcp_tool_class_for(tool_type).get_output_schema()
    assert schema is not None, f"{tool_type} should declare an output schema"
    return schema


# ~~~~~~~~~~~~~ schema shape ~~~~~~~~~~~~~


def test_only_scoped_tools_declare_an_output_schema():
    # Exactly the four scoped tools declare a schema; every other tool opts out (returns None). Kept
    # explicit so a tool that gains or loses a schema trips this test.
    declaring = {
        tool_type
        for tool_type in ToolTypes
        if mcp_tool_class_for(tool_type).get_output_schema() is not None
    }
    assert declaring == SCOPED_TOOL_TYPES


def test_declared_schemas_are_valid_object_schemas():
    # A declared schema must be a well-formed JSON Schema of object type: reviewers read it straight
    # from the running server and the MCP host rejects non-object ones.
    for tool_type in SCOPED_TOOL_TYPES:
        schema = _schema(tool_type)
        assert schema["type"] == "object", f"{tool_type} output schema must be an object"
        jsonschema.Draft202012Validator.check_schema(schema)


# ~~~~~~~~~~~~~ runtime content validates against the declared schema ~~~~~~~~~~~~~

# Representative structured content for each scoped tool. These double as captured sample responses
# for the submission package.
GENERIC_CASES: dict = {
    ToolTypes.AVAILABLE_TERMS: AvailableTermsStructuredContent(
        terms=[GlossaryTermRecord(term="GDP", domain="Economy", source="IMF")], count=1
    ),
    ToolTypes.TERM_DEFINITIONS: TermDefinitionsStructuredContent(
        definitions=[
            GlossaryDefinitionRecord(
                term="GDP", definition="Gross ...", domain="Economy", source="IMF"
            )
        ],
        not_found=["unknown"],
    ),
    ToolTypes.AVAILABLE_DATASETS: AvailableDatasetsStructuredContent(
        providers=[ProviderRecord(name="IMF", dataset_count=1)],
        datasets=[
            DatasetRecord(
                id="IMF:CPI(1.0.0)",
                name="Consumer Price Index",
                description="Prices.",
                provider="IMF",
                last_updated="2024-01-31",
                url="https://example.org/imf-cpi",
                number_of_indicators=42,
            )
        ],
        total_datasets=1,
        total_indicators=42,
        total_agencies=1,
    ),
    ToolTypes.DATASET_STRUCTURE: DatasetStructureStructuredContent(
        dataset_id="IMF:CPI(1.0.0)",
        name="Consumer Price Index",
        last_updated="2024-01-31",
        dimensions=[
            DatasetComponentRecord(
                id="REF_AREA",
                name="Reference area",
                type="category",
                total_values=200,
                sample_values=[DatasetValueRecord(id="US", name="United States")],
            )
        ],
        attributes=[DatasetComponentRecord(id="UNIT_MULT", name="Unit multiplier", type="string")],
    ),
    ToolTypes.AVAILABILITY_QUERY: AvailabilityStructuredContent(
        dataset_id="IMF:CPI(1.0.0)",
        found=True,
        dimensions=[
            AvailabilityDimensionRecord(
                id="REF_AREA",
                name="Reference area",
                total_available=2,
                returned=2,
                truncated=False,
                values=[
                    AvailabilityValueRecord(id="US", name="United States"),
                    AvailabilityValueRecord(id="GB", name="United Kingdom"),
                ],
            )
        ],
        time_coverage=TimeCoverageRecord(
            dimension_id="TIME_PERIOD", name="Time period", start="2000", end="2024"
        ),
    ),
    ToolTypes.DATA_QUERY: DataQueryStructuredContent(
        status=DataQueryStatus.DATA_AVAILABLE,
        message="Tell the user where the data comes from.",
        executed_at="2026-09-23T10:00:00+00:00",
        queries=[
            QueryRecord(
                query_id="dq_ab12cd34ef",
                dataset_urn="IMF:CPI(1.0.0)",
                dataset_name="Consumer Price Index",
                is_official=True,
                provider="IMF",
                last_updated="2026-09-01",
                dataset_url="https://data.imf.org/en/datasets/IMF:CPI",
                query_summary="Consumer prices in the United States from 2020 to 2024.",
                executed=True,
                filters=[
                    QueryFilter(
                        dimension_id="REF_AREA",
                        dimension_name="Reference area",
                        operator=JsonQueryOperator.IN,
                        values=[FilterValue(id="US", name="United States")],
                    ),
                    QueryFilter(
                        dimension_id="INDICATOR",
                        dimension_name="Indicator",
                        operator=JsonQueryOperator.IN,
                        values=[FilterValue(id="CPI", name="Consumer price index")],
                        is_indicator=True,
                        is_default=True,
                    ),
                ],
                requested_period=RequestedPeriod(
                    start_period="2020-01-01", end_period="2024-12-31", is_default=True
                ),
                invalid_period=InvalidPeriodRecord(
                    rejected_bound="endPeriod",
                    requested_value="2030",
                    available_period=PeriodRange(start_period="2000", end_period="2024"),
                ),
                factual_period=PeriodRange(start_period="2020", end_period="2024"),
                series_count=1,
                execution=QueryExecution(
                    result=ExecutionResult.PARTIALLY_PARSED,
                    reason="Some of the data could not be parsed.",
                    advice="Tell the user that the data is incomplete.",
                ),
                data_explorer_url="https://data.imf.org/en/Data-Explorer?datasetUrn=IMF:CPI",
            )
        ],
        candidate_datasets=[
            CandidateDatasetRecord(
                id="IMF:CPI(1.0.0)",
                name="Consumer Price Index",
                is_official=True,
                query=QueryRecord(
                    query_id="dq_0011223344", dataset_urn="IMF:CPI(1.0.0)", executed=False
                ),
            )
        ],
    ),
}


def _structured_content(tool_type: ToolTypes) -> dict:
    # Serialized the way the tools serialize it: by alias, with the null fields dropped.
    return GENERIC_CASES[tool_type].model_dump(mode="json", by_alias=True, exclude_none=True)


def _tool_result(tool_type: ToolTypes) -> ToolResult:
    return StatGptMcpTool._structured_only(GENERIC_CASES[tool_type])


@pytest.mark.parametrize("tool_type", sorted(GENERIC_CASES, key=str))
def test_structured_content_validates_against_the_declared_schema(tool_type: ToolTypes):
    # The model each tool declares must serialize to something its own schema accepts.
    assert GENERIC_CASES[tool_type].__class__ is mcp_tool_class_for(tool_type).get_output_model()
    jsonschema.validate(instance=_structured_content(tool_type), schema=_schema(tool_type))


def test_scoped_tools_cover_the_generic_cases():
    # The captured samples must stay in lockstep with the scoped tools.
    assert set(GENERIC_CASES) == SCOPED_TOOL_TYPES


@pytest.mark.parametrize("tool_type", sorted(STRUCTURED_ONLY_TOOL_TYPES, key=str))
def test_structured_only_tool_repeats_the_payload_as_json_text(tool_type: ToolTypes):
    # A client that reads only `content` receives the same payload as `structuredContent`.
    result = _tool_result(tool_type)
    assert [block.type for block in result.content] == ["text"]
    assert json.loads(result.content[0].text) == _structured_content(tool_type)


def test_structured_only_tool_omits_null_fields():
    content = AvailableDatasetsStructuredContent(
        datasets=[DatasetRecord(id="IMF:CPI(1.0.0)", name="CPI")],
        total_datasets=1,
        total_agencies=0,
    )
    structured = StatGptMcpTool._structured_only(content).structured_content
    assert structured is not None
    assert structured["datasets"] == [{"id": "IMF:CPI(1.0.0)", "name": "CPI"}]
    assert "totalIndicators" not in structured
    jsonschema.validate(instance=structured, schema=_schema(ToolTypes.AVAILABLE_DATASETS))
