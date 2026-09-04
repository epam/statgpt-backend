"""Contract tests for MCP tool output schemas.

Only four tools declare an MCP output schema: the two glossary tools (available-terms,
term-definitions) and the two dataset-metadata tools (available-datasets, dataset-structure). Every
other tool opts out (no declared schema), even where it emits structured content. These tests fail the
build when a scoped tool's runtime content drifts from its declared schema, and lock the reduced
scope so a tool cannot silently gain or lose a schema. They double as the captured sample responses
for the marketplace submission package.
"""

import jsonschema
import pytest
from fastmcp.tools import ToolResult

from statgpt.app.mcp.tools import StatGptMcpTool, mcp_tool_class_for
from statgpt.app.schemas.mcp import (
    AvailableDatasetsStructuredContent,
    AvailableTermsStructuredContent,
    DatasetComponentRecord,
    DatasetRecord,
    DatasetStructureStructuredContent,
    DatasetValueRecord,
    GlossaryDefinitionRecord,
    GlossaryTermRecord,
    ProviderAgencyRecord,
    ProviderRecord,
    TermDefinitionsStructuredContent,
)
from statgpt.common.schemas import ToolTypes

# The only tools in scope for explicit MCP output schemas. Every other tool opts out.
SCOPED_TOOL_TYPES = {
    ToolTypes.AVAILABLE_TERMS,
    ToolTypes.TERM_DEFINITIONS,
    ToolTypes.AVAILABLE_DATASETS,
    ToolTypes.DATASET_STRUCTURE,
}

# Of the scoped tools, these carry their full result in structured content: the MCP response drops
# the text block and omits null optional fields.
STRUCTURED_ONLY_TOOL_TYPES = {ToolTypes.AVAILABLE_DATASETS, ToolTypes.DATASET_STRUCTURE}


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
                term="GDP", found=True, domain="Economy", source="IMF", definition="Gross ..."
            ),
            GlossaryDefinitionRecord(term="unknown", found=False),
        ]
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
        found=True,
        name="Consumer Price Index",
        description="Prices.",
        provider="IMF",
        last_updated="2024-01-31",
        url="https://example.org/imf-cpi",
        provider_agencies=[ProviderAgencyRecord(id="IMF", name="Intl Monetary Fund")],
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
}


def _tool_result(tool_type: ToolTypes) -> ToolResult:
    # The two ways an MCP tool emits structured content: as the whole result (structured-only) or
    # alongside the text rendering.
    model = GENERIC_CASES[tool_type]
    if tool_type in STRUCTURED_ONLY_TOOL_TYPES:
        return StatGptMcpTool._structured_only(model)
    return ToolResult(
        content=StatGptMcpTool._text_content("Tool response."), structured_content=model
    )


@pytest.mark.parametrize("tool_type", sorted(GENERIC_CASES, key=str))
def test_structured_content_validates_against_the_declared_schema(tool_type: ToolTypes):
    # The model each tool declares must serialize to something its own schema accepts.
    assert GENERIC_CASES[tool_type].__class__ is mcp_tool_class_for(tool_type).get_output_model()
    tool_result = _tool_result(tool_type)
    assert tool_result.structured_content is not None
    jsonschema.validate(instance=tool_result.structured_content, schema=_schema(tool_type))


def test_scoped_tools_cover_the_generic_cases():
    # The captured samples must stay in lockstep with the scoped tools.
    assert set(GENERIC_CASES) == SCOPED_TOOL_TYPES


@pytest.mark.parametrize("tool_type", sorted(STRUCTURED_ONLY_TOOL_TYPES, key=str))
def test_structured_only_tool_drops_text(tool_type: ToolTypes):
    # A structured-only tool returns only structuredContent: no text content block is emitted.
    assert _tool_result(tool_type).content == []


def test_structured_only_tool_omits_null_fields():
    # Null optional fields are dropped from the structured content of a structured-only tool.
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
