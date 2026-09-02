"""Contract tests for MCP tool output schemas.

Every App-MCP tool that emits ``structuredContent`` must declare an output schema, and the
content it returns at runtime must validate against that declared schema. These tests fail the
build when the two drift — the "validate in CI" requirement of the output-schema contract — and
double as the captured sample responses for the marketplace submission package.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jsonschema
import pandas as pd
import pytest

import statgpt.app.chains  # noqa: F401  # imported for its side effect: populate the tool registry
from statgpt.app.chains.tools import _TOOL_IMPLEMENTATIONS
from statgpt.app.mcp.output_schema import model_to_output_schema
from statgpt.app.mcp.provider import _McpToolAdapter
from statgpt.app.schemas.data_query_outcome import (
    DataQueryMcpPayload,
    DataQueryStatus,
    DataSetChoice,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.mcp import (
    AvailableDatasetsStructuredContent,
    AvailablePublicationsStructuredContent,
    AvailableTermsStructuredContent,
    DatasetComponentRecord,
    DatasetRecord,
    DatasetStructureStructuredContent,
    GlossaryDefinitionRecord,
    GlossaryTermRecord,
    PublicationTypeRecord,
    TermDefinitionsStructuredContent,
    TextToolStructuredContent,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.service import ChannelDatasetsMetadataResponse
from statgpt.app.schemas.tool_artifact import (
    DataQueryArtifact,
    DatasetsMetadataAppArtifact,
    SdmxQueryAppArtifact,
    ToolArtifact,
)
from statgpt.app.schemas.tool_states import ToolMessageState
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.query import JsonQueryMetadata, JsonQueryWithMetadata
from statgpt.common.schemas.tools import DataQueryTool

# Tools whose result is inherently prose (no machine-readable payload beyond the answer): they
# fall back to the text envelope. Every other registered tool declares a bespoke schema. Kept
# explicit so a tool that gains or loses a bespoke schema trips the completeness test below.
ENVELOPE_TOOL_TYPES = {
    ToolTypes.FILE_RAG,
    ToolTypes.WEB_SEARCH,
    ToolTypes.WEB_SEARCH_AGENT,
    ToolTypes.PLAIN_CONTENT,
    ToolTypes.DEEP_RESEARCH,
    ToolTypes.RESUME_DEEP_RESEARCH,
}


def _schema(tool_type: ToolTypes) -> dict:
    schema = _TOOL_IMPLEMENTATIONS[tool_type].get_mcp_output_schema()
    assert schema is not None, f"{tool_type} should declare an output schema"
    return schema


# ~~~~~~~~~~~~~ schema shape ~~~~~~~~~~~~~


def test_every_registered_tool_declares_a_valid_object_output_schema():
    # Every tool has an output schema, and it must be a well-formed JSON Schema of object type:
    # reviewers read it straight from the running server and the MCP host rejects non-object ones.
    for tool_type, tool_cls in _TOOL_IMPLEMENTATIONS.items():
        schema = tool_cls.get_mcp_output_schema()
        assert schema is not None, f"{tool_type} must declare an output schema"
        assert schema["type"] == "object", f"{tool_type} output schema must be an object"
        jsonschema.Draft202012Validator.check_schema(schema)


def test_only_prose_tools_fall_back_to_the_text_envelope():
    # Prose tools advertise the text envelope; every other tool declares a bespoke schema.
    text_envelope = model_to_output_schema(TextToolStructuredContent)
    envelope = {
        tool_type
        for tool_type, tool_cls in _TOOL_IMPLEMENTATIONS.items()
        if tool_cls.get_mcp_output_schema() == text_envelope
    }
    assert envelope == ENVELOPE_TOOL_TYPES


def test_data_query_schema_carries_a_version():
    # The structured content is a published contract; the version field lets clients detect the
    # schema revision, so it must stay part of the declared schema.
    assert "version" in _schema(ToolTypes.DATA_QUERY)["properties"]


# ~~~~~~~~~~~~~ runtime content validates against the declared schema ~~~~~~~~~~~~~


def _adapter(
    result, *, tool_type: ToolTypes, sdmx_query_app=SimpleNamespace(name="sdmx_query_app")
):
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(return_value=result))
    return _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        # out_of_scope=None disables the guardrail so run() proceeds straight to the tool.
        channel_config=SimpleNamespace(  # type: ignore[arg-type]
            out_of_scope=None, sdmx_query_app=sdmx_query_app
        ),
        # Only read for the data-query artifact branch; a DataQueryTool config satisfies it and is
        # ignored by the other branches.
        tool_config=DataQueryTool(name="fake_tool", description="Query data"),
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
        name="fake_tool",
        parameters={},
        output_schema=_schema(tool_type),
    )


async def _run_and_validate(adapter: _McpToolAdapter) -> dict:
    tool_result = await adapter.run({})
    assert tool_result.structured_content is not None
    # The declared schema (advertised in tools/list) must accept the content the tool returns.
    jsonschema.validate(instance=tool_result.structured_content, schema=adapter.output_schema)
    return tool_result.structured_content


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
    )


def _data_query_artifact(status: DataQueryStatus, **kwargs) -> DataQueryArtifact:
    return DataQueryArtifact.model_construct(state=SimpleNamespace(status=status), **kwargs)


DATA_QUERY_CASES = {
    DataQueryStatus.DATA_AVAILABLE: _data_query_artifact(
        DataQueryStatus.DATA_AVAILABLE,
        data_responses={"ds1": _data_response(pd.DataFrame({"REF_AREA": ["FR"], "2024": [1.5]}))},
    ),
    DataQueryStatus.NO_DATA: _data_query_artifact(DataQueryStatus.NO_DATA, data_responses={}),
    DataQueryStatus.NOT_EXECUTED: _data_query_artifact(
        DataQueryStatus.NOT_EXECUTED,
        data_responses={},
        mcp_payload=DataQueryMcpPayload(
            constructed_queries=[
                AppJsonQueryWithMetadata.from_common(_json_query("IMF:CPI(1.0.0)"))
            ]
        ),
    ),
    DataQueryStatus.DATASET_SELECTION_REQUIRED: _data_query_artifact(
        DataQueryStatus.DATASET_SELECTION_REQUIRED,
        data_responses={},
        mcp_payload=DataQueryMcpPayload(
            candidate_datasets=[DataSetChoice(id="IMF:CPI", name="Consumer Price Index")]
        ),
    ),
    DataQueryStatus.MISSING_DIMENSIONS: _data_query_artifact(
        DataQueryStatus.MISSING_DIMENSIONS,
        data_responses={},
        mcp_payload=DataQueryMcpPayload(
            missing_dimensions=MissingDimensionsInfo(dataset_id="IMF:CPI", dimensions=[])
        ),
    ),
    DataQueryStatus.INVALID_TIME_PERIOD: _data_query_artifact(
        DataQueryStatus.INVALID_TIME_PERIOD, data_responses={}
    ),
}


@pytest.mark.parametrize("status", sorted(DATA_QUERY_CASES, key=str))
async def test_data_query_structured_content_validates(status: DataQueryStatus):
    result = SimpleNamespace(content="Data query response.", artifact=DATA_QUERY_CASES[status])
    structured = await _run_and_validate(_adapter(result, tool_type=ToolTypes.DATA_QUERY))
    assert structured["status"] == status


async def test_sdmx_proxy_structured_content_validates():
    artifact = SdmxQueryAppArtifact.model_construct(
        status_code=200, content_type="application/json"
    )
    result = SimpleNamespace(content="<xml/>", artifact=artifact)
    structured = await _run_and_validate(_adapter(result, tool_type=ToolTypes.SDMX_QUERY_APP))
    assert structured == {"statusCode": 200, "contentType": "application/json"}


async def test_datasets_metadata_structured_content_validates():
    response = ChannelDatasetsMetadataResponse(
        deployment_id="dep", title="Channel", n_datasets=0, datasets=[]
    )
    artifact = DatasetsMetadataAppArtifact(
        state=ToolMessageState(type=ToolTypes.DATASETS_METADATA_APP), response=response
    )
    result = SimpleNamespace(content=response.model_dump_json(), artifact=artifact)
    structured = await _run_and_validate(
        _adapter(result, tool_type=ToolTypes.DATASETS_METADATA_APP)
    )
    assert structured["deployment_id"] == "dep"


# Representative structured content for each tool that attaches it to the artifact (the generic
# provider path). These double as captured sample responses for the submission package.
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
    ToolTypes.AVAILABLE_PUBLICATIONS: AvailablePublicationsStructuredContent(
        publication_types=[PublicationTypeRecord(name="Reports", description="Annual reports")],
        count=1,
    ),
    ToolTypes.AVAILABLE_DATASETS: AvailableDatasetsStructuredContent(
        datasets=[DatasetRecord(id="IMF:CPI(1.0.0)", name="Consumer Price Index")], count=1
    ),
    ToolTypes.DATASETS_METADATA: AvailableDatasetsStructuredContent(
        datasets=[DatasetRecord(id="IMF:CPI(1.0.0)", name="Consumer Price Index")], count=1
    ),
    ToolTypes.DATASET_STRUCTURE: DatasetStructureStructuredContent(
        dataset_id="IMF:CPI(1.0.0)",
        found=True,
        name="Consumer Price Index",
        dimensions=[DatasetComponentRecord(id="REF_AREA", name="Reference area")],
        attributes=[DatasetComponentRecord(id="UNIT_MULT", name="Unit multiplier")],
    ),
}


@pytest.mark.parametrize("tool_type", sorted(GENERIC_CASES, key=str))
async def test_tool_built_structured_content_validates(tool_type: ToolTypes):
    # Tools that build their own structured content attach it to the artifact; the provider
    # surfaces it and it must validate against the tool's declared schema.
    artifact = ToolArtifact(
        state=ToolMessageState(type=tool_type), mcp_structured=GENERIC_CASES[tool_type]
    )
    result = SimpleNamespace(content="Tool response.", artifact=artifact)
    await _run_and_validate(_adapter(result, tool_type=tool_type))


async def test_prose_tool_returns_the_text_envelope():
    # A prose tool builds no structured content; the provider mirrors its text into the envelope,
    # which must validate against the declared (envelope) schema.
    artifact = ToolArtifact(state=ToolMessageState(type=ToolTypes.FILE_RAG))
    result = SimpleNamespace(content="Here is the answer.", artifact=artifact)
    structured = await _run_and_validate(_adapter(result, tool_type=ToolTypes.FILE_RAG))
    assert structured == {"text": "Here is the answer."}


async def test_bespoke_tool_without_structured_content_omits_it():
    # A bespoke-schema tool that builds no structured content must NOT be handed the text envelope
    # (that would violate its declared schema); the provider leaves structuredContent unset.
    artifact = ToolArtifact(state=ToolMessageState(type=ToolTypes.AVAILABLE_DATASETS))
    result = SimpleNamespace(content="Some prose.", artifact=artifact)
    tool_result = await _adapter(result, tool_type=ToolTypes.AVAILABLE_DATASETS).run({})
    assert tool_result.structured_content is None
