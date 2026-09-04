from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.common.schemas.tool_details import AvailableDatasetsDetails
from statgpt.common.schemas.tools import AvailableDatasetsTool, DatasetStructureTool

_AUTH = SimpleNamespace()


def _dataset(
    source_id: str = "IMF:CPI(1.0.0)",
    entity_id: str = "cpi",
    provider: str | None = "IMF",
    updated_at: datetime | None = None,
) -> SimpleNamespace:
    citation = (
        SimpleNamespace(
            description=None,
            provider=provider,
            provider_agency_names_with_fallback_to_provider=[provider],
            provider_agencies=None,
            last_updated="2023",
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
        dimensions=lambda: [],
        attributes=lambda: [],
    )


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
                "lastUpdated": "2023",
            },
            {"id": "WB:GDP(1.0)", "name": "Consumer Price Index", "description": "Prices."},
        ],
        "totalDatasets": 2,
        "totalAgencies": 1,
    }


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
