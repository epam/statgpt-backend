"""Tests for the discovery datasets channel configuration.

The block owns both halves of the feature, so what matters here is that the two are read
independently: `enabled` must gate the chat lookup without taking indexing down with it.
"""

import pytest
from pydantic import ValidationError

from statgpt.common.schemas import (
    ChannelConfig,
    DiscoveryPreFilterAxis,
    SupremeAgentConfig,
    ToolTypes,
)

_SUPREME_AGENT = SupremeAgentConfig(
    name="T", domain="D", terminology_domain="T", language_instructions=["i"]
)


def _config(
    application_id: str = "generic-rag-app",
    details: dict[str, object] | None = None,
    **overrides: object,
) -> ChannelConfig:
    block: dict[str, object] = {
        "type": "DISCOVERY_DATASETS",
        "name": "discovery_datasets",
        "description": "d",
        "details": {
            "applicationId": application_id,
            "templates": {"wrapper": "{items}", "item": "- {name}"},
            **(details or {}),
        },
    }
    block.update(overrides)
    return ChannelConfig.model_validate(
        {"supremeAgent": _SUPREME_AGENT, "discoveryDatasets": block}
    )


def test_a_configured_block_enables_both_halves() -> None:
    config = _config()

    assert config.is_discovery_lookup_available is True
    assert config.discovery_application_id == "generic-rag-app"


def test_disabling_the_block_stops_the_lookup_but_not_indexing() -> None:
    """So discovery data can be indexed before it is surfaced to users."""
    config = _config(enabled=False)

    assert config.is_discovery_lookup_available is False
    assert config.discovery_application_id == "generic-rag-app"


def test_the_application_id_resolves_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_RAG_APP", "grade-c-rag")

    assert _config("$env:{DISCOVERY_RAG_APP}").discovery_application_id == "grade-c-rag"


def test_the_block_is_never_offered_to_the_agent_or_mcp() -> None:
    """The lookup is run by the data query tool; nothing may call it as a tool."""
    config = _config()

    assert "discovery_datasets" not in config.tool_fields
    assert ToolTypes.DISCOVERY_DATASETS not in {tool.type for tool in config.tools}
    assert ToolTypes.DISCOVERY_DATASETS not in {tool.type for tool in config.agent_tools}


def test_templates_are_required() -> None:
    """There is nothing sensible to default the rendered block to."""
    with pytest.raises(ValidationError, match="templates"):
        ChannelConfig.model_validate(
            {
                "supremeAgent": _SUPREME_AGENT,
                "discoveryDatasets": {
                    "type": "DISCOVERY_DATASETS",
                    "name": "discovery_datasets",
                    "description": "d",
                    "details": {"applicationId": "generic-rag-app"},
                },
            }
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the pre-filter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_pre_filter_is_on_by_default_with_every_axis() -> None:
    """A channel that says nothing about it gets the narrowing, on every axis.

    Partner areas included: a record covering a country as a counterpart is still an answer to
    a query naming it, and the two area axes are alternatives rather than an extra requirement.
    """
    pre_filter = _config().discovery_datasets.details.pre_filter  # type: ignore[union-attr]

    assert pre_filter.enabled is True
    assert pre_filter.axes == list(DiscoveryPreFilterAxis)
    assert DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA in pre_filter.axes


def test_a_channel_without_a_vocabulary_has_no_reference_area_application() -> None:
    """The axis is then unavailable, which the pre-filter reports and works around."""
    assert _config().discovery_reference_area_application_id is None


def test_the_vocabulary_application_id_resolves_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_AREAS_APP", "areas-rag")
    config = _config(details={"referenceAreaApplicationId": "$env:{DISCOVERY_AREAS_APP}"})

    assert config.discovery_reference_area_application_id == "areas-rag"


def test_disabling_the_lookup_leaves_the_vocabulary_publishable() -> None:
    """Publishing is administrative; only the chat-time lookup is gated by `enabled`."""
    config = _config(details={"referenceAreaApplicationId": "areas-rag"}, enabled=False)

    assert config.is_discovery_lookup_available is False
    assert config.discovery_reference_area_application_id == "areas-rag"


def test_the_axes_can_be_narrowed_to_one() -> None:
    config = _config(details={"preFilter": {"axes": ["agency"], "referenceAreaTopN": 5}})
    pre_filter = config.discovery_datasets.details.pre_filter  # type: ignore[union-attr]

    assert pre_filter.axes == [DiscoveryPreFilterAxis.AGENCY]
    assert pre_filter.reference_area_top_n == 5


def test_an_unknown_axis_is_rejected() -> None:
    """A misspelled axis would otherwise silently narrow nothing."""
    with pytest.raises(ValidationError, match="axes"):
        _config(details={"preFilter": {"axes": ["reference_areas"]}})
