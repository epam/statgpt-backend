"""Tests for the discovery datasets channel configuration.

The block owns both halves of the feature, so what matters here is that the two are read
independently: `enabled` must gate the chat lookup without taking indexing down with it.
"""

import pytest
from pydantic import ValidationError

from statgpt.common.schemas import ChannelConfig, SupremeAgentConfig, ToolTypes

_SUPREME_AGENT = SupremeAgentConfig(
    name="T", domain="D", terminology_domain="T", language_instructions=["i"]
)


def _config(application_id: str = "generic-rag-app", **overrides: object) -> ChannelConfig:
    block: dict[str, object] = {
        "type": "DISCOVERY_DATASETS",
        "name": "discovery_datasets",
        "description": "d",
        "details": {
            "applicationId": application_id,
            "templates": {"wrapper": "{items}", "item": "- {name}"},
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
