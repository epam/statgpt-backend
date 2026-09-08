from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.common.schemas.tool_details import AvailableTermsDetails, TermDefinitionsDetails
from statgpt.common.schemas.tools import AvailableTermsTool, TermDefinitionsTool

_GDP = SimpleNamespace(term="GDP", domain="Economy", source="IMF", definition="Gross ...")
_CPI = SimpleNamespace(term="CPI", domain="Prices", source="IMF", definition="Consumer ...")
# A term the glossary has no domain/source for: upstream they are non-optional strings, so the
# missing value arrives as "".
_PPP = SimpleNamespace(term="PPP", domain="", source="", definition="Purchasing ...")


def _inputs(terms: list) -> tuple[dict, AsyncMock]:
    get_available_terms = AsyncMock(return_value=terms)
    data_service = SimpleNamespace(get_available_terms=get_available_terms)
    return {ChainParametersConfig.DATA_SERVICE: data_service}, get_available_terms


def _build(tool_config, inputs: dict) -> StatGptMcpTool:
    return StatGptMcpTool.from_config(
        tool_config,
        # out_of_scope=None disables the guardrail, so run() proceeds straight to the tool.
        SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=""), out_of_scope=None),  # type: ignore[arg-type]
        inputs=inputs,
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
    )


# ~~~~~~~~~~~~~ available terms ~~~~~~~~~~~~~


async def test_available_terms_returns_records_only():
    inputs, _ = _inputs([_GDP, _CPI, _PPP])
    tool_config = AvailableTermsTool(
        name="terms",
        description="Terms.",
        details=AvailableTermsDetails(include_domain=True, include_source=False),
    )

    tool_result = await _build(tool_config, inputs).run({})

    # Structured-only: the text rendering would only duplicate the records.
    assert tool_result.content == []
    # `source` is dropped because the tool is not configured to expose it, `PPP`'s `domain` because
    # the glossary has no value for it.
    assert tool_result.structured_content == {
        "terms": [
            {"term": "GDP", "domain": "Economy"},
            {"term": "CPI", "domain": "Prices"},
            {"term": "PPP"},
        ],
        "count": 3,
    }


# ~~~~~~~~~~~~~ term definitions ~~~~~~~~~~~~~


def _definitions_config(limit: int | None = None) -> TermDefinitionsTool:
    return TermDefinitionsTool(
        name="definitions", description="Definitions.", details=TermDefinitionsDetails(limit=limit)
    )


async def test_term_definitions_separates_found_and_missing_terms():
    inputs, _ = _inputs([_GDP, _PPP])

    tool_result = await _build(_definitions_config(), inputs).run(
        {"terms": ["gdp ", "PPP", "unknown"]}
    )

    # Structured-only, and a found term needs no `found` flag. `PPP` carries neither domain nor
    # source because the glossary has no value for them.
    assert tool_result.content == []
    assert tool_result.structured_content == {
        "definitions": [
            {"term": "GDP", "definition": "Gross ...", "domain": "Economy", "source": "IMF"},
            {"term": "PPP", "definition": "Purchasing ..."},
        ],
        "notFound": ["unknown"],
    }


async def test_term_definitions_omits_not_found_when_every_term_resolves():
    inputs, _ = _inputs([_GDP])

    tool_result = await _build(_definitions_config(), inputs).run({"terms": ["GDP"]})

    assert tool_result.structured_content is not None
    assert "notFound" not in tool_result.structured_content


async def test_term_definitions_over_limit_raises():
    # An over-limit request fetches nothing and must be retried with fewer terms, so it fails
    # rather than returning an empty result that would read as "none of these terms exist".
    inputs, get_available_terms = _inputs([_GDP])

    with pytest.raises(ToolError, match="exceeds the limit of 1"):
        await _build(_definitions_config(limit=1), inputs).run({"terms": ["GDP", "CPI"]})

    get_available_terms.assert_not_called()


def test_term_definitions_schema_spells_out_the_limit():
    inputs, _ = _inputs([])

    tool = _build(_definitions_config(limit=5), inputs)

    assert "limited to 5" in tool.parameters["properties"]["terms"]["description"]
    assert "inputs" not in tool.parameters["properties"]
