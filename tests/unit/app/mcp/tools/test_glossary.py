from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp.types import TextContent

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.common.schemas.tool_details import AvailableTermsDetails, TermDefinitionsDetails
from statgpt.common.schemas.tools import AvailableTermsTool, TermDefinitionsTool

_GDP = SimpleNamespace(term="GDP", domain="Economy", source="IMF", definition="Gross ...")
_CPI = SimpleNamespace(term="CPI", domain="Prices", source="IMF", definition="Consumer ...")


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


async def test_available_terms_returns_markdown_and_records():
    inputs, _ = _inputs([_GDP, _CPI])
    tool_config = AvailableTermsTool(
        name="terms",
        description="Terms.",
        details=AvailableTermsDetails(include_domain=True, include_source=False),
    )

    tool_result = await _build(tool_config, inputs).run({})

    assert tool_result.content == [
        TextContent(
            type="text",
            text=(
                "Glossary contains 2 terms.\n\n*List of available glossary terms:*\n"
                "- **GDP**, domain: Economy\n- **CPI**, domain: Prices"
            ),
        )
    ]
    # Domain/source are exposed only when configured, mirroring the text rendering.
    assert tool_result.structured_content == {
        "terms": [
            {"term": "GDP", "domain": "Economy", "source": None},
            {"term": "CPI", "domain": "Prices", "source": None},
        ],
        "count": 2,
    }


# ~~~~~~~~~~~~~ term definitions ~~~~~~~~~~~~~


def _definitions_config(limit: int | None = None) -> TermDefinitionsTool:
    return TermDefinitionsTool(
        name="definitions", description="Definitions.", details=TermDefinitionsDetails(limit=limit)
    )


async def test_term_definitions_flags_found_and_missing_terms():
    inputs, _ = _inputs([_GDP])

    tool_result = await _build(_definitions_config(), inputs).run({"terms": ["gdp ", "unknown"]})

    assert tool_result.structured_content == {
        "definitions": [
            {
                "term": "GDP",
                "found": True,
                "domain": "Economy",
                "source": "IMF",
                "definition": "Gross ...",
            },
            {"term": "unknown", "found": False, "domain": None, "source": None, "definition": None},
        ]
    }
    text = tool_result.content[0]
    assert isinstance(text, TextContent)
    assert "### GDP" in text.text
    assert "The term is not available in the glossary." in text.text


async def test_term_definitions_over_limit_fetches_nothing():
    # Structured content still matches the declared schema (no definitions); the reason lives in
    # the text rendering.
    inputs, get_available_terms = _inputs([_GDP])

    tool_result = await _build(_definitions_config(limit=1), inputs).run({"terms": ["GDP", "CPI"]})

    get_available_terms.assert_not_called()
    assert tool_result.structured_content == {"definitions": []}
    text = tool_result.content[0]
    assert isinstance(text, TextContent)
    assert "exceeds the limit of 1" in text.text


def test_term_definitions_schema_spells_out_the_limit():
    inputs, _ = _inputs([])

    tool = _build(_definitions_config(limit=5), inputs)

    assert "limited to 5" in tool.parameters["properties"]["terms"]["description"]
    assert "inputs" not in tool.parameters["properties"]
