from fastmcp.tools import ToolResult
from pydantic import PrivateAttr

from statgpt.app.chains.glossary_tools import (
    AvailableTermsRunner,
    BaseTermDefinitionsArgs,
    TermDefinitionsOutcome,
    TermDefinitionsRunner,
    build_term_definitions_args,
)
from statgpt.app.chains.tools import ToolArgs
from statgpt.app.schemas.mcp import (
    AvailableTermsStructuredContent,
    GlossaryDefinitionRecord,
    GlossaryTermRecord,
    TermDefinitionsStructuredContent,
)
from statgpt.common import schemas
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.tools import AvailableTermsTool as AvailableTermsToolConfig
from statgpt.common.schemas.tools import TermDefinitionsTool as TermDefinitionsToolConfig

from .base import StatGptMcpTool


def available_terms_structured_content(
    terms: list[schemas.GlossaryTerm], *, include_domain: bool, include_source: bool
) -> AvailableTermsStructuredContent:
    # Expose domain/source only when the tool is configured to (mirrors the text rendering).
    records = [
        GlossaryTermRecord(
            term=term.term,
            domain=term.domain if include_domain else None,
            source=term.source if include_source else None,
        )
        for term in terms
    ]
    return AvailableTermsStructuredContent(terms=records, count=len(terms))


def term_definitions_structured_content(
    outcome: TermDefinitionsOutcome,
) -> TermDefinitionsStructuredContent:
    # Over-limit: still matches the declared schema (no definitions); the reason lives in the text.
    records = [
        (
            GlossaryDefinitionRecord(
                term=lookup.found.term,
                found=True,
                domain=lookup.found.domain,
                source=lookup.found.source,
                definition=lookup.found.definition,
            )
            if lookup.found is not None
            else GlossaryDefinitionRecord(term=lookup.requested, found=False)
        )
        for lookup in outcome.lookups
    ]
    return TermDefinitionsStructuredContent(definitions=records)


class AvailableTermsMcpTool(
    StatGptMcpTool[AvailableTermsToolConfig, ToolArgs], tool_type=ToolTypes.AVAILABLE_TERMS
):
    _runner: AvailableTermsRunner = PrivateAttr()

    def __init__(
        self, tool_config: AvailableTermsToolConfig, channel_config, inputs, auth_context, **kwargs
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._runner = AvailableTermsRunner(tool_config.details)

    @classmethod
    def get_output_model(cls) -> type[AvailableTermsStructuredContent]:
        return AvailableTermsStructuredContent

    async def _execute(self, args: ToolArgs) -> ToolResult:
        terms = await self._runner.run(args.inputs)
        details = self._tool_config.details
        return ToolResult(
            content=self._text_content(self._runner.to_markdown(terms)),
            structured_content=available_terms_structured_content(
                terms, include_domain=details.include_domain, include_source=details.include_source
            ),
        )


class TermDefinitionsMcpTool(
    StatGptMcpTool[TermDefinitionsToolConfig, BaseTermDefinitionsArgs],
    tool_type=ToolTypes.TERM_DEFINITIONS,
):
    _runner: TermDefinitionsRunner = PrivateAttr()

    def __init__(
        self, tool_config: TermDefinitionsToolConfig, channel_config, inputs, auth_context, **kwargs
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._runner = TermDefinitionsRunner(tool_config.details)

    @classmethod
    def get_args_schema(
        cls, tool_config: TermDefinitionsToolConfig
    ) -> type[BaseTermDefinitionsArgs]:
        return build_term_definitions_args(tool_config)

    @classmethod
    def get_output_model(cls) -> type[TermDefinitionsStructuredContent]:
        return TermDefinitionsStructuredContent

    async def _execute(self, args: BaseTermDefinitionsArgs) -> ToolResult:
        outcome = await self._runner.run(args.inputs, args.terms)
        return ToolResult(
            content=self._text_content(self._runner.to_markdown(outcome)),
            structured_content=term_definitions_structured_content(outcome),
        )
