from typing import Any

from fastmcp.exceptions import ToolError
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
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.tools import AvailableTermsTool as AvailableTermsToolConfig
from statgpt.common.schemas.tools import TermDefinitionsTool as TermDefinitionsToolConfig

from .base import StatGptMcpTool


def available_terms_structured_content(
    terms: list[schemas.GlossaryTerm], *, include_domain: bool, include_source: bool
) -> AvailableTermsStructuredContent:
    # Expose domain/source only when the tool is configured to (mirrors the LangChain rendering).
    # Both are non-optional strings upstream, so a missing value arrives as "": map it to None so
    # the field is dropped from the payload instead of being sent empty.
    records = [
        GlossaryTermRecord(
            term=term.term,
            domain=(term.domain or None) if include_domain else None,
            source=(term.source or None) if include_source else None,
        )
        for term in terms
    ]
    return AvailableTermsStructuredContent(terms=records, count=len(terms))


def term_definitions_structured_content(
    outcome: TermDefinitionsOutcome,
) -> TermDefinitionsStructuredContent:
    definitions = [
        GlossaryDefinitionRecord(
            term=lookup.found.term,
            definition=lookup.found.definition,
            domain=lookup.found.domain or None,
            source=lookup.found.source or None,
        )
        for lookup in outcome.lookups
        if lookup.found is not None
    ]
    not_found = [lookup.requested for lookup in outcome.lookups if lookup.found is None]
    return TermDefinitionsStructuredContent(
        definitions=definitions, not_found=not_found or None  # empty list would not be dropped
    )


class AvailableTermsMcpTool(
    StatGptMcpTool[AvailableTermsToolConfig, ToolArgs], tool_type=ToolTypes.AVAILABLE_TERMS
):
    """Structured-only: the complete result lives in `structuredContent`, so no text block."""

    _runner: AvailableTermsRunner = PrivateAttr()

    def __init__(
        self,
        tool_config: AvailableTermsToolConfig,
        channel_config: schemas.ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._runner = AvailableTermsRunner(tool_config.details)

    @classmethod
    def get_output_model(cls) -> type[AvailableTermsStructuredContent]:
        return AvailableTermsStructuredContent

    async def _execute(self, args: ToolArgs) -> ToolResult:
        terms = await self._runner.run(args.inputs)
        details = self._tool_config.details
        return self._structured_only(
            available_terms_structured_content(
                terms, include_domain=details.include_domain, include_source=details.include_source
            )
        )


class TermDefinitionsMcpTool(
    StatGptMcpTool[TermDefinitionsToolConfig, BaseTermDefinitionsArgs],
    tool_type=ToolTypes.TERM_DEFINITIONS,
):
    """Structured-only: the complete result lives in `structuredContent`, so no text block."""

    _runner: TermDefinitionsRunner = PrivateAttr()

    def __init__(
        self,
        tool_config: TermDefinitionsToolConfig,
        channel_config: schemas.ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
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
        if outcome.limit_exceeded:
            # Nothing was fetched and the caller must retry with fewer terms: that is an error, not
            # an empty result that would read as "none of these terms exist".
            raise ToolError(self._runner.limit_exceeded_message(outcome.limit))
        return self._structured_only(term_definitions_structured_content(outcome))
