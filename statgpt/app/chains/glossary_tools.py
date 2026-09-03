from dataclasses import dataclass

from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.common import schemas
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.tool_details import AvailableTermsDetails, TermDefinitionsDetails
from statgpt.common.schemas.tools import AvailableTermsTool as AvailableTermsToolConfig
from statgpt.common.schemas.tools import TermDefinitionsTool as TermDefinitionsToolConfig

# ~~~~~~~~~~~~~ Available terms ~~~~~~~~~~~~~


class AvailableTermsRunner:
    """Lists the channel's glossary terms; shared by the LangChain and MCP interfaces."""

    def __init__(self, details: AvailableTermsDetails):
        self._details = details

    async def run(self, inputs: dict) -> list[schemas.GlossaryTerm]:
        data_service = ChainParameters.get_data_service(inputs)
        return await data_service.get_available_terms()

    def to_markdown_lines(self, terms: list[schemas.GlossaryTerm]) -> list[str]:
        formatted_terms = []
        for term in terms:
            formatted_term = f"- **{term.term}**"
            if self._details.include_domain:
                formatted_term += f", domain: {term.domain}"
            if self._details.include_source:
                formatted_term += f", source: {term.source}"
            formatted_terms.append(formatted_term)
        return formatted_terms

    def to_markdown(self, terms: list[schemas.GlossaryTerm]) -> str:
        return (
            f"Glossary contains {len(terms)} terms.\n\n*List of available glossary terms:*\n"
            + "\n".join(self.to_markdown_lines(terms))
        )


class AvailableTermsTool(
    StatGptTool[AvailableTermsToolConfig], tool_type=ToolTypes.AVAILABLE_TERMS
):
    def __init__(
        self, tool_config: AvailableTermsToolConfig, channel_config: schemas.ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._runner = AvailableTermsRunner(tool_config.details)

    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        terms = await self._runner.run(inputs)
        formatted_terms = self._runner.to_markdown_lines(terms)
        response = self._runner.to_markdown(terms)

        target = ChainParameters.get_target(inputs)
        if target:
            number_of_terms_to_show = min(10, len(formatted_terms))
            target.append_content(
                f"Glossary contains {len(terms)} terms.\n\nFirst {number_of_terms_to_show} terms:\n"
            )
            target.append_content("\n".join(formatted_terms[:number_of_terms_to_show]) + "\n")

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))


# ~~~~~~~~~~~~~ Term definitions ~~~~~~~~~~~~~


class BaseTermDefinitionsArgs(ToolArgs):
    # NOTE: we introduce a separate model here for tool calling eval.
    # tool arguments checker (LLM) does not need to see
    # limit message in tool description.
    terms: list[str] = Field(
        description=(
            "List of terms to get definitions for."
            " Each value must be exactly the same as returned by the `Available_Terms` tool."
        ),
        # max_length=tool_config.details.limit,  # This keyword is not yet supported by the OpenAI API
    )


def build_term_definitions_args(
    tool_config: TermDefinitionsToolConfig,
) -> type[BaseTermDefinitionsArgs]:
    """The term-definitions args schema, with the configured limit spelled out in the description."""

    limit_msg = (
        f" Maximum number of terms is limited to {tool_config.details.limit}."
        if tool_config.details.limit
        else ""
    )

    class TermDefinitionsArgs(BaseTermDefinitionsArgs):
        terms: list[str] = Field(
            description=BaseTermDefinitionsArgs.model_fields["terms"].description + limit_msg,  # type: ignore
            # max_length=tool_config.details.limit,  # This keyword is not yet supported by the OpenAI API
        )

    return TermDefinitionsArgs


@dataclass(frozen=True)
class TermLookup:
    """One requested term and the glossary entry it resolved to, if any."""

    requested: str
    found: schemas.GlossaryTerm | None


@dataclass(frozen=True)
class TermDefinitionsOutcome:
    """What a term-definitions call produced: either the lookups, or nothing because the request
    exceeded the configured limit."""

    lookups: list[TermLookup]
    limit_exceeded: bool = False
    limit: int | None = None


class TermDefinitionsRunner:
    """Resolves requested terms against the glossary; shared by the LangChain and MCP interfaces."""

    def __init__(self, details: TermDefinitionsDetails):
        self._details = details

    async def run(self, inputs: dict, terms: list[str]) -> TermDefinitionsOutcome:
        limit = self._details.limit
        if limit and len(terms) > limit:
            # Over-limit: nothing is fetched; the interfaces explain the reason in their text.
            return TermDefinitionsOutcome(lookups=[], limit_exceeded=True, limit=limit)

        data_service = ChainParameters.get_data_service(inputs)
        all_terms = await data_service.get_available_terms()
        return TermDefinitionsOutcome(lookups=self.lookup(terms, all_terms))

    @staticmethod
    def lookup(terms: list[str], all_terms: list[schemas.GlossaryTerm]) -> list[TermLookup]:
        all_terms_dict = {term.term.lower(): term for term in all_terms}
        return [
            TermLookup(requested=term, found=all_terms_dict.get(term.strip().lower()))
            for term in terms
        ]

    @staticmethod
    def to_markdown(outcome: TermDefinitionsOutcome) -> str:
        if outcome.limit_exceeded:
            return (
                f"The number of requested terms exceeds the limit of {outcome.limit}. "
                "Please reduce the number of terms and try again. Also, mind that massive requests "
                "are not supported (e.g. asking for definitions of all available terms), as this is "
                "not the intended use case of this tool."
            )

        response = "## Glossary term definitions:\n"
        for lookup in outcome.lookups:
            if term_db := lookup.found:
                response += f"### {term_db.term}\n"
                response += f"**Domain:** {term_db.domain}  \n"
                response += f"**Source:** {term_db.source}  \n"
                response += f"**Definition:**  \n{term_db.definition}\n\n"
            else:
                response += f"### {lookup.requested}\n"
                response += "The term is not available in the glossary.\n\n"
        return response


class TermDefinitionsTool(
    StatGptTool[TermDefinitionsToolConfig], tool_type=ToolTypes.TERM_DEFINITIONS
):
    def __init__(
        self,
        tool_config: TermDefinitionsToolConfig,
        channel_config: schemas.ChannelConfig,
        **kwargs,
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._runner = TermDefinitionsRunner(tool_config.details)

    @classmethod
    def get_args_schema(cls, tool_config: TermDefinitionsToolConfig) -> type[ToolArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return build_term_definitions_args(tool_config)

    async def _arun(self, inputs: dict, terms: list[str]) -> tuple[str, ToolArtifact]:
        outcome = await self._runner.run(inputs, terms)
        response = self._runner.to_markdown(outcome)

        if not outcome.limit_exceeded:
            target = ChainParameters.get_target(inputs)
            if target:
                target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))
