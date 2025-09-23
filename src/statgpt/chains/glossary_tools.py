from pydantic import Field

from common import models
from common.schemas import ToolTypes
from common.schemas.tools import AvailableTermsTool as AvailableTermsToolConfig
from common.schemas.tools import TermDefinitionsTool as TermDefinitionsToolConfig
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.schemas import ToolArtifact, ToolMessageState


class AvailableTermsTool(
    StatGptTool[AvailableTermsToolConfig], tool_type=ToolTypes.AVAILABLE_TERMS
):
    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)

        terms = await data_service.get_available_terms()
        formatted_terms = self._terms_to_markdown(terms)
        response = "# List of available glossary terms:\n" + "\n".join(formatted_terms)

        target = ChainParameters.get_target(inputs)
        number_of_terms_to_show = min(10, len(formatted_terms))
        target.append_content(
            f"Glossary contains {len(terms)} terms.\n\nFirst {number_of_terms_to_show} terms:\n"
        )
        target.append_content("\n".join(formatted_terms[:number_of_terms_to_show]) + "\n")

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

    def _terms_to_markdown(self, terms: list[models.GlossaryTerm]) -> list[str]:
        include_domain = self._tool_config.details.include_domain
        include_source = self._tool_config.details.include_source

        formatted_terms = []
        for term in terms:
            formatted_term = f"- **{term.term}**"
            if include_domain:
                formatted_term += f", domain: {term.domain}"
            if include_source:
                formatted_term += f", source: {term.source}"
            formatted_terms.append(formatted_term)
        return formatted_terms


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


class TermDefinitionsTool(
    StatGptTool[TermDefinitionsToolConfig], tool_type=ToolTypes.TERM_DEFINITIONS
):
    @classmethod
    def get_args_schema(cls, tool_config: TermDefinitionsToolConfig) -> type[ToolArgs]:
        """Return the schema for the arguments that this tool accepts."""

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

    async def _arun(self, inputs: dict, terms: list[str]) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)

        if self._tool_config.details.limit and len(terms) > self._tool_config.details.limit:
            return (
                f"The number of requested terms exceeds the limit of {self.config.details.limit}. "
                "Please reduce the number of terms and try again. Also, mind that massive requests "
                "are not supported (e.g. asking for definitions of all available terms), as this is "
                "not the intended use case of this tool.",
                ToolArtifact(state=ToolMessageState(type=self.tool_type)),
            )

        all_terms = await data_service.get_available_terms()
        response = self.term_definition_to_markdown(terms, all_terms)

        target = ChainParameters.get_target(inputs)
        target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

    @staticmethod
    def term_definition_to_markdown(terms: list[str], all_terms: list[models.GlossaryTerm]) -> str:
        all_terms_dict = {term.term.lower(): term for term in all_terms}

        response = "## Glossary term definitions:\n"
        for term in terms:
            term_to_search = term.strip().lower()

            if term_db := all_terms_dict.get(term_to_search):
                response += f"### {term_db.term}\n"
                response += f"**Domain:** {term_db.domain}  \n"
                response += f"**Source:** {term_db.source}  \n"
                response += f"**Definition:**  \n{term_db.definition}\n\n"
            else:
                response += f"### {term}\n"
                response += "The term is not available in the glossary.\n\n"
        return response
