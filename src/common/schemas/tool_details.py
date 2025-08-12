import re

from pydantic import Field

from common.config import utils as config_utils

from .base import BaseYamlModel
from .enums import RAGVersion


class FakeCall(BaseYamlModel):
    tool_call_id: str = Field(description="The tool call id of the fake call")
    args: str = Field(default="{}", description="Fake call arguments as JSON string")


class StageRules(BaseYamlModel):
    pattern: str = Field(description="The regex pattern to match the stage")
    debug_only: bool = Field(
        description="Whether the stage is only shown in debug mode. If False, it is always shown."
    )


class StagesConfig(BaseYamlModel):
    tool_call_name: str | None = Field(default=None, description="The stage name of the tool call")
    debug_only: bool = Field(
        default=True,
        description=(
            "A general setting that determines whether all tool stages will be displayed in debug mode only."
            " Might be overridden by rules defined in the `rule` field."
        ),
    )
    rules: list[StageRules] = Field(
        default_factory=list, description="The rules for displaying stages"
    )

    def is_stage_debug(self, stage_name: str) -> bool:
        """Check if the stage should be displayed in debug mode only."""
        for rule in self.rules:
            if re.match(rule.pattern, stage_name):
                return rule.debug_only
        return self.debug_only


class BaseToolDetails(BaseYamlModel):
    class Prompts(BaseYamlModel):
        system_prompt: str | None = Field(default=None)

    # TODO: Remove the `prompt` field from here and move it to the appropriate tool details.
    prompts: Prompts = Field(default_factory=Prompts)
    fake_call: FakeCall | None = Field(
        default=None,
        description="If not None, a fake call to this tool will be created at the start of the conversation.",
    )
    stages_config: StagesConfig = Field(default_factory=StagesConfig)  # type: ignore


class FileRagDetails(BaseToolDetails):
    version: RAGVersion

    # For Dial RAG:
    always_show_stages: bool = Field(
        default=False,
        description=(
            "If enabled, the stages received from the DIAL RAG will always be shown."
            " Otherwise, they will be displayed depending on the conversation debug flag."
        ),
    )
    attachment_url_override: str | None = Field(
        default=None,
        description=(
            "Replace the attachment `reference_url` with this value if provided."
            " If None, the original URL will be used."
        ),
    )
    decoder_of_latest: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping the publication type to a function that generates a time range "
        "corresponding to the 'latest'",
    )

    def get_attachment_url_override(self) -> str | None:
        if self.attachment_url_override is None or not self.attachment_url_override.strip():
            return None
        return config_utils.replace_env(self.attachment_url_override.strip())


class WebSearchDetails(BaseToolDetails):
    class Domains(BaseYamlModel):
        field_name: str = Field(description="Argument field name in web search tool")
        field_description: str = Field(description="Argument field description in web search tool")
        allowed_values: list[str] = Field(
            description="The list of allowed domains for the web search tool"
        )

    deployment_id: str | None = Field(
        default=None, description="The DIAL deployment_id to use for the web search tool"
    )
    domains: Domains | None = Field(
        default=None, description="The list of allowed domains for the web search tool"
    )
    always_show_stages: bool = Field(
        default=False,
        description=(
            "If enabled, the stages received from the DIAL WEB RAG will always be shown."
            " Otherwise, they will be displayed depending on the conversation debug flag."
        ),
    )
    urls_only: bool = Field(
        default=False,
        description=(
            "If disabled, the tool returns the response from the DIAL WEB RAG."
            " Otherwise, it returns only the URLs of the attachments."
        ),
    )


class WebSearchAgentDetails(BaseToolDetails):
    deployment_id: str | None = Field(
        default=None, description="The DIAL deployment_id of the web search agent"
    )
    system_prompt: str | None = Field(
        default=None,
        description="The system prompt for the web search agent.",
    )
    always_show_stages: bool = Field(
        default=False,
        description=(
            "If enabled, the stages received from the agent will always be shown."
            " Otherwise, they will be displayed depending on the conversation debug flag."
        ),
    )
    urls_only: bool = Field(
        default=False,
        description=(
            "If disabled, the tool returns the response from the agent."
            " Otherwise, it returns only the URLs of the attachments."
        ),
    )


class PublicationType(BaseYamlModel):
    name: str = Field(description="The name of the publication type")
    description: str = Field(description="The description of the publication type")


class AvailablePublicationsDetails(BaseToolDetails):
    publication_types: list[PublicationType] = Field(
        description="The list of publication types", default_factory=list
    )


class PlainContentDetails(BaseToolDetails):
    file_path: str = Field(
        default="", description="The path to the file containing the plain content"
    )
    replace_envs: bool = Field(
        default=False,
        description="Whether to replace environment variables in the file content",
    )


class TermDefinitionsDetails(BaseToolDetails):
    limit: int | None = Field(
        default=None, description="The maximum number of term definitions returned by the tool"
    )
