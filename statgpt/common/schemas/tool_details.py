import re
from typing import Any

from pydantic import AliasChoices, Field, PrivateAttr, model_validator

from statgpt.common.config import LLMModelsEnum
from statgpt.common.config import utils as config_utils

from .base import BaseYamlModel
from .enums import AvailableDatasetsHeaderFormat, AvailableDatasetsVersion, RAGVersion
from .model_config import LLMModelConfig


class FakeCall(BaseYamlModel):
    """A predefined tool call injected into history before the agent's first turn.

    Fake calls run the real tool ungated during speculative optimistic-guardrails runs
    (before the in-scope verdict), so a tool given a ``fake_call`` must be
    side-effect-free.
    """

    tool_call_id: str = Field(description="The tool call id of the fake call")
    args: str = Field(default="{}", description="Fake call arguments as JSON string")


class StageRules(BaseYamlModel):
    pattern: str | None = Field(
        default=None,
        description=(
            "Regex pattern matched against the rendered display name of a stage."
            " Prefer `key` instead when matching pipeline stages, since display names can be renamed via config."
        ),
    )
    key: str | None = Field(
        default=None,
        description=(
            "Stable logical key of the stage, matched exactly."
            " Use this for pipeline stages so visibility rules survive display-name renames."
        ),
    )
    debug_only: bool = Field(
        description="Whether the stage is only shown in debug mode. If False, it is always shown."
    )

    @model_validator(mode="after")
    def _require_pattern_or_key(self) -> "StageRules":
        if (self.pattern is None) == (self.key is None):
            raise ValueError("StageRules must specify exactly one of `pattern` or `key`.")
        return self


class StagesConfig(BaseYamlModel):
    tool_call_name: str | None = Field(
        default=None,
        description="The stage name of the tool call. Supports {} placeholders for tool args",
    )
    tool_result_name: str | None = Field(
        default=None,
        description="The stage name of the tool result, supports {} placeholders for tool args",
    )
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

    def is_stage_debug(self, key: str | None = None, name: str | None = None) -> bool:
        """Check if the stage should be displayed in debug mode only.

        Rules with `key` are matched exactly against `key`; rules with `pattern` are
        regex-matched against `name`. The first matching rule wins. Falls back to the
        config-level `debug_only` default.
        """
        for rule in self.rules:
            if rule.key is not None and key is not None and rule.key == key:
                return rule.debug_only
            if rule.pattern is not None and name is not None and re.match(rule.pattern, name):
                return rule.debug_only
        return self.debug_only


class StageDescriptor(BaseYamlModel):
    """Runtime descriptor for a configured stage.

    `name` is the (configurable) display name set via YAML.
    `key` is the stable logical key — a private attribute set by the parent model
    based on the descriptor's field name, so it cannot be overridden from YAML.
    """

    name: str
    _key: str = PrivateAttr(default="")

    @property
    def key(self) -> str:
        return self._key

    def is_debug(self, stages_config: StagesConfig) -> bool:
        return stages_config.is_stage_debug(key=self._key, name=self.name)


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
    deployment_id_raw: str = Field(
        default="statgpt-dial-rag-pgvector",
        validation_alias=AliasChoices("deployment_id", "deploymentId"),
        description="The DIAL deployment ID to use for the file RAG tool. Supports $env:{VAR} syntax.",
        serialization_alias="deploymentId",
    )
    metadata_endpoint_raw: str = Field(
        default="/indexing/documents/metadata",
        validation_alias=AliasChoices("metadata_endpoint", "metadataEndpoint"),
        description="The metadata endpoint path for DIAL RAG. Supports $env:{VAR} syntax.",
        serialization_alias="metadataEndpoint",
    )
    prefilter_llm_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
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

    def get_deployment_id(self) -> str:
        return config_utils.replace_env(self.deployment_id_raw)

    def get_metadata_endpoint(self) -> str:
        return config_utils.replace_env(self.metadata_endpoint_raw)

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

    deployment_id_raw: str = Field(
        validation_alias=AliasChoices("deployment_id", "deploymentId"),
        description="The DIAL deployment_id of the web search agent. Supports $env:{VAR} syntax.",
        serialization_alias="deploymentId",
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

    def get_deployment_id(self) -> str:
        return config_utils.replace_env(self.deployment_id_raw)


class WebSearchAgentDetails(BaseToolDetails):
    deployment_id_raw: str = Field(
        validation_alias=AliasChoices("deployment_id", "deploymentId"),
        description="The DIAL deployment_id of the web search agent. Supports $env:{VAR} syntax.",
        serialization_alias="deploymentId",
    )
    configuration: dict[str, Any] | None = Field(
        default=None, description="The configuration for the web search agent"
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

    def get_deployment_id(self) -> str:
        return config_utils.replace_env(self.deployment_id_raw)


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


class AvailableDatasetsDetails(BaseToolDetails):
    version: AvailableDatasetsVersion = Field(
        default=AvailableDatasetsVersion.short,
        description="The version of the available datasets tool",
    )
    include_indicator_count: bool = Field(
        default=False,
        description="Whether to include the number of indexed indicators per dataset and total.",
    )
    stats_header_format: AvailableDatasetsHeaderFormat = Field(
        default=AvailableDatasetsHeaderFormat.totals,
        description="The format of the statistics header in the tool output.",
    )


class OneShotToolDetails(BaseToolDetails):
    system_prompt: str | None = Field(
        default=None,
        description="The system prompt for the one-shot tool.",
    )
    llm_model_config: LLMModelConfig = Field(
        default_factory=LLMModelConfig,
        description="The LLM model configuration for the one-shot tool.",
    )


class DatasetsMetadataLLMConfig(LLMModelConfig):
    deployment: LLMModelsEnum = LLMModelsEnum.GPT_4_1_2025_04_14


class DatasetsMetadataDetails(OneShotToolDetails):
    llm_model_config: DatasetsMetadataLLMConfig = Field(
        default_factory=DatasetsMetadataLLMConfig,
        description="The LLM model configuration for the datasets metadata tool.",
    )


class SdmxQueryAppDetails(BaseToolDetails):
    base_url_raw: str = Field(
        validation_alias=AliasChoices("base_url", "baseUrl"),
        serialization_alias="baseUrl",
        description=(
            "Trusted base URL prefix prepended to every caller-provided path "
            "(e.g. an SDMX query application or proxy passthrough endpoint). The caller "
            "cannot specify a domain, so the tool can only reach this configured host. "
            "Supports $env:{VAR} syntax."
        ),
    )

    def get_base_url(self) -> str:
        return config_utils.replace_env(self.base_url_raw).rstrip("/")

    @model_validator(mode="after")
    def _validate_base_url(self) -> "SdmxQueryAppDetails":
        # Resolve and validate the base URL once at config-load time, so a missing
        # env var or a malformed URL fails fast here instead of on every request.
        try:
            base_url = self.get_base_url()
        except ValueError as e:
            raise ValueError(f"Could not resolve SDMX query app `base_url`: {e}") from e
        if not base_url:
            raise ValueError("SDMX query app `base_url` resolved to an empty value.")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                "SDMX query app `base_url` must start with 'http://' or 'https://',"
                f" got {base_url!r}."
            )
        return self


class AvailableTermsDetails(BaseToolDetails):
    include_domain: bool = Field(
        default=False, description="Whether to include the domain of each term in the output"
    )
    include_source: bool = Field(
        default=False, description="Whether to include the source of each term in the output"
    )
