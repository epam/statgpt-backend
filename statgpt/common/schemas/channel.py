from collections import Counter
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from statgpt.common.config import utils as config_utils
from statgpt.common.settings.elastic import ElasticSearchSettings
from statgpt.common.utils.media_types import MediaTypes

from .auditable import Auditable
from .base import BaseYamlModel, DbDefaultBase
from .enums import ChannelIndexStatusScope, LocaleEnum, McpResourceTypes, PreprocessingStatusEnum
from .model_config import LLMModelConfig
from .onboarding import OnboardingConfig
from .tools import (
    AvailableDatasetsTool,
    AvailablePublicationsTool,
    AvailableTermsTool,
    BaseToolConfig,
    DataQueryTool,
    DatasetsMetadataAppTool,
    DatasetsMetadataTool,
    DatasetStructureTool,
    DeepResearchTool,
    FileRagTool,
    PlainContentTool,
    SdmxQueryAppTool,
    TermDefinitionsTool,
    WebSearchAgentTool,
    WebSearchTool,
)

_elasticsearch_settings = ElasticSearchSettings()


class SupremeAgentConfig(BaseYamlModel):
    name: str = Field(description="The name of the chatbot displayed to the user")
    domain: str = Field(description="The domain of the chatbot displayed to the user.")
    terminology_domain: str = Field(
        description="The terminology domain of the chatbot for chatbot's instructions."
    )
    language_instructions: list[str] = Field(
        description="Instructions on the Chatbot's language (e.g. tone, variant of English, etc.)",
        default_factory=list,
    )
    max_agent_iterations: int = Field(
        default=5,
        description=(
            "The maximum number of tool calling iterations the chatbot can perform in a single response."
        ),
    )
    llm_model_config: LLMModelConfig = Field(
        default_factory=LLMModelConfig,
        description="LLM model configuration for the supreme agent",
    )
    additional_context: str = Field(
        default="",
        description="Additional context for the supreme agent",
    )
    general_section: str = Field(
        default="",
        description=(
            "Custom content for the 'General' section of the system prompt."
            " If empty, the default content is used."
        ),
    )
    tool_usage_section: str = Field(
        default="",
        description=(
            "Custom content for the 'Tool Usage' section of the system prompt."
            " If empty, the default content is used."
        ),
    )
    no_calculations_section: str = Field(
        default="",
        description=(
            "Custom content for the 'No Calculations and Analytics' section of the system prompt."
            " If empty, the default content is used."
        ),
    )


class OutOfScopeConfig(BaseYamlModel):
    llm_model_config: LLMModelConfig = Field(
        default_factory=LLMModelConfig,
        description="LLM model configuration for guardrails.",
    )
    domain: str = Field(
        description="The domain of the chat bot. Other domains are considered out of scope."
    )
    custom_blacklist: list[str] | None = Field(
        description=(
            "List of specific topics, questions, and subject matters that the chatbot should"
            " not engage with or provide information about."
        ),
        default=None,
    )
    use_general_topics_blacklist: bool = Field(
        default=True,
        description=(
            "Whether to use the general topics blacklist from guardrails_default_prompts."
            " It contains common out-of-scope topics like harmful content, prompt engineering, etc."
            " If false, only custom_blacklist will be used."
        ),
    )
    start_new_conversation_message: str = Field(
        default=(
            "Threshold of out-of-scope messages in conversation history exceeded. Please start a new chat if you'd like "
            "to discuss topics related to the official statistics."
        ),
        description=(
            "The message sent to the user when the chatbot detects the user is trying to"
            " discuss out-of-scope topics continuously."
        ),
    )
    start_new_conversation_messages_threshold: int = Field(
        default=3,
        description=(
            "The limit to number of out-of-scope messages in conversation history to not trigger the"
            " start_new_conversation_message. If the number of out-of-scope messages exceeds this"
            " threshold, the start_new_conversation_message will be sent to the user. If set to -1, the"
            " feature is disabled."
        ),
    )


class McpResourceConfig(BaseYamlModel):
    """Base config for an MCP resource exposed via the MCP server.

    Subclasses add kind-specific fields; the ``type`` field discriminates between them.
    """

    type: McpResourceTypes
    uri: str = Field(
        description=(
            "The resource URI exposed via MCP, e.g. 'ui://statgpt/data-widget.html'."
            " Bound to a tool via its `mcp_app_resource_uri` so the host can preload the widget."
        ),
    )

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, uri: str) -> str:
        if not uri.startswith("ui://"):
            raise ValueError(f"MCP resource `uri` must start with 'ui://', got {uri!r}.")
        return uri


class ProxiedResourceConfig(McpResourceConfig):
    """An MCP resource whose HTML is proxied verbatim from an external HTTP endpoint.

    The backend stores no HTML: on `resources/read` it does a server-to-server GET against
    `html_url`, returns the body verbatim, and caches it for `cache_ttl_seconds`.
    """

    type: Literal[McpResourceTypes.PROXIED] = McpResourceTypes.PROXIED
    origin_raw: str = Field(
        validation_alias=AliasChoices("origin", "originRaw"),
        serialization_alias="origin",
        description=(
            "Origin the widget HTML loads its JS/CSS/fonts from; exposed to the host as"
            " `_meta.ui.csp.resourceDomains`. Supports $env:{VAR} syntax."
        ),
    )
    html_url_raw: str = Field(
        validation_alias=AliasChoices("html_url", "htmlUrl", "htmlUrlRaw"),
        serialization_alias="htmlUrl",
        description=(
            "Internal endpoint the backend fetches the resource HTML from (server-to-server)."
            " Supports $env:{VAR} syntax."
        ),
    )
    cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        description="TTL (seconds) for the in-process cache of the fetched HTML.",
    )
    mime_type: str = Field(
        default=MediaTypes.HTML_MCP_APP,
        description=(
            "MIME type reported for the resource content. Defaults to the MCP Apps UI HTML"
            " type 'text/html;profile=mcp-app' (ext-apps 2026-01-26)."
        ),
    )

    def get_origin(self) -> str:
        return config_utils.replace_env(self.origin_raw).rstrip("/")

    def get_html_url(self) -> str:
        return config_utils.replace_env(self.html_url_raw)

    @model_validator(mode="after")
    def _validate_urls(self) -> "ProxiedResourceConfig":
        # Resolve $env:{VAR} once at config-load time so a missing var or malformed URL
        # fails fast here instead of on every resources/read.
        origin = self.get_origin()  # already $env-resolved and rstrip("/")
        parts = urlsplit(origin)
        if (
            parts.scheme not in ("http", "https")
            or not parts.netloc
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "MCP resource `origin` must be a bare origin like 'https://host[:port]'"
                f" (no path, query, or fragment), got {origin!r}."
            )
        html_url = self.get_html_url()
        if not html_url.startswith(("http://", "https://")):
            raise ValueError(
                "MCP resource `html_url` must start with 'http://' or 'https://',"
                f" got {html_url!r}."
            )
        return self


class McpConfig(BaseYamlModel):
    tool_name_prefix: str = Field(
        default="",
        pattern=r'^[a-zA-Z0-9_\.-]*$',
        description=(
            "Prefix prepended to tool names exposed via MCP (e.g. 'statgpt__'). "
            "Internal agent tool names are unaffected. Empty string disables prefixing."
        ),
    )
    resources: list[ProxiedResourceConfig] = Field(
        default_factory=list,
        description=(
            "MCP resources (e.g. MCP-App UI widgets) served by the MCP server."
            " Empty (the default) disables the feature."
        ),
    )

    @field_validator("resources")
    @classmethod
    def _validate_unique_uris(
        cls, resources: list[ProxiedResourceConfig]
    ) -> list[ProxiedResourceConfig]:
        counts = Counter(r.uri for r in resources)
        duplicates = {uri for uri, count in counts.items() if count > 1}
        if duplicates:
            raise ValueError(f"Duplicate MCP resource uri(s): {sorted(duplicates)}.")
        return resources

    @property
    def resource_uris(self) -> set[str]:
        return {r.uri for r in self.resources}


class TokenUsageConfig(BaseYamlModel):
    debug_only: bool = Field(
        default=True,
        description=(
            "If enabled, the stage will only be displayed in debug mode."
            " Otherwise, the stage will always be shown."
        ),
    )
    stage_name: str = Field(
        default="[DEBUG] Token Usage", description="The stage name of the token usage"
    )


class ConversationStarterConfig(BaseYamlModel):
    title: str = Field(description="The title of the conversation starter")
    text: str = Field(
        description="The text sent to the chatbot when the conversation starter is clicked"
    )


class ConversationStartersConfig(BaseYamlModel):
    intro_text: str = Field(
        description="The text displayed to the user when the conversation starts."
    )
    title: str | None = Field(
        default=None,
        description=(
            "Optional override for the conversation-starter widget title. "
            "If unset, the default JSON-schema title is used."
        ),
    )
    input_placeholder: str | None = Field(
        default=None,
        description=(
            "Optional placeholder text for the chat input field. "
            "Rendered as the 'statgpt:inputPlaceholder' JSON-schema extension."
        ),
    )
    buttons: list[ConversationStarterConfig] = Field(
        description="The buttons displayed to the user when the conversation starts."
    )


class DiscoveryRagConfig(BaseYamlModel):
    """Where this channel's discovery dataset records are published.

    The indexing job needs the target before it can publish anything, so a channel without
    this block cannot be indexed - which is reported when the job is triggered rather than
    discovered by a background run that fails.
    """

    application_id_raw: str = Field(
        validation_alias=AliasChoices("application_id", "applicationId"),
        serialization_alias="applicationId",
        description=(
            "The DIAL application id of the Generic RAG channel holding this channel's"
            " discovery records. Supports $env:{VAR} syntax."
        ),
    )

    def get_application_id(self) -> str:
        return config_utils.replace_env(self.application_id_raw)


class ChannelConfig(BaseYamlModel):
    locale: LocaleEnum = Field(default=LocaleEnum.EN, description="The locale of the channel")
    conversation_starters: ConversationStartersConfig | None = Field(
        default=None, description="The conversation starters configuration"
    )
    onboarding: OnboardingConfig | None = Field(
        default=None, description="The onboarding configuration"
    )
    named_entity_types: list[str] = Field(
        default_factory=list,
        description="The named entity types used for named entity extraction",
    )
    country_named_entity_type: str = Field(
        default="Country/Reference Area",
        description="The country named entity type used for named entity extraction",
    )
    supreme_agent: SupremeAgentConfig = Field(description="The supreme agent configuration")
    out_of_scope: OutOfScopeConfig | None = Field(
        None, description="The out of scope configuration"
    )
    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig)
    mcp: McpConfig = Field(default_factory=McpConfig, description="MCP server configuration")
    bearer_token_required: bool = Field(
        default=False,
        description=(
            "Whether this channel requires bearer token forwarding to external APIs. "
            "When True and no bearer token is present, system user context will be used "
            "if the user has an allowed role."
        ),
    )
    discovery_rag: DiscoveryRagConfig | None = Field(
        default=None,
        description=(
            "The Generic RAG channel this channel's discovery dataset records are published"
            " to. Required to run a discovery indexing job."
        ),
    )

    # ~~~ Tools: ~~~
    available_datasets: AvailableDatasetsTool | None = Field(None)
    datasets_metadata: DatasetsMetadataTool | None = Field(None)
    dataset_structure: DatasetStructureTool | None = Field(None)
    available_publications: AvailablePublicationsTool | None = Field(None)
    available_terms: AvailableTermsTool | None = Field(None)
    data_query: DataQueryTool | None = Field(default=None)
    file_rag: FileRagTool | None = Field(None)
    plain_content: PlainContentTool | None = Field(None)
    sdmx_query_app: SdmxQueryAppTool | None = Field(None)
    datasets_metadata_app: DatasetsMetadataAppTool | None = Field(None)
    term_definitions: TermDefinitionsTool | None = Field(None)
    web_search: WebSearchTool | None = Field(None)
    web_search_agent: WebSearchAgentTool | None = Field(None)
    deep_research: DeepResearchTool | None = Field(None)

    @property
    def tool_fields(self) -> list[str]:
        return [
            'available_datasets',
            'datasets_metadata',
            'dataset_structure',
            'available_publications',
            'available_terms',
            'data_query',
            'file_rag',
            'plain_content',
            'sdmx_query_app',
            'datasets_metadata_app',
            'term_definitions',
            'web_search',
            'web_search_agent',
        ]

    @property
    def tools(self) -> list[BaseToolConfig]:
        tools = [
            getattr(self, field) for field in self.tool_fields if getattr(self, field) is not None
        ]
        tools = [tool for tool in tools if tool.enabled]
        return tools

    @property
    def agent_tools(self) -> list[BaseToolConfig]:
        """Enabled tools visible to the Supreme Agent / LLM.

        Excludes ``mcp_only`` tools, which are surfaced exclusively via the MCP server
        (e.g. UI-initiated tool calls) and must never be exposed to the agent or any
        agent-facing consumer (e.g. the out-of-scope checker).
        """
        return [tool for tool in self.tools if not tool.mcp_only]

    @property
    def is_deep_research_available(self) -> bool:
        """Whether the channel has the Deep Research tool configured and enabled."""
        return self.deep_research is not None and self.deep_research.enabled

    def list_named_entity_types(self) -> list[str]:
        return [
            self.country_named_entity_type,
            *self.named_entity_types,
        ]

    @model_validator(mode="after")
    def _validate_mcp_app_resource_bindings(self) -> "ChannelConfig":
        # Every tool that binds a UI widget must reference a resource declared in mcp.resources.
        declared = self.mcp.resource_uris
        for tool in self.tools:
            uri = tool.mcp_app_resource_uri
            if uri is not None and uri not in declared:
                raise ValueError(
                    f"Tool {tool.name!r} binds `mcp_app_resource_uri` {uri!r}, which is not"
                    " declared in `mcp.resources`."
                )
        return self


class ChannelBase(BaseModel):
    title: str
    description: str = ""
    deployment_id: str = Field(description="Must be unique for each channel")
    llm_model: str
    details: ChannelConfig = Field(default_factory=ChannelConfig)  # type: ignore


class ChannelUpdate(BaseModel):
    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    deployment_id: str | None = Field(default=None, description="Must be unique for each channel")
    llm_model: str | None = Field(default=None)
    details: ChannelConfig | None = Field(default=None)


class Channel(DbDefaultBase, ChannelBase, Auditable):
    def get_entity_id(self) -> str:
        return self.deployment_id

    def get_entity_name(self) -> str:
        return self.title

    def get_state_after(self) -> dict:
        return self.model_dump(mode='json', exclude={"created_at", "updated_at"})

    def get_item_id(self) -> int:
        return self.id

    @property
    def indicator_table_name(self) -> str:
        return f"Indicators_{self.id}"

    @property
    def non_indicator_dimensions_table_name(self) -> str:
        return f"AvailableDimensions_{self.id}"

    @property
    def special_dimensions_table_name(self) -> str:
        return f"SpecialDimensions_{self.id}"

    @property
    def matching_index_name(self) -> str:
        return f"{_elasticsearch_settings.matching_index}_{self.id}"

    @property
    def indicators_index_name(self) -> str:
        return f"{_elasticsearch_settings.indicators_index}_{self.id}"


class DeduplicationStatus(BaseModel):
    """Status information about channel deduplication requirements."""

    deduplication_required: bool = Field(
        description="Whether deduplication is required for the channel"
    )
    total_duplicate_count: int = Field(
        description="Total number of duplicate documents across all dimension stores"
    )
    non_indicator_dimensions_duplicate_count: int = Field(
        description=(
            "Number of duplicate documents in the non-indicator dimensions store "
            "(regular dimensions such as country, frequency, region — excludes "
            "indicator and special dimensions)."
        )
    )
    special_dimensions_duplicate_count: int = Field(
        description=(
            "Number of duplicate documents in the special dimensions store. Special "
            "dimensions require dedicated processor-based handling (configured per "
            "channel) and are counted separately — they are NOT part of indicator or "
            "non-indicator dimensions."
        )
    )
    indicator_dimensions_duplicate_count: int = Field(
        description=(
            "Number of duplicate documents in the indicator dimensions store "
            "(dimensions used as measures, e.g. GDP, Inflation)."
        )
    )


class VectorStoreSizes(BaseModel):
    """Size information about channel vector store index"""

    non_indicator_dimensions_size: int = Field(
        description=(
            "Number of documents in the non-indicator dimensions store "
            "(regular dimensions such as country, frequency, region — excludes "
            "indicator and special dimensions)."
        )
    )
    special_dimensions_size: int = Field(
        description=(
            "Number of documents in the special dimensions store. Special dimensions "
            "require dedicated processor-based handling (configured per channel) and "
            "are counted separately — they are NOT part of indicator or non-indicator "
            "dimensions."
        )
    )
    indicator_dimensions_size: int = Field(
        description=(
            "Number of documents in the indicator dimensions store "
            "(dimensions used as measures, e.g. GDP, Inflation)."
        )
    )


class VectorStoreStatus(BaseModel):
    """Status information about channel vector store index"""

    deduplication: DeduplicationStatus = Field(
        description="Deduplication status information for the channel"
    )
    sizes: VectorStoreSizes = Field(description="Size information for the channel vector store")


class ChannelIndexStatus(BaseModel):
    """Status information about channel index"""

    scope: ChannelIndexStatusScope = Field(description="The scope of the channel index status")
    vector_store: VectorStoreStatus = Field(
        description="Vector store status information for the channel"
    )


class DeduplicationJob(DbDefaultBase):
    """Schema for a deduplication job record."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    channel_id: int

    status: PreprocessingStatusEnum
    """Job execution status (QUEUED, IN_PROGRESS, COMPLETED, FAILED)."""

    reason_for_failure: str | None = None

    non_indicator_remapped: int | None = None
    """Metadata rows remapped to keeper documents in the non-indicator dimensions store."""

    non_indicator_deleted: int | None = None
    """Orphaned documents deleted from the non-indicator dimensions store."""

    special_remapped: int | None = None
    """Metadata rows remapped to keeper documents in the special dimensions store."""

    special_deleted: int | None = None
    """Orphaned documents deleted from the special dimensions store."""
