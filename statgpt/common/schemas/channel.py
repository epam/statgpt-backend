from pydantic import BaseModel, Field

from statgpt.common.settings.elastic import ElasticSearchSettings

from .auditable import Auditable
from .base import BaseYamlModel, DbDefaultBase
from .enums import ChannelIndexStatusScope, LocaleEnum
from .model_config import LLMModelConfig
from .onboarding import OnboardingConfig
from .tools import (
    AvailableDatasetsTool,
    AvailablePublicationsTool,
    AvailableTermsTool,
    BaseToolConfig,
    DataQueryTool,
    DatasetsMetadataTool,
    DatasetStructureTool,
    FileRagTool,
    PlainContentTool,
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
    additional_instructions: str = Field(
        default="",
        description="Additional instructions to put to the agent's system prompt",
    )
    additional_context: str = Field(
        default="",
        description="Additional context for the supreme agent",
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
    buttons: list[ConversationStarterConfig] = Field(
        description="The buttons displayed to the user when the conversation starts."
    )


class ChannelConfig(BaseYamlModel):
    locale: LocaleEnum = Field(default=LocaleEnum.EN, description="The locale of the channel")
    conversation_starters: ConversationStartersConfig | None = Field(
        default=None, description="The conversation starters configuration"
    )
    onboarding: OnboardingConfig | None = Field(
        default=None, description="The onboarding configuration"
    )
    named_entity_types: list[str] = Field(
        default_factory=list, description="The named entity types used for named entity extraction"
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
    bearer_token_required: bool = Field(
        default=False,
        description=(
            "Whether this channel requires bearer token forwarding to external APIs. "
            "When True and no bearer token is present, system user context will be used "
            "if the user has an allowed role."
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
    term_definitions: TermDefinitionsTool | None = Field(None)
    web_search: WebSearchTool | None = Field(None)
    web_search_agent: WebSearchAgentTool | None = Field(None)

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

    def list_named_entity_types(self) -> list[str]:
        return [
            self.country_named_entity_type,
            *self.named_entity_types,
        ]


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
