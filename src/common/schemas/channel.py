from pydantic import BaseModel, Field

from .base import BaseYamlModel, DbDefaultBase
from .enums import LocaleEnum
from .model_config import LLMModelConfig
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


class OutOfScopeConfig(BaseYamlModel):
    domain: str = Field(
        description="The domain of the chat bot. Other domains are considered out of scope."
    )
    custom_instructions: list[str] | None = Field(
        description=(
            "List of specific topics, questions, and subject matters that the chatbot should"
            " not engage with or provide information about."
        ),
        default=None,
    )
    use_general_topics_blacklist: bool = Field(
        default=True,
        description=(
            "Whether to use the general topics blacklist from NotSupportedScenariosPrompts."
            " It contains common out-of-scope topics like harmful content, prompt engineering, etc."
            " If false, only custom_instructions will be used."
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
    named_entity_types: list[str] = Field(
        default=[], description="The named entity types used for named entity extraction"
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
        return [
            getattr(self, field) for field in self.tool_fields if getattr(self, field) is not None
        ]

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


class Channel(DbDefaultBase, ChannelBase):
    pass
