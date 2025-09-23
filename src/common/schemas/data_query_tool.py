from pydantic import Field, TypeAdapter, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from common.config.utils import replace_env

from .base import BaseYamlModel, SystemUserPrompt
from .enums import (
    DataQueryVersion,
    IndexerVersion,
    IndicatorSelectionVersion,
    SpecialDimensionsProcessorType,
)
from .model_config import LLMModelConfig
from .tool_details import BaseToolDetails


def bool_from_str(value: str) -> bool:
    """
    Converts a string to a boolean value.
    If the string is an environment variable reference, it will be replaced with its value before conversion.
    """
    return TypeAdapter(bool).validate_python(replace_env(value))


class DataQueryPrompts(BaseYamlModel):
    datetime_prompt: str | None = Field(default=None)
    group_expander_prompt: str | None = Field(default=None)
    group_expander_fallback_prompt: str | None = Field(default=None)
    normalization_prompt: str | None = Field(default=None)
    named_entities_prompt: str | None = Field(default=None)
    dataset_selection_prompt: str | None = Field(default=None)
    indicators_selection_system_prompt: str | None = Field(default=None)
    validation_system_prompt: str | None = Field(default=None)
    validation_user_prompt: str | None = Field(default=None)
    incomplete_queries_prompt: str | None = Field(default=None)


class DataQueryMessages(BaseYamlModel):
    no_data_for_country: str | None = Field(
        default=None,
        description="Message for the no data for country response, can contain {country_details} placeholder",
    )
    no_data: str | None = Field(
        default=None,
        description="Message for the no data response",
    )
    data_query_executed_agent_only: str | None = Field(
        default=None,
        description="Message for the data query executed response, only for agent, won't be shown to the user. "
        "Will be appended to the end of the tool response if present.",
    )
    multiple_datasets_agent_only: str | None = Field(
        default=None,
        description="Message for the multiple datasets response, only for agent, won't be shown to the user. "
        "Will be appended to the end of the tool response if present.",
    )


class ToolAttachment(BaseYamlModel):
    enabled_str: str = Field(
        description=(
            "Whether the tool should return this attachment."
            " The value can be a reference to an environment variable."
        )
    )
    name: str | None = Field(
        default=None, description="Attachment name template, may be None if disabled."
    )

    @field_validator('enabled_str', mode='after')
    @classmethod
    def validate_enabled(cls, enabled: str) -> str:
        """Validate the `enabled` field to ensure it can return a boolean value."""
        try:
            bool_from_str(enabled)
        except Exception as e:
            raise ValueError(f"Invalid value for enabled_str: {enabled}. Error: {e}")
        return enabled

    @field_validator("name", mode="after")
    def validate_name(cls, name: str | None, info: FieldValidationInfo) -> str | None:
        """Validate the `name` field to ensure it is not empty if `enabled` is True."""
        enabled_str = info.data.get("enabled_str")
        if enabled_str and bool_from_str(enabled_str) and not name:
            raise ValueError("Attachment name must be provided if the attachment is enabled.")
        return name

    @property
    def enabled(self) -> bool:
        return bool_from_str(self.enabled_str)


class DataQueryAttachments(BaseYamlModel):
    """Represents the attachments that can be returned by the data query tool."""

    custom_table: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(enabled_str="True", name="Data: {dataset_source_id}")
    )
    plotly_grid: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Plotly Grid: {dataset_source_id}"
        )
    )
    csv_file: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="True", name="Data (CSV): {dataset_source_id}.csv"
        )
    )
    plotly_graphs: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(enabled_str="True", name="Graph: {figure_title}")
    )
    json_query: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Query (JSON): {dataset_source_id}"
        )
    )
    python_code: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Python Code: {dataset_source_id}"
        )
    )


class DataQueryLLMModels(BaseYamlModel):
    datasets_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    dimensions_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    indicators_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    incomplete_queries_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    group_expander_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    named_entities_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    time_period_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    query_normalization_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)


class SpecialDimensionsProcessor(BaseYamlModel):
    id: str = Field(description="Unique identifier of the processor.")
    # alias: str = Field(description="Alias for dimensions.")
    type: SpecialDimensionsProcessorType = Field()
    llm_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    # TODO: top_k is specific to LHCL, move it to LHCL config subclass later
    top_k: int = Field(description="Number of candidates retrieved from vector search", default=50)
    prompt: SystemUserPrompt


class DataQueryDetails(BaseToolDetails):

    version: DataQueryVersion = DataQueryVersion.v2
    indexer_version: IndexerVersion = Field(
        default=IndexerVersion.semantic, description="The version of the indexer"
    )
    indicator_selection_version: IndicatorSelectionVersion = Field(
        default=IndicatorSelectionVersion.semantic_v4,
        description="The version of the indicator selection algorithm",
    )
    special_dimensions_processors: list[SpecialDimensionsProcessor] = Field(default_factory=list)
    llm_models: DataQueryLLMModels = Field(default_factory=DataQueryLLMModels)  # type: ignore
    prompts: DataQueryPrompts = Field(default_factory=DataQueryPrompts)  # type: ignore
    messages: DataQueryMessages = Field(default_factory=DataQueryMessages)  # type: ignore
    attachments: DataQueryAttachments = Field(default_factory=DataQueryAttachments)  # type: ignore
    tool_response_max_cells: int = Field(
        default=300,
        description=(
            "Maximum number of cells to include in the tool response. If the result exceeds this number, "
            "the data won't be included in the response shown to agent. The user will always see the data in the "
            "UI table, regardless of this limitation."
        ),
        ge=0,
    )
