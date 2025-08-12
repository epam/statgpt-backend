from pydantic import Field, TypeAdapter, field_validator

from common.config.utils import replace_env

from .base import BaseYamlModel
from .enums import DataQueryVersion, IndexerVersion, IndicatorSelectionVersion
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
    name: str = Field(description="Attachment name template")

    @field_validator('enabled_str', mode='after')
    @classmethod
    def validate_enabled(cls, enabled: str) -> str:
        """Validate the `enabled` field to ensure it can return a boolean value."""
        try:
            bool_from_str(enabled)
        except Exception as e:
            raise ValueError(f"Invalid value for enabled_str: {enabled}. Error: {e}")
        return enabled

    @property
    def enabled(self) -> bool:
        return bool_from_str(self.enabled_str)

    def __bool__(self) -> bool:
        """Return True if the attachment is enabled."""
        return self.enabled


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


class DataQueryDetails(BaseToolDetails):

    version: DataQueryVersion = DataQueryVersion.v2
    indexer_version: IndexerVersion = Field(
        default=IndexerVersion.semantic, description="The version of the indexer"
    )
    indicator_selection_version: IndicatorSelectionVersion = Field(
        default=IndicatorSelectionVersion.semantic_v4,
        description="The version of the indicator selection algorithm",
    )
    prompts: DataQueryPrompts = Field(default_factory=DataQueryPrompts)  # type: ignore
    messages: DataQueryMessages = Field(default_factory=DataQueryMessages)  # type: ignore
    attachments: DataQueryAttachments = Field(default_factory=DataQueryAttachments)  # type: ignore
