from pydantic import BaseModel, Field, field_validator

from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.channel_dataset import ChannelDatasetExpandedWithLastUpdatedAt
from statgpt.common.schemas.tools import BaseToolConfig


class SettingsResponse(BaseModel):
    enable_dev_commands: bool = Field()
    enable_direct_tool_calls: bool = Field()
    git_commit: str = Field()


class ChannelMetadataResponse(BaseModel):
    deployment_id: str
    title: str
    description: str = ""
    locale: str
    country_named_entity_type: str
    tools: list[BaseToolConfig] = Field(default_factory=list)


class ChannelDatasetsMetadataResponse(BaseModel):
    deployment_id: str
    title: str
    n_datasets: int
    datasets: list[ChannelDatasetExpandedWithLastUpdatedAt] = Field(default_factory=list)


class GeneratePythonCodeRequest(BaseModel):
    queries: list[AppJsonQueryWithMetadata] = Field(
        min_length=1,
        max_length=64,
        description="List of JSON queries with metadata to generate Python code for",
    )

    @field_validator("queries")
    @classmethod
    def _validate_at_least_one_enabled_query(
        cls, value: list[AppJsonQueryWithMetadata]
    ) -> list[AppJsonQueryWithMetadata]:
        if all(query.disabled for query in value):
            raise ValueError("All queries are disabled; at least one enabled query is required")
        return value


class GeneratePythonCodeResponse(BaseModel):
    python_code: str = Field(description="Generated Python code using the sdmx1 library")
