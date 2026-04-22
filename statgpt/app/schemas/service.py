from pydantic import BaseModel, Field

from statgpt.common.schemas.channel_dataset import ChannelDatasetExpanded
from statgpt.common.schemas.query import JsonQueryWithMetadata
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
    datasets: list[ChannelDatasetExpanded] = Field(default_factory=list)


class GeneratePythonCodeRequest(BaseModel):
    queries: list[JsonQueryWithMetadata] = Field(
        min_length=1,
        description="List of JSON queries with metadata to generate Python code for",
    )


class GeneratePythonCodeResponse(BaseModel):
    python_code: str = Field(description="Generated Python code using the sdmx1 library")
