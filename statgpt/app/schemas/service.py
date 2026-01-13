from pydantic import BaseModel, Field

from statgpt.common.data.base import DimensionType
from statgpt.common.schemas.channel_dataset import ChannelDatasetExpanded
from statgpt.common.schemas.tools import BaseToolConfig


class GitVersionResponse(BaseModel):
    git_commit: str = Field()


class SettingsResponse(BaseModel):
    enable_dev_commands: bool = Field()
    enable_direct_tool_calls: bool = Field()
    git_commit: str = Field()


class DimTypesResponse(BaseModel):
    channel_name: str
    n_datasets: int
    dataset_dim_types: dict[str, dict[str, DimensionType]] = Field(
        default_factory=dict, description="dataset -> dimension -> types mapping"
    )


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
