from pydantic import BaseModel, Field

from statgpt.common.schemas.channel_dataset import ChannelDatasetExpanded
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
