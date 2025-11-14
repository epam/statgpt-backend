from pydantic import BaseModel, Field

from common.data.base.dimension import DimensionProcessingType


class GitVersionResponse(BaseModel):
    git_commit: str = Field()


class SettingsResponse(BaseModel):
    enable_dev_commands: bool = Field()
    enable_direct_tool_calls: bool = Field()
    git_commit: str = Field()


class DimTypesResponse(BaseModel):
    channel_name: str
    n_datasets: int
    dataset_dim_types: dict[str, dict[str, DimensionProcessingType]] = Field(
        default_factory=dict, description="dataset -> dimension -> types mapping"
    )
