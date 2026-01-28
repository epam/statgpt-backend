from pydantic import BaseModel, Field

from .channel import Channel
from .channel_dataset import ChannelDatasetVersion
from .dataset import DataSet
from .enums import ChannelDatasetUpdateStatus


class ChannelDatasetUpdateResult(BaseModel):
    channel_dataset_id: int
    status: ChannelDatasetUpdateStatus
    channel: Channel
    new_version: ChannelDatasetVersion | None = Field(default=None)


class DataSetUpdateResponse(BaseModel):
    dataset: DataSet
    channel_results: list[ChannelDatasetUpdateResult] = Field(default_factory=list)
