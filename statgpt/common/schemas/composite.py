from pydantic import BaseModel, Field

from .auditable import Auditable
from .channel import Channel
from .channel_dataset import ChannelDatasetVersion
from .dataset import DataSet
from .enums import ChannelDatasetUpdateStatus


class ChannelDatasetUpdateResult(BaseModel):
    channel_dataset_id: int
    status: ChannelDatasetUpdateStatus
    channel: Channel
    new_version: ChannelDatasetVersion | None = Field(default=None)


class DataSetUpdateResponse(BaseModel, Auditable):
    dataset: DataSet
    channel_results: list[ChannelDatasetUpdateResult] = Field(default_factory=list)

    def get_entity_id(self) -> str | None:
        return self.dataset.get_entity_id()

    def get_entity_name(self) -> str | None:
        return self.dataset.get_entity_name()

    def get_state_after(self) -> dict | None:
        return self.dataset.get_state_after()

    def get_item_id(self) -> int | None:
        return self.dataset.id
