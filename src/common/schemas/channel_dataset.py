from .base import DbDefaultBase
from .dataset import DataSet
from .enums import PreprocessingStatusEnum


class ChannelDatasetBase(DbDefaultBase):
    channel_id: int
    dataset_id: int
    preprocessing_status: PreprocessingStatusEnum


class ChannelDatasetExpanded(ChannelDatasetBase):
    dataset: DataSet
