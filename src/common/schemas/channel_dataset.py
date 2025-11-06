from pydantic import Field

from .base import DbDefaultBase
from .dataset import DataSet
from .enums import PreprocessingStatusEnum


class ChannelDatasetBase(DbDefaultBase):
    channel_id: int
    dataset_id: int


class ChannelDatasetVersion(DbDefaultBase):
    channel_dataset_id: int
    version: int = Field(
        description=(
            "The version number starts at 1 and increments by 1 for each channel dataset"
            " independently of each other. This field is recommended to be displayed to users."
        )
    )
    preprocessing_status: PreprocessingStatusEnum
    creation_reason: str
    reason_for_failure: str | None
    pointer_to: int | None = Field(
        description=(
            "If this version is a rollback, this field points to the version which contains the data."
            " This may be a different version than the one used to roll back,"
            " if the rollback version was also a rollback to a previous version."
        )
    )

    @property
    def version_data_id(self) -> int:
        """The ID of the version which contains the actual data for this version."""
        return self.pointer_to if self.pointer_to is not None else self.id


class ChannelDatasetExpanded(ChannelDatasetBase):
    dataset: DataSet
    preprocessing_status: PreprocessingStatusEnum = Field(
        description="The preprocessing status of the latest version."
    )

    last_completed_version: ChannelDatasetVersion | None
    previous_completed_version: ChannelDatasetVersion | None = Field(
        description=(
            "The last completed version before the latest completed version, if any."
            " This version will be used for rollback (if needed)."
        )
    )
    latest_version: ChannelDatasetVersion | None
