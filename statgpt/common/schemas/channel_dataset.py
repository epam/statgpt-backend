from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from .auto_update import AutoUpdateJob
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
    indexing_config_hash: str | None
    structure_metadata: dict | None
    structure_hash: str | None
    indicator_dimensions_hash: str | None
    non_indicator_dimensions_hash: str | None
    special_dimensions_hash: str | None
    resolved_config: dict | None = Field(
        description="Resolved dataset configuration at indexing time (dynamic URN values resolved to concrete values)",
    )
    indexing_stats: dict | None = Field(
        default=None,
        description="Normalization/harmonization error statistics collected during indexing",
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
    clearing_status: PreprocessingStatusEnum = Field(
        description="The clearing status of the channel dataset."
    )

    last_completed_version: ChannelDatasetVersion | None
    previous_completed_version: ChannelDatasetVersion | None = Field(
        description=(
            "The last completed version before the latest completed version, if any."
            " This version will be used for rollback (if needed)."
        )
    )
    latest_version: ChannelDatasetVersion | None
    last_auto_update_job: AutoUpdateJob | None = Field(
        description=(
            "The most recent auto-update job for this channel dataset,"
            " or null if no auto-update job has ever been created."
        )
    )


class ChannelDatasetExpandedWithLastUpdate(ChannelDatasetExpanded):
    last_updated_at: datetime | None = Field(
        default=None,
        description=(
            "Timestamp of when the dataset's data was last updated at the source,"
            " resolved via the provider-specific DataSet.updated_at logic."
            " Null when the provider exposes no such value or the dataset is offline."
        ),
    )


class BaseChange(BaseModel):
    message: str
    last_version_hash: str | None
    actual_hash: str | None


class ConfigChange(BaseChange):
    pass


class DataChange(BaseChange):
    pass


class StructureChange(BaseChange):
    details: dict


class ChangesBetweenVersionAndActualData(BaseModel):
    config_changes: list[ConfigChange] = Field(
        description=(
            "List of changes between the config of the dataset used during the indexing"
            " of the latest completed version and the actual config."
        )
    )
    data_changes: list[DataChange] = Field(
        description="List of changes between the indexed data of the latest completed version and the actual data."
    )
    structure_change: StructureChange | None = Field(
        description="Details about the structure change, if any."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_changes(self) -> bool:
        """Indicates whether there are any changes between the latest completed version and the actual data/config."""
        return bool(self.config_changes or self.data_changes or self.structure_change)
