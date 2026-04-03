from pydantic import ConfigDict

from .base import DbDefaultBase
from .enums import AutoUpdateResult, PreprocessingStatusEnum


class AutoUpdateJob(DbDefaultBase):
    """Schema for an auto-update job record."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    channel_dataset_id: int
    base_version_id: int | None
    """The base version used for comparison (last completed version at job creation time)."""

    created_version_id: int | None
    """The newly created version after reindexing (set when reindex is triggered)."""

    status: PreprocessingStatusEnum
    """Job execution status (QUEUED, IN_PROGRESS, COMPLETED, FAILED)."""

    result: AutoUpdateResult | None
    """Outcome of the auto-update (set when job completes)."""

    details: str | None
    """Additional details about the job execution, such as what changes were detected."""

    reason_for_failure: str | None
