from pydantic import BaseModel, Field

from common.utils import AttachmentResponse

from .base import DbDefaultBase
from .enums import JobType, PreprocessingStatusEnum


class Job(DbDefaultBase):
    """Import/export job."""

    type: JobType
    status: PreprocessingStatusEnum
    file: str | None = Field(description="URL to the file. Left for debugging purposes only.")
    channel_id: int | None
    reason_for_failure: str | None = Field(
        default=None, description="Reason for failure if the job has failed."
    )


class ClearJobsResult(BaseModel):
    """Result of clearing jobs."""

    reason_for_failure: str | None = Field(
        default=None, description="Reason for failure if the job has failed."
    )

    deleted_files: list[AttachmentResponse] = Field(
        description="List of deleted files in the Dial storage."
    )
    deleted_jobs: list[Job] = Field(description="List of deleted jobs.")
