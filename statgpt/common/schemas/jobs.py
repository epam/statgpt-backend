from pydantic import BaseModel, Field

from statgpt.common.utils import AttachmentResponse

from .auditable import Auditable
from .base import DbDefaultBase
from .enums import JobType, PreprocessingStatusEnum


class Job(DbDefaultBase, Auditable):
    """Import/export job."""

    type: JobType
    status: PreprocessingStatusEnum
    file: str | None = Field(description="URL to the file. Left for debugging purposes only.")
    channel_id: int | None
    reason_for_failure: str | None = Field(
        default=None, description="Reason for failure if the job has failed."
    )

    def get_entity_id(self) -> str | None:
        return str(self.channel_id) if self.channel_id is not None else None

    def get_entity_name(self) -> str | None:
        return str(self.channel_id) if self.channel_id is not None else None


class ClearJobsResult(BaseModel):
    """Result of clearing jobs."""

    reason_for_failure: str | None = Field(
        default=None, description="Reason for failure if the job has failed."
    )

    deleted_files: list[AttachmentResponse] = Field(
        description="List of deleted files in the Dial storage."
    )
    deleted_jobs: list[Job] = Field(description="List of deleted jobs.")
