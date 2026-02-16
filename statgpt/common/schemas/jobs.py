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

    def get_entity_id(self) -> str:
        return str(self.type)

    def get_entity_name(self) -> str:
        if self.type == JobType.EXPORT:
            return f"Export channel(id={self.channel_id})"
        if self.type == JobType.IMPORT:
            return f"Import channel from {self.file!r}"
        raise RuntimeError(f"Unknown job type: {self.type}")

    def get_state_after(self) -> dict:
        return self.model_dump(mode='json', include={"id", "file", "channel_id"})

    def get_item_id(self) -> int:
        return self.id


class ClearJobsResult(BaseModel):
    """Result of clearing jobs."""

    reason_for_failure: str | None = Field(
        default=None, description="Reason for failure if the job has failed."
    )

    deleted_files: list[AttachmentResponse] = Field(
        description="List of deleted files in the Dial storage."
    )
    deleted_jobs: list[Job] = Field(description="List of deleted jobs.")
