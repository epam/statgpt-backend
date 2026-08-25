from pydantic import ConfigDict

from .base import BaseYamlModel, DbDefaultBase
from .enums import PreprocessingStatusEnum


class DiscoveryIndexingJob(BaseYamlModel, DbDefaultBase):
    """A job record for tracking an indexing run over a channel's discovery datasets.

    A run re-validates every record of the channel, then reconciles the channel's discovery
    RAG documents with the verdicts: valid records are published, invalid ones have their
    documents withdrawn, and documents no record claims any more are removed.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    channel_id: int

    status: PreprocessingStatusEnum
    """Job execution status (QUEUED, IN_PROGRESS, COMPLETED, FAILED)."""

    details: str | None = None
    """What the run did, as a one-line summary of the counts below."""

    reason_for_failure: str | None = None
    """Why the run itself failed. Per-record failures do not fail the run."""

    records_total: int | None = None
    """Records the run evaluated (populated on COMPLETED)."""

    records_valid: int | None = None
    """Records that passed validation and were eligible for publishing."""

    records_invalid: int | None = None
    """Records that failed validation and were therefore not published."""

    documents_upserted: int | None = None
    """Documents published to the discovery index."""

    documents_deleted: int | None = None
    """Documents removed from the discovery index.

    Counts every removal: an invalid record's document being withdrawn, the old document of
    a record that was republished, and one no record claims any more.
    """
