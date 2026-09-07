import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError

from statgpt.common import schemas

_log = logging.getLogger(__name__)

# Postgres SQLSTATE for unique_violation. SQLAlchemy's asyncpg adapter copies the
# driver's sqlstate onto the DBAPI error it raises, so matching on it is stable across
# driver versions - unlike matching on the formatted exception message.
_UNIQUE_VIOLATION_SQLSTATE = "23505"

_PROBLEMS_IN_MESSAGE = 5
"""How many problems a one-line message names before falling back to a count."""


def _render_problem(problem: schemas.DiscoveryPayloadProblem) -> str:
    """Render a problem as `<where>: <what>`, using the most precise location it carries."""
    if problem.cell:
        where = problem.cell
    elif problem.row is not None:
        where = f"row {problem.row}"
    elif problem.index is not None:
        where = f"item {problem.index}"
    else:
        return problem.message
    return f"{where}: {problem.message}"


class AdminServiceError(Exception):
    """Base class for admin service errors mapped to HTTP responses in routers."""


def _is_unique_violation(e: IntegrityError) -> bool:
    return getattr(e.orig, "sqlstate", None) == _UNIQUE_VIOLATION_SQLSTATE


def raise_for_integrity_error(e: IntegrityError, conflict_detail: str) -> NoReturn:
    """Translate a DB integrity failure into an actionable HTTP error.

    A unique-constraint violation becomes a 409 carrying `conflict_detail`, instead of
    leaking a driver traceback as a 500. Any other integrity failure stays a 500.
    """
    _log.warning(e)

    if _is_unique_violation(e):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict_detail)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unknown db error"
    )


def raise_for_conflict(e: IntegrityError, conflict: AdminServiceError) -> NoReturn:
    """Re-raise a unique-constraint violation as a domain error, mapped to 409 at the edge.

    The domain-error counterpart of `raise_for_integrity_error`, for services that report
    through `AdminServiceError` rather than by raising `HTTPException` themselves. Any other
    integrity failure is a bug or an operational fault, so it propagates unchanged.
    """
    _log.warning(e)

    if _is_unique_violation(e):
        raise conflict from e

    raise e


@dataclass(frozen=True)
class BlockingDataset:
    dataset_id: int
    dataset_title: str
    channel_count: int

    @property
    def channels_label(self) -> str:
        suffix = "" if self.channel_count == 1 else "s"
        return f"{self.channel_count} channel{suffix}"

    @property
    def usage(self) -> str:
        return f"'{self.dataset_title}' ({self.channels_label})"


class DatasetInUseError(AdminServiceError):
    def __init__(self, blocking_datasets: list[BlockingDataset]) -> None:
        self.blocking_datasets = blocking_datasets
        super().__init__(f"Dataset(s) still used in channels: {self.usage_summary}")

    @property
    def usage_summary(self) -> str:
        return ", ".join(ds.usage for ds in self.blocking_datasets)


class DiscoveryNotFoundError(AdminServiceError):
    """A discovery record or indexing job addressed by id does not exist."""


class DiscoveryDatasetNotFoundError(DiscoveryNotFoundError):
    def __init__(self, item_id: int) -> None:
        super().__init__(f"Discovery dataset with id={item_id} not found.")


class DiscoveryIndexingJobNotFoundError(DiscoveryNotFoundError):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"Discovery indexing job with id={job_id} not found.")


class DiscoveryDatasetConflictError(AdminServiceError):
    """A discovery dataset with the same natural key already exists in the channel.

    Carries a message naming the colliding record(s) where the caller could work them out,
    so an admin need not bisect a rejected batch.
    """


class DiscoveryPayloadError(AdminServiceError):
    """A discovery dataset write is structurally unusable, so nothing was saved.

    Carries one problem per offending record rather than a single message, so the caller
    can see every row at fault instead of fixing them one request at a time.
    """

    def __init__(
        self, problems: list[schemas.DiscoveryPayloadProblem], truncated: bool = False
    ) -> None:
        self.problems = problems
        self.truncated = truncated
        super().__init__(self._message())

    def _message(self) -> str:
        """Summarize the problems inline.

        The structured `detail` below only reaches an HTTP caller. Everywhere else - a log
        line, or the `reason_for_failure` of an import job that carried a bad archive - it is
        ``str(exc)`` that survives, so it has to name a few of the offending records rather
        than only counting them.
        """
        summary = f"{len(self.problems)} structural problem(s) in the payload"
        shown = [_render_problem(problem) for problem in self.problems[:_PROBLEMS_IN_MESSAGE]]
        if not shown:
            return f"{summary}."
        remaining = len(self.problems) - len(shown)
        listed = "; ".join(shown) + (f"; and {remaining} more" if remaining > 0 else "")
        return f"{summary}: {listed}."

    @property
    def detail(self) -> schemas.DiscoveryPayloadErrorDetail:
        count = len(self.problems)
        suffix = "" if count == 1 else "s"
        return schemas.DiscoveryPayloadErrorDetail(
            message=f"The payload has {count} problem{suffix}; nothing was saved.",
            problems=self.problems,
            truncated=self.truncated,
        )


class DiscoveryUploadFormatError(AdminServiceError):
    """An uploaded file could not be read as a discovery workbook or CSV."""


class DiscoveryUploadTooLargeError(AdminServiceError):
    """An uploaded file exceeds the configured byte cap."""


class DiscoveryRagNotConfiguredError(AdminServiceError):
    """The channel does not say where its discovery records should be published.

    Reported when an indexing job is triggered, not from inside the run: an administrator
    who has not configured the target should be told so by the request, rather than by a job
    that accepts the trigger and then fails.
    """

    def __init__(self, channel_id: int) -> None:
        super().__init__(
            f"Channel {channel_id} has no discovery RAG configuration, so its discovery"
            f" datasets cannot be published. Set the Generic RAG application id in the"
            f" channel configuration (`details.discoveryDatasets.details.applicationId`) and"
            f" try again."
        )


class DiscoveryMetadataSchemaError(AdminServiceError):
    """The target RAG channel cannot filter on the metadata discovery search relies on.

    Its metadata schema is configured on the DIAL application, outside this codebase, so a
    run reports what is missing instead of publishing documents that search could never
    narrow down.
    """

    def __init__(self, missing_fields: Sequence[str], *, channel: str = "discovery RAG") -> None:
        fields = ", ".join(missing_fields)
        super().__init__(
            f"The {channel} channel does not declare these metadata field(s) as"
            f" filterable: {fields}. Add them to the Generic RAG application's"
            f" `metadata_schema` with `enable_filtering` set, then run the job again."
        )


class DiscoveryReferenceAreaIndexingError(AdminServiceError):
    """The reference-area vocabulary channel accepted a document and failed to index it.

    Fails the run, unlike a discovery record that cannot be indexed. A record's failure costs
    that one dataset its place in the results, while a missing vocabulary entry silently
    narrows away every dataset covering that area, for every query naming it - so it must not
    be reported as a completed job.
    """

    def __init__(self, values: Sequence[str]) -> None:
        listed = ", ".join(repr(value) for value in values)
        super().__init__(
            f"The reference-area RAG channel failed to index {len(values)} vocabulary"
            f" document(s): {listed}. The channel reports no reason for it. Until this is"
            f" resolved, queries naming those areas cannot be narrowed by reference area."
        )


class IndexingJobInProgressError(AdminServiceError):
    """An indexing job for the same channel is already queued or running.

    `job_id` is optional because the job can also be discovered by losing a race to the
    unique index, and it may have finished by the time the loser looks for it.
    """

    def __init__(self, channel_id: int, job_id: int | None = None) -> None:
        job = f"Indexing job {job_id}" if job_id is not None else "An indexing job"
        super().__init__(
            f"{job} is already in progress for channel {channel_id}. "
            f"Wait for it to finish before starting another one."
        )
