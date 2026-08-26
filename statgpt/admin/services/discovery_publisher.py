"""Publishing discovery dataset records into a channel's Generic RAG channel.

Maps a stored record onto one RAG document and reconciles what the channel holds with what
the records say it should hold. Run by an indexing job, after validation.

Nothing here derives, paraphrases or summarizes: the document carries the submitted metadata
verbatim, so how discoverable a dataset is stays a property of the source's own metadata
rather than of an algorithm here.
"""

import asyncio
import hashlib
import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import NamedTuple

from statgpt.admin.settings.discovery import DiscoveryPublishSettings
from statgpt.common import models, schemas
from statgpt.common.services import GenericRagIngestionClient, RecordKey, record_key
from statgpt.common.utils import (
    MediaTypes,
    async_utils,
    escape_invalid_filename_chars,
    format_exception_reason,
    get_ts_utcnow,
)

from .discovery_validation import DiscoveryRecord
from .exceptions import DiscoveryMetadataSchemaError

_log = logging.getLogger(__name__)

_SETTINGS = DiscoveryPublishSettings()


class _RenderedField(NamedTuple):
    """One `DiscoveryRecord` field and the heading it is rendered under."""

    label: str
    attribute: str


_SUMMARY_FIELDS: tuple[_RenderedField, ...] = (
    _RenderedField("Agency / organization", "agency"),
    _RenderedField("Dataset ID", "dataset_id"),
    _RenderedField("Reference area / country", "reference_area"),
    _RenderedField("Regional coverage", "regional_coverage"),
    _RenderedField("Excluded regional values", "excluded_regional_values"),
    _RenderedField("Time coverage", "time_coverage"),
    _RenderedField("Frequency coverage", "frequency_coverage"),
    _RenderedField("Dataset URL", "url"),
)
"""Fields rendered as the document's leading bullet list, in workbook order."""

_SECTION_FIELDS: tuple[_RenderedField, ...] = (
    _RenderedField("Description", "description"),
    _RenderedField("Indicators coverage", "indicators_coverage"),
    _RenderedField("Relevant indicators not present in the dataset", "missing_indicators"),
)
"""Fields rendered as their own section, being prose or long ';'-separated lists."""

_MAX_FILENAME_STEM = 100
_FALLBACK_FILENAME_STEM = "discovery-dataset"
_FILENAME_DIGEST_CHARS = 12

_SETTLE_INTERVAL_SECONDS = 1.0
"""How long to wait before the first re-listing of the settle phase."""

_SETTLE_MAX_INTERVAL_SECONDS = 5.0
"""The longest the settle phase waits between two passes.

The interval doubles up to this, so a channel that indexes a handful of small documents is
done in about a second, while one still working after half a minute is not asked every
second for as long as it takes.
"""

_INDEXING_FAILED = (
    "The discovery RAG channel accepted this document and then failed to parse or index it,"
    " leaving it in the 'error' state. The channel reports no reason for it. The next"
    " indexing run publishes the record again."
)
"""Why a record that was uploaded successfully still ends a run unpublished."""


def render_document_body(record: DiscoveryRecord) -> str:
    """Render a record as the markdown document the RAG channel indexes.

    Every workbook field goes in, including the ones that also travel as metadata: metadata
    is what search filters on, the body is what it retrieves over. Leaving `name` or
    `indicators_coverage` out of the body would hide from search exactly what a user's
    question matches on.

    An empty field is omitted rather than rendered as an empty label, so a sparse record
    does not fill the index with headings.
    """
    lines = [f"# {record.name or record.dataset_id}", ""]
    lines.extend(
        f"- **{rendered.label}:** {value}"
        for rendered in _SUMMARY_FIELDS
        if (value := getattr(record, rendered.attribute))
    )
    for rendered in _SECTION_FIELDS:
        if value := getattr(record, rendered.attribute):
            lines.extend(["", f"## {rendered.label}", "", value])
    return "\n".join(lines) + "\n"


def document_filename(record: DiscoveryRecord, channel: str, grade: schemas.DiscoveryGrade) -> str:
    """Name the uploaded file, which the channel echoes back as the document's display name.

    The name has to identify the record on its own. The service derives a document's storage
    path from the name alone - `documents/{sha1(basename.lower())}{ext}` - so two records
    whose names agree are one document, and one of them silently disappears from the index.
    A readable name cannot carry that weight: it drops the channel and the grade, path
    characters are escaped to the same `_`, and it is capped in length, so three separate ways
    exist for records that are not the same to end up named the same.

    So the readable part is a label, and a digest of what actually identifies the record makes
    the name unique. The digest covers the folded natural key, so the label is the only part a
    re-spelled record changes - and a document is matched by its metadata rather than by its
    name, so the renamed record's old document is found and replaced like any other edit.
    """
    digest = hashlib.sha256(
        "\0".join([grade, channel, *record_key(record.agency, record.dataset_id)]).encode()
    ).hexdigest()[:_FILENAME_DIGEST_CHARS]
    label = escape_invalid_filename_chars(f"{record.agency} - {record.dataset_id}")
    stem = label[:_MAX_FILENAME_STEM].strip() or _FALLBACK_FILENAME_STEM
    return f"{stem} [{digest}].md"


@dataclass(frozen=True)
class DocumentFile:
    """The file a record is uploaded as."""

    filename: str
    content: bytes
    mime_type: str


def render_document(
    record: DiscoveryRecord, channel: str, grade: schemas.DiscoveryGrade
) -> DocumentFile:
    """Render a record as the file the RAG channel ingests.

    The one place the document's format is decided. The channel parses what it is given with
    `unstructured`, whose `by_title` chunking follows the markdown headings below, so the
    record arrives as the sections it was written as.
    """
    return DocumentFile(
        filename=document_filename(record, channel, grade),
        content=render_document_body(record).encode("utf-8"),
        mime_type=MediaTypes.MARKDOWN,
    )


def build_metadata(
    record: DiscoveryRecord, channel: str, grade: schemas.DiscoveryGrade
) -> schemas.DiscoveryDocumentMetadata:
    """Build the document metadata: every workbook field except the description.

    The description is the one field left out - it is prose, it is in the body, and it is
    nothing search would filter on.
    """
    return schemas.DiscoveryDocumentMetadata(
        grade=grade,
        statgpt_channel=channel,
        agency=record.agency,
        reference_area=record.reference_area,
        dataset_id=record.dataset_id,
        name=record.name,
        url=record.url,
        regional_coverage=record.regional_coverage,
        excluded_regional_values=record.excluded_regional_values,
        time_coverage=record.time_coverage,
        frequency_coverage=record.frequency_coverage,
        indicators_coverage=record.indicators_coverage,
        missing_indicators=record.missing_indicators,
    )


def document_key(document: schemas.GenericRagDocument) -> RecordKey | None:
    """The natural key a document claims, or None if it does not claim one.

    Folded by the same helper the database's generated columns use, so a document matches
    the record that produced it however its casing or spacing was submitted.
    """
    agency = document.metadata.get("agency")
    dataset_id = document.metadata.get("dataset_id")
    if not isinstance(agency, str) or not isinstance(dataset_id, str):
        return None
    key = record_key(agency, dataset_id)
    return key if all(key) else None


def is_channel_document(
    document: schemas.GenericRagDocument, channel: str, grade: schemas.DiscoveryGrade
) -> bool:
    """Whether this document was published by this grade, for this channel.

    The channel is identified by its deployment id rather than its row id, so a document stays
    recognizable as this channel's after the channel is moved to another environment, where the
    row id would differ.
    """
    metadata = document.metadata
    return metadata.get("grade") == grade and metadata.get("statgpt_channel") == channel


async def withdraw_documents(
    client: GenericRagIngestionClient,
    channel: str,
    keys: Collection[RecordKey],
    *,
    grade: schemas.DiscoveryGrade = schemas.DiscoveryGrade.C,
    concurrency: int | None = None,
) -> int:
    """Delete this channel's documents for `keys`, returning how many went.

    What a record's deletion needs, as opposed to what an indexing run needs. It reconciles
    nothing: a key not in `keys` is left alone, whatever state it is in, so removing one record
    cannot disturb another one's document.

    Two differences from the reconciliation the publisher does, both following from that:

    Every document claiming one of the keys goes, not one per key. `_load_documents` keeps the
    highest id of a duplicated key and treats the rest as orphans for the run to sweep; here
    there is no run to follow, and a leftover duplicate would keep serving a record that was
    deleted - which is the whole point of withdrawing.

    A failure propagates. `_delete_orphan` swallows one because an orphan belongs to no record
    and the next run retries it; here the caller is about to drop the row that is the last
    thing pointing at the document, so it has to learn that the document is still there.
    """
    if not keys:
        return 0

    documents = [
        document
        for document in await client.list_documents()
        if is_channel_document(document, channel, grade) and document_key(document) in keys
    ]
    try:
        await async_utils.gather_with_concurrency(
            concurrency if concurrency is not None else _SETTINGS.concurrency,
            *(client.delete_document(document.id) for document in documents),
        )
    except BaseExceptionGroup as group:
        # The group is an artifact of the task group inside `gather_with_concurrency`, and it
        # matches neither an `except GenericRagIngestionError` nor an exception handler
        # registered on it. The remaining leaves are the same outage counted once per document.
        raise _first_leaf(group) from group
    return len(documents)


def _first_leaf(exc: BaseException) -> BaseException:
    """The first real exception inside a possibly nested exception group."""
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


@dataclass
class PublishCounts:
    """What a publish stage did. Reported on the job row."""

    upserted: int = 0
    """Records published, whether for the first time or refreshed."""

    deleted: int = 0
    """Documents removed: withdrawn invalid records, rebuilt records, and orphans."""

    skipped: int = 0
    """Records already indexed and unchanged, so left alone."""

    failed: int = 0
    """Records whose publish attempt failed. Recorded on the record as well."""

    def add(self, other: "PublishCounts") -> None:
        """Fold one record's counts into the run's."""
        self.upserted += other.upserted
        self.deleted += other.deleted
        self.skipped += other.skipped
        self.failed += other.failed


@dataclass(frozen=True)
class _AwaitedDocument:
    """A document the channel had not finished indexing when it answered the upload."""

    record: models.DiscoveryDataset
    document_id: int


@dataclass
class _RecordOutcome:
    """What one record's turn did, folded into the run's counts by the caller.

    One per record rather than a counter every turn increments: the turns run concurrently,
    and a result each of them owns is both easier to follow and easier to assert on.
    """

    counts: PublishCounts = field(default_factory=PublishCounts)
    awaiting: _AwaitedDocument | None = None


class DiscoveryPublisher:
    """Reconciles a channel's discovery records with the documents of its RAG channel.

    One instance per run. Mutates the records it is given - indexing status, timestamp and
    error - and leaves persisting them to the caller.
    """

    def __init__(
        self,
        client: GenericRagIngestionClient,
        channel: str,
        grade: schemas.DiscoveryGrade = schemas.DiscoveryGrade.C,
        *,
        force: bool = False,
        concurrency: int | None = None,
        settle_timeout_seconds: float | None = None,
        settle_interval_seconds: float = _SETTLE_INTERVAL_SECONDS,
        settle_max_interval_seconds: float = _SETTLE_MAX_INTERVAL_SECONDS,
    ) -> None:
        """`force` republishes every valid record, whatever its stored indexing status.

        The concurrency and the timings default to the configured settings; they are
        arguments so that a test does not have to wait out a five-minute deadline.
        """
        self._client = client
        self._channel = channel
        self._grade = grade
        self._force = force
        self._concurrency = concurrency if concurrency is not None else _SETTINGS.concurrency
        self._settle_timeout = (
            _SETTINGS.settle_timeout_seconds
            if settle_timeout_seconds is None
            else settle_timeout_seconds
        )
        self._settle_interval = settle_interval_seconds
        self._settle_max_interval = settle_max_interval_seconds

    async def verify_metadata_schema(self) -> None:
        """Refuse to publish into a channel that cannot filter on what search needs.

        What must be filterable is declared by `DiscoveryDocumentMetadata` itself, so the
        check and the schema an administrator configures come from one definition.
        """
        declared = (await self._client.get_metadata_schema()).filterable_fields
        missing = sorted(schemas.DiscoveryDocumentMetadata.filterable_fields() - declared)
        if missing:
            raise DiscoveryMetadataSchemaError(missing)

    async def publish(self, records: Sequence[models.DiscoveryDataset]) -> PublishCounts:
        """Bring the RAG channel in line with `records`.

        A failure on one record is recorded on that record and the run continues: one
        unpublishable dataset must not keep a channel's other datasets out of the index.
        """
        counts = PublishCounts()
        by_key, unclaimed = await self._load_documents()

        # Pair every record with its document before any of them is published: the turns run
        # concurrently, and none of them may find this map being mutated underneath it.
        pairs = [
            (record, by_key.pop(record_key(record.agency, record.dataset_id), None))
            for record in records
        ]
        outcomes: list[_RecordOutcome] = await async_utils.gather_with_concurrency(
            self._concurrency,
            *(self._apply_safely(record, document) for record, document in pairs),
        )

        awaited: list[_AwaitedDocument] = []
        for outcome in outcomes:
            counts.add(outcome.counts)
            if outcome.awaiting is not None:
                awaited.append(outcome.awaiting)

        # Whatever is left claims a key no record has: a record deleted through the API, a
        # key that was corrected, or debris from a run that died mid-flight.
        await self._delete_all(list(by_key.values()) + unclaimed, counts)
        await self._settle(awaited, counts)
        return counts

    async def _load_documents(
        self,
    ) -> tuple[dict[RecordKey, schemas.GenericRagDocument], list[schemas.GenericRagDocument]]:
        """Index this channel's documents by natural key.

        Returns the index and the documents that cannot be indexed into it - one claiming no
        key, or a duplicate of one already seen. Both are ours and belong to no record, so
        they are deleted; keeping the highest id of a duplicated key means the survivor is
        the most recently uploaded one.

        Documents of other channels and other grades are ignored entirely: one RAG channel
        can serve several StatGPT channels, and Grade B will publish into the same one.
        """
        by_key: dict[RecordKey, schemas.GenericRagDocument] = {}
        unclaimed: list[schemas.GenericRagDocument] = []

        for document in await self._client.list_documents():
            if not self._is_ours(document):
                continue
            key = document_key(document)
            if key is None:
                unclaimed.append(document)
                continue
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = document
            elif document.id > previous.id:
                by_key[key] = document
                unclaimed.append(previous)
            else:
                unclaimed.append(document)

        return by_key, unclaimed

    def _is_ours(self, document: schemas.GenericRagDocument) -> bool:
        """Whether this document was published by this grade, for this channel."""
        return is_channel_document(document, self._channel, self._grade)

    async def _apply_safely(
        self, record: models.DiscoveryDataset, document: schemas.GenericRagDocument | None
    ) -> _RecordOutcome:
        """One record's turn, which must not raise.

        The turns share a task group, so an exception escaping here would cancel every other
        record's turn - the opposite of what a per-record failure is supposed to mean.
        """
        outcome = _RecordOutcome()
        try:
            await self._apply(record, document, outcome)
        except Exception as e:
            _log.exception(
                f"Failed to publish discovery record {record.id}"
                f" (agency={record.agency!r} dataset_id={record.dataset_id!r})"
            )
            record.indexing_status = schemas.DiscoveryIndexingStatus.FAILED
            record.index_error = format_exception_reason(e)
            outcome.counts.failed += 1
        return outcome

    async def _apply(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        outcome: _RecordOutcome,
    ) -> None:
        """Bring one record and its document into agreement, filling in `outcome`.

        The outcome is filled in as the work happens rather than returned at the end, so a
        step that fails halfway through does not take the count of what already succeeded
        with it - a document deleted before a failing upload is still a document deleted.
        """
        if record.validation_status is not schemas.DiscoveryValidationStatus.VALID:
            await self._withdraw(record, document, outcome)
            return

        already_published = (
            not self._force
            and record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED
            and document is not None
            and not document.is_failed
        )
        if already_published:
            outcome.counts.skipped += 1
            return

        await self._publish_record(record, document, outcome)

    async def _withdraw(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        outcome: _RecordOutcome,
    ) -> None:
        """Take an invalid record out of the index, then mark it unpublished.

        A record that fails validation is not indexed, so one that was indexed before has to
        be removed - otherwise a verdict only changes what the admin portal shows while the
        index keeps serving the record.

        The document goes first, the status second. A crash in between leaves a record
        claiming to be indexed with no document behind it, which the next run repairs; the
        opposite order strands a live document behind a record that reads as unpublished.
        """
        if document is not None:
            await self._client.delete_document(document.id)
            outcome.counts.deleted += 1

        record.indexing_status = schemas.DiscoveryIndexingStatus.NEW
        record.indexed_at = None
        record.index_error = None

    async def _publish_record(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        outcome: _RecordOutcome,
    ) -> None:
        file = render_document(record, self._channel, self._grade)
        metadata = build_metadata(record, self._channel, self._grade)

        if document is not None and self._replaces_in_place(document, file.filename):
            published = await self._client.update_document(
                document.id,
                filename=file.filename,
                content=file.content,
                mime_type=file.mime_type,
                metadata=metadata,
            )
        else:
            if document is not None:
                await self._client.delete_document(document.id)
                outcome.counts.deleted += 1
            published = await self._client.upload_document(
                filename=file.filename,
                content=file.content,
                mime_type=file.mime_type,
                metadata=metadata,
                # The file name is derived from this record's natural key, so whatever sits
                # under it belongs to this record and nothing else is entitled to it.
                # Refusing to overwrite would let a file that outlived its document - after
                # the channel's database was reset, say - block this record for good.
                overwrite=True,
            )

        if published.is_failed:
            # A channel that indexes inline has already decided by the time it answers.
            self._mark_index_failure(record)
            outcome.counts.failed += 1
            return

        record.indexing_status = schemas.DiscoveryIndexingStatus.INDEXED
        record.indexed_at = get_ts_utcnow()
        record.index_error = None
        outcome.counts.upserted += 1
        if not published.is_terminal:
            outcome.awaiting = _AwaitedDocument(record, published.id)

    def _replaces_in_place(self, document: schemas.GenericRagDocument, filename: str) -> bool:
        """Whether this record's document can be refreshed rather than rebuilt.

        Three things rule it out, and they share one cause: an update is only ever a content
        refresh, so anything that needs the document itself rebuilt has to delete it.

        A forced run wants the document built again from nothing - an update sends the same
        bytes for a record that has not changed, the service compares etags, and nothing is
        re-parsed under whatever the channel is configured with now. A document already in
        `error` needs exactly that re-parse, and for an unchanged record the etag match would
        deny it: the record would keep failing every run, with only a forced run able to clear
        it. And an update cannot rename - the service keeps the document's display name - so a
        record whose label has changed would carry the old one for as long as the document
        lives.
        """
        return not self._force and not document.is_failed and document.display_name == filename

    @staticmethod
    def _mark_index_failure(record: models.DiscoveryDataset) -> None:
        """Record that the channel took the document and then failed to index it.

        `indexed_at` is left as it stands: it says when the record was last uploaded
        successfully, which is still true and reads usefully beside the failure. Only a
        withdrawal clears it, where the document really is gone.
        """
        record.indexing_status = schemas.DiscoveryIndexingStatus.FAILED
        record.index_error = _INDEXING_FAILED

    async def _delete_all(
        self, documents: Iterable[schemas.GenericRagDocument], counts: PublishCounts
    ) -> None:
        """Delete every document no record claims."""
        removed = await async_utils.gather_with_concurrency(
            self._concurrency, *(self._delete_orphan(document) for document in documents)
        )
        counts.deleted += sum(removed)

    async def _delete_orphan(self, document: schemas.GenericRagDocument) -> int:
        """Delete one unclaimed document, reporting whether it went.

        A failure here belongs to no record, so it is logged and left for the next run
        instead of failing the job: an undeleted stale document is a retrieval nuisance, not
        a reason to report that publishing did not happen.
        """
        try:
            await self._client.delete_document(document.id)
            return 1
        except Exception:
            _log.exception(
                f"Failed to delete orphaned discovery document {document.id}"
                f" ({document.display_name!r}) from channel {self._channel}"
            )
            return 0

    async def _settle(self, awaited: list[_AwaitedDocument], counts: PublishCounts) -> None:
        """Wait for what this run published to be indexed, and report what was not.

        An upload response only says the file was stored. A channel that indexes in the
        background answers long before the document is retrievable, and one that cannot parse
        it leaves it in `error`, holding no content at all. Without this the run reports every
        record published, the record reads INDEXED, and only the next run notices - silently,
        and possibly days later.

        A document still in flight when the deadline passes keeps its INDEXED status for the
        next run to judge, which is where this stood before.
        """
        pending = {item.document_id: item.record for item in awaited}
        if not pending or self._settle_timeout <= 0:
            return

        deadline = monotonic() + self._settle_timeout
        interval = self._settle_interval
        while True:
            try:
                listed = {document.id: document for document in await self._client.list_documents()}
            except Exception:
                # The publish results are established. Losing them to a failure of the
                # confirmation would be worse than confirming nothing.
                _log.exception(
                    f"Could not confirm the documents published to channel {self._channel}"
                )
                return

            for document_id, record in list(pending.items()):
                document = listed.get(document_id)
                if document is None:
                    _log.warning(
                        f"Discovery document {document_id} of channel {self._channel}"
                        f" disappeared while it was being indexed"
                    )
                elif document.is_failed:
                    self._mark_index_failure(record)
                    counts.failed += 1
                    # It was counted as published on upload, and it is not in the index.
                    counts.upserted -= 1
                elif not document.is_terminal:
                    continue
                del pending[document_id]

            if not pending:
                return
            if monotonic() >= deadline:
                _log.warning(
                    f"{len(pending)} discovery document(s) of channel {self._channel} were"
                    f" still being indexed after {self._settle_timeout}s;"
                    f" the next run will judge them"
                )
                return
            await asyncio.sleep(interval)
            interval = min(self._settle_max_interval, interval * 2)
