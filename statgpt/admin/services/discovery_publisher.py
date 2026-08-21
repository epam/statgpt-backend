"""Publishing discovery dataset records into a channel's Generic RAG channel.

Maps a stored record onto one RAG document and reconciles what the channel holds with what
the records say it should hold. Run by an indexing job, after validation.

Nothing here derives, paraphrases or summarizes: the document carries the submitted metadata
verbatim, so how discoverable a dataset is stays a property of the source's own metadata
rather than of an algorithm here.
"""

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from statgpt.common import models, schemas
from statgpt.common.services import GenericRagIngestionClient, RecordKey, record_key
from statgpt.common.utils import (
    MediaTypes,
    escape_invalid_filename_chars,
    format_exception_reason,
    get_ts_utcnow,
)

from .discovery_validation import DiscoveryRecord
from .exceptions import DiscoveryMetadataSchemaError

_log = logging.getLogger(__name__)

_SUMMARY_FIELDS: tuple[tuple[str, str], ...] = (
    ("Agency / organization", "agency"),
    ("Dataset ID", "dataset_id"),
    ("Reference area / country", "reference_area"),
    ("Regional coverage", "regional_coverage"),
    ("Excluded regional values", "excluded_regional_values"),
    ("Time coverage", "time_coverage"),
    ("Frequency coverage", "frequency_coverage"),
    ("Dataset URL", "url"),
)
"""Fields rendered as the document's leading bullet list, in workbook order."""

_SECTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("Description", "description"),
    ("Indicators coverage", "indicators_coverage"),
    ("Relevant indicators not present in the dataset", "missing_indicators"),
)
"""Fields rendered as their own section, being prose or long ';'-separated lists."""

_MAX_FILENAME_STEM = 100
_FALLBACK_FILENAME_STEM = "discovery-dataset"
_FILENAME_DIGEST_CHARS = 12


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
        f"- **{label}:** {value}"
        for label, field in _SUMMARY_FIELDS
        if (value := getattr(record, field))
    )
    for label, field in _SECTION_FIELDS:
        if value := getattr(record, field):
            lines.extend(["", f"## {label}", "", value])
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
    label = escape_invalid_filename_chars(f"{record.agency} - {record.dataset_id}").strip()
    return f"{label[:_MAX_FILENAME_STEM] or _FALLBACK_FILENAME_STEM} [{digest}].md"


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


@dataclass
class PublishCounts:
    """What a publish stage did. Reported on the job row."""

    upserted: int = 0
    """Records published, whether for the first time or refreshed."""

    deleted: int = 0
    """Documents removed: withdrawn invalid records, refreshed records, and orphans."""

    skipped: int = 0
    """Records already indexed and unchanged, so left alone."""

    failed: int = 0
    """Records whose publish attempt failed. Recorded on the record as well."""


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
    ) -> None:
        self._client = client
        self._channel = channel
        self._grade = grade

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

        for record in records:
            document = by_key.pop(record_key(record.agency, record.dataset_id), None)
            try:
                await self._apply(record, document, counts)
            except Exception as e:
                _log.exception(
                    f"Failed to publish discovery record {record.id}"
                    f" (agency={record.agency!r} dataset_id={record.dataset_id!r})"
                )
                record.indexing_status = schemas.DiscoveryIndexingStatus.FAILED
                record.index_error = format_exception_reason(e)
                counts.failed += 1

        # Whatever is left claims a key no record has: a record deleted through the API, a
        # key that was corrected, or debris from a run that died mid-flight.
        await self._delete_all(list(by_key.values()) + unclaimed, counts)
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
        """Whether this document was published by this grade, for this channel.

        The channel is identified by its deployment id rather than its row id, so a document
        stays recognizable as this channel's after the channel is moved to another
        environment, where the row id would differ.
        """
        metadata = document.metadata
        return (
            metadata.get("grade") == self._grade
            and metadata.get("statgpt_channel") == self._channel
        )

    async def _apply(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        counts: PublishCounts,
    ) -> None:
        if record.validation_status is not schemas.DiscoveryValidationStatus.VALID:
            await self._withdraw(record, document, counts)
            return

        already_published = (
            record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED
            and document is not None
            and not document.is_failed
        )
        if already_published:
            counts.skipped += 1
            return

        await self._publish_record(record, document, counts)

    async def _withdraw(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        counts: PublishCounts,
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
            counts.deleted += 1

        record.indexing_status = schemas.DiscoveryIndexingStatus.NEW
        record.indexed_at = None
        record.index_error = None

    async def _publish_record(
        self,
        record: models.DiscoveryDataset,
        document: schemas.GenericRagDocument | None,
        counts: PublishCounts,
    ) -> None:
        if document is not None:
            # The channel API can replace neither content nor metadata, so a refresh is a
            # delete followed by an upload.
            await self._client.delete_document(document.id)
            counts.deleted += 1

        document_file = render_document(record, self._channel, self._grade)
        await self._client.upload_document(
            filename=document_file.filename,
            content=document_file.content,
            mime_type=document_file.mime_type,
            metadata=build_metadata(record, self._channel, self._grade),
            # The file name is derived from this record's natural key, so whatever sits under
            # it belongs to this record and nothing else is entitled to it. Refusing to
            # overwrite would let a file outliving its document - after the channel's database
            # was reset, say - block this record from ever being published again.
            overwrite=True,
        )

        record.indexing_status = schemas.DiscoveryIndexingStatus.INDEXED
        record.indexed_at = get_ts_utcnow()
        record.index_error = None
        counts.upserted += 1

    async def _delete_all(
        self, documents: Iterable[schemas.GenericRagDocument], counts: PublishCounts
    ) -> None:
        """Delete documents no record claims, one failure at a time.

        A failure here belongs to no record, so it is logged and left for the next run
        instead of failing the job: an undeleted stale document is a retrieval nuisance, not
        a reason to report that publishing did not happen.
        """
        for document in documents:
            try:
                await self._client.delete_document(document.id)
                counts.deleted += 1
            except Exception:
                _log.exception(
                    f"Failed to delete orphaned discovery document {document.id}"
                    f" ({document.display_name!r}) from channel {self._channel}"
                )
