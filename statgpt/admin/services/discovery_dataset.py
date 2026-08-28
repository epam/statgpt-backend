import asyncio
import csv
import io
import logging
import os
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from fastapi import UploadFile
from pydantic import ValidationError
from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.settings.discovery import DiscoveryUploadSettings
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import models, schemas, utils
from statgpt.common.services import (
    ChannelSerializer,
    ChannelService,
    DiscoveryDatasetService,
    GenericRagIngestionClient,
    RecordKey,
    record_key,
)

from .discovery_publisher import withdraw_documents
from .discovery_upload import COLUMN_FIELDS, FIELD_LABELS, REQUIRED_FIELDS, parse_discovery_file
from .exceptions import (
    DiscoveryDatasetConflictError,
    DiscoveryDatasetNotFoundError,
    DiscoveryPayloadError,
    DiscoveryUploadTooLargeError,
    raise_for_conflict,
)

_log = logging.getLogger(__name__)

_UPLOAD_CHUNK_SIZE = 1 << 20

_PYDANTIC_VALUE_ERROR_PREFIX = "Value error, "
"""Pydantic's prefix on a message raised by a validator, stripped for readability."""

_FIELD_BY_ALIAS: dict[str, str] = {
    (info.alias or name): name for name, info in schemas.DiscoveryDatasetBase.model_fields.items()
}
"""Alias -> field name.

The schemas carry a camelCase alias generator, and pydantic locates an error by alias, so a
violation of `dataset_id` arrives as ``datasetId``. Everything downstream - the column
labels, the cell references, the `field` a caller reads - is keyed by field name.
"""

_DESCRIPTIVE_FIELDS = frozenset(COLUMN_FIELDS)
"""The fields a write may touch. Anything else a payload carries (`id`) is not a column."""

_DiscoveryFields = schemas.DiscoveryDatasetBase | schemas.DiscoveryDatasetUpdate
"""Either shape a write arrives in: a whole record, or the subset of an edit."""


def _quote_key(key: RecordKey) -> str:
    agency, dataset_id = key
    return f"agency={agency!r}, dataset id={dataset_id!r}"


def _conflict_detail(keys: Iterable[RecordKey] = ()) -> str:
    """Build a conflict message naming the colliding record(s), so callers need not bisect."""
    named = sorted({_quote_key(key) for key in keys})
    if not named:
        return (
            "A discovery dataset with the same agency and dataset id already exists"
            " in this channel."
        )
    if len(named) == 1:
        return f"A discovery dataset with {named[0]} already exists in this channel."
    return "Discovery datasets already exist in this channel: " + "; ".join(named) + "."


def _conflict(keys: Iterable[RecordKey] = ()) -> DiscoveryDatasetConflictError:
    return DiscoveryDatasetConflictError(_conflict_detail(keys))


@dataclass(frozen=True)
class _DiscoveryCandidate:
    """A record about to be written, with enough context to report a problem against it."""

    record: schemas.DiscoveryDatasetBase

    index: int | None = None
    """0-based position in a JSON payload."""

    row: int | None = None
    """1-based row number in an uploaded file."""

    cells: dict[str, str] = field(default_factory=dict)
    """Field name -> cell reference, for a spreadsheet upload."""

    @property
    def key(self) -> RecordKey:
        return record_key(self.record.agency, self.record.dataset_id)

    @property
    def location(self) -> str:
        if self.row is not None:
            return f"row {self.row}"
        if self.index is not None:
            return f"item {self.index}"
        return "this record"

    def problem(
        self, message: str, field_name: str | None = None
    ) -> schemas.DiscoveryPayloadProblem:
        return schemas.DiscoveryPayloadProblem(
            message=message,
            field=field_name,
            index=self.index,
            row=self.row,
            cell=self.cells.get(field_name) if field_name else None,
        )


@dataclass(frozen=True)
class _RawRecord:
    """One record's values as a file supplied them, plus where they came from.

    A file's values have not been through the schema yet, so they travel with their location
    until they have: a violation has to be reported against the cell that caused it.
    """

    values: dict[str, str]

    index: int | None = None
    """0-based position in the file's records."""

    row: int | None = None
    """1-based row number in the file."""

    cells: dict[str, str] = field(default_factory=dict)
    """Field name -> cell reference, for a spreadsheet upload."""


def _build_candidates(
    data: Sequence[schemas.DiscoveryDatasetBase],
) -> list[_DiscoveryCandidate]:
    """Wrap an already-validated JSON payload so problems can be reported by position."""
    return [_DiscoveryCandidate(record=item, index=index) for index, item in enumerate(data)]


def _build_candidates_from_rows(
    raw: Sequence[_RawRecord],
) -> tuple[list[_DiscoveryCandidate], list[schemas.DiscoveryPayloadProblem]]:
    """Validate a file's records against the schema, keeping each problem at its location.

    Returns the records that passed and the problems of the ones that did not, rather than
    raising, so a caller can add its own problems and report everything in one response.
    """
    candidates: list[_DiscoveryCandidate] = []
    problems: list[schemas.DiscoveryPayloadProblem] = []

    for item in raw:
        try:
            record = schemas.DiscoveryDatasetBase.model_validate(item.values)
        except ValidationError as e:
            problems.extend(
                _problems_from_validation_error(e, index=item.index, row=item.row, cells=item.cells)
            )
            continue
        candidates.append(
            _DiscoveryCandidate(record=record, index=item.index, row=item.row, cells=item.cells)
        )

    return candidates, problems


def _duplicate_problems(
    candidates: Sequence[_DiscoveryCandidate],
) -> list[schemas.DiscoveryPayloadProblem]:
    """Report each candidate that lands on a key an earlier candidate already claimed.

    Two records that normalize onto one key cannot both be stored, and the payload cannot be
    repaired in the UI, so it is refused rather than half-applied.

    Keys are compared folded, exactly like the unique constraint; comparing raw strings
    would let ``'BI'`` and ``'bi '`` through, only to die on an IntegrityError at flush - a
    generic 409 where the point here is a 400 naming the two rows.

    Callers pass a single channel's candidates. The constraint is scoped by channel, so a
    payload spanning channels would report records that do not in fact collide - which is why
    the bulk update endpoint, whose ids are global, leaves duplicates to the database.
    """
    problems: list[schemas.DiscoveryPayloadProblem] = []
    seen: dict[RecordKey, _DiscoveryCandidate] = {}

    for candidate in candidates:
        first = seen.get(candidate.key)
        if first is None:
            seen[candidate.key] = candidate
            continue
        problems.append(
            candidate.problem(
                f"Duplicate of {first.location}: {_quote_key(candidate.key)}."
                f" Ignoring case and spacing, the two describe the same dataset.",
                "dataset_id",
            )
        )

    return problems


def _raise_for_problems(
    problems: Sequence[schemas.DiscoveryPayloadProblem], max_problems: int
) -> None:
    """Refuse a structurally unusable payload, capping how many problems are reported."""
    if problems:
        raise DiscoveryPayloadError(
            problems=list(problems[:max_problems]), truncated=len(problems) > max_problems
        )


def _problems_from_validation_error(
    error: ValidationError,
    index: int | None = None,
    row: int | None = None,
    cells: dict[str, str] | None = None,
) -> list[schemas.DiscoveryPayloadProblem]:
    """Render one record's schema violations as located payload problems.

    The natural key is constrained on the model, so an empty agency arrives here as a
    ValidationError instead of as a hand-written check. It still has to be reported against
    the cell it came from - a file whose problems surface as a raw pydantic dump is not the
    per-cell report this endpoint exists to produce.
    """
    problems: list[schemas.DiscoveryPayloadProblem] = []

    for detail in error.errors():
        located = str(detail["loc"][0]) if detail["loc"] else None
        field_name = _FIELD_BY_ALIAS.get(located, located) if located else None
        label = FIELD_LABELS.get(field_name or "")
        if detail["type"] == "missing":
            message = "is missing"
        else:
            message = str(detail["msg"]).removeprefix(_PYDANTIC_VALUE_ERROR_PREFIX)
        problems.append(
            schemas.DiscoveryPayloadProblem(
                message=f"{label} {message}." if label else f"{message}.",
                field=field_name,
                index=index,
                row=row,
                cell=(cells or {}).get(field_name) if field_name else None,
            )
        )

    return problems


class AdminPortalDiscoveryDatasetService(DiscoveryDatasetService):
    """Write access to the discovery dataset records of a channel.

    Also the layer that turns an unknown id into a domain error: the read service in `common`
    reports an absent record by returning `None`.
    """

    _UPLOAD_SETTINGS = DiscoveryUploadSettings()

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def _new_model(
        channel_id: int, record: schemas.DiscoveryDatasetBase
    ) -> models.DiscoveryDataset:
        return models.DiscoveryDataset(
            channel_id=channel_id,
            validation_status=schemas.DiscoveryValidationStatus.NOT_VALIDATED,
            indexing_status=schemas.DiscoveryIndexingStatus.NEW,
            **record.model_dump(),
        )

    @staticmethod
    def _reset_derived_state(item: models.DiscoveryDataset) -> None:
        """Forget verdicts that describe the record as it was before this edit.

        `validation_status` is reset rather than recomputed: recomputing would run the
        semantic check set on the write path, which is exactly what it must not do.
        A record that was published now has a stale document in the index, which is what
        `OUTDATED` exists to make visible.

        A previous publish failure described the old content, so it is cleared and the record
        goes back to awaiting its first attempt. That keeps the invariant the UI relies on:
        `index_error` is set exactly while `indexing_status` is FAILED.
        """
        item.validation_status = schemas.DiscoveryValidationStatus.NOT_VALIDATED
        item.validation_issues = None
        item.validated_at = None
        if item.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED:
            item.indexing_status = schemas.DiscoveryIndexingStatus.OUTDATED
        elif item.indexing_status is schemas.DiscoveryIndexingStatus.FAILED:
            item.indexing_status = schemas.DiscoveryIndexingStatus.NEW
            item.index_error = None
        item.updated_at = func.now()  # type: ignore[assignment]

    @classmethod
    def _apply_values(
        cls,
        item: models.DiscoveryDataset,
        data: _DiscoveryFields,
        ignore_key_case: bool,
    ) -> bool:
        """Write the changed descriptive fields of `data` onto a stored record.

        Only the fields the payload actually carries are written. `exclude_unset` is what
        draws that line: a file may legitimately hold a subset of the columns - an agency
        correcting one of them - and the fields it does not mention must survive. Comparing
        against the defaults instead would blank them, and `exclude_defaults` cannot tell an
        absent column from a cell the submitter deliberately cleared. On an edit, `None`
        means "not provided" too, so it is dropped rather than written.

        With `ignore_key_case`, a key field differing only in case is left as it is: a
        workbook resubmitting 'bi' against a stored 'BI' describes the same dataset, and
        rewriting it would mark the record outdated and republish an unchanged document.
        A deliberate single-record edit is a correction, so it does rewrite the casing.
        """
        changed = False
        for name, new_value in data.model_dump(exclude_unset=True).items():
            if name not in _DESCRIPTIVE_FIELDS or new_value is None:
                continue
            current = getattr(item, name)
            if ignore_key_case and name in REQUIRED_FIELDS and current.lower() == new_value.lower():
                continue
            if current != new_value:
                setattr(item, name, new_value)
                changed = True

        if changed:
            cls._reset_derived_state(item)
        return changed

    async def _get_item_or_raise(self, item_id: int) -> models.DiscoveryDataset:
        item = await self.get_record_model_by_id(item_id)
        if item is None:
            raise DiscoveryDatasetNotFoundError(item_id)
        return item

    async def _existing_by_key(self, channel_id: int) -> dict[RecordKey, models.DiscoveryDataset]:
        records = await self.get_record_models_by_channel(channel_id, limit=None, offset=0)
        return {record_key(item.agency, item.dataset_id): item for item in records}

    async def _raise_for_conflicting_records(
        self, channel_id: int, candidates: Sequence[_DiscoveryCandidate]
    ) -> None:
        """Reject an insert up-front, naming every key that would collide."""
        keys = {candidate.key for candidate in candidates}
        if not keys:
            return

        query = select(
            models.DiscoveryDataset.agency_key, models.DiscoveryDataset.dataset_id_key
        ).where(models.DiscoveryDataset.channel_id == channel_id)
        async with self._lock_session() as session:
            stored = (await session.execute(query)).all()

        conflicting = [
            (agency, dataset_id) for agency, dataset_id in stored if (agency, dataset_id) in keys
        ]
        if conflicting:
            raise _conflict(conflicting)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ single record ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def get_record_schema_by_id(self, item_id: int) -> schemas.DiscoveryDataset:
        return self._serialize(await self._get_item_or_raise(item_id))

    async def add_record(
        self, channel_id: int, data: schemas.DiscoveryDatasetBase
    ) -> schemas.DiscoveryDataset:
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        item = self._new_model(channel.id, data)
        self._session.add(item)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_conflict(e, _conflict([record_key(data.agency, data.dataset_id)]))

        return self._serialize(item)

    async def update(
        self, item_id: int, data: schemas.DiscoveryDatasetUpdate
    ) -> schemas.DiscoveryDataset:
        item = await self._get_item_or_raise(item_id)

        if self._apply_values(item, data, ignore_key_case=False):
            # Read before the commit: a rollback expires the instance, and re-reading an
            # expired attribute in an async session raises instead of lazy-loading.
            key = record_key(item.agency, item.dataset_id)
            try:
                await self._session.commit()
            except IntegrityError as e:
                await self._session.rollback()
                raise_for_conflict(e, _conflict([key]))
            await self._session.refresh(item)

        return self._serialize(item)

    async def delete(self, item_id: int) -> None:
        item = await self._get_item_or_raise(item_id)
        _log.info(f"Deleting {item}")

        await self._withdraw_documents([item])
        await self._session.delete(item)
        await self._session.commit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bulk ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def add_records_bulk(
        self, channel_id: int, data: Sequence[schemas.DiscoveryDatasetBase]
    ) -> list[schemas.DiscoveryDataset]:
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        candidates = _build_candidates(data)
        _raise_for_problems(
            _duplicate_problems(candidates), self._UPLOAD_SETTINGS.max_reported_problems
        )
        await self._raise_for_conflicting_records(channel.id, candidates)

        items = [self._new_model(channel.id, candidate.record) for candidate in candidates]
        self._session.add_all(items)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            # The pre-check above names the conflict in the common case; getting here means
            # a concurrent writer inserted the same key in the meantime.
            raise_for_conflict(e, _conflict())

        return [self._serialize(item) for item in items]

    async def update_records_bulk(
        self, data: Sequence[schemas.DiscoveryDatasetUpdateBulk]
    ) -> list[schemas.DiscoveryDataset]:
        existing = {
            item.id: item for item in await self.get_record_models_by_ids([d.id for d in data])
        }

        updated_ids: list[int] = []
        for update_request in data:
            item = existing.get(update_request.id)
            if item is None:
                raise DiscoveryDatasetNotFoundError(update_request.id)
            if self._apply_values(item, update_request, ignore_key_case=False):
                updated_ids.append(update_request.id)

        if updated_ids:
            _log.info(f"Updating {len(updated_ids)} of {len(data)} discovery datasets.")
            try:
                await self._session.commit()
            except IntegrityError as e:
                await self._session.rollback()
                raise_for_conflict(e, _conflict())
            refreshed = {item.id: item for item in await self.get_record_models_by_ids(updated_ids)}
            existing.update(refreshed)
        else:
            _log.info(f"All {len(data)} discovery datasets are up-to-date.")

        return [self._serialize(existing[item.id]) for item in data]

    async def delete_records_bulk(
        self, item_ids: list[int] | None = None, channel_id: int | None = None
    ) -> list[schemas.DiscoveryDataset]:
        if item_ids is not None and channel_id is not None:
            raise RuntimeError("Only one of item_ids or channel_id must be provided.")

        where_clause: ColumnElement[bool]
        if item_ids is not None:
            where_clause = models.DiscoveryDataset.id.in_(item_ids)
        elif channel_id is not None:
            where_clause = models.DiscoveryDataset.channel_id == channel_id
        else:
            raise RuntimeError("Either item_ids or channel_id must be provided.")

        # Read the rows before deleting them rather than using `DELETE ... RETURNING`: their
        # documents have to be withdrawn while the records that name them still exist, and the
        # response has to be built while the instances are still readable.
        query = select(models.DiscoveryDataset).where(where_clause)
        doomed = list((await self._session.execute(query)).scalars().all())
        if not doomed:
            return []

        # Serialized up front: the commit below expires the instances, and reading an expired
        # attribute in an async session raises instead of lazy-loading.
        deleted = [self._serialize(item) for item in doomed]

        await self._withdraw_documents(doomed)
        await self._session.execute(delete(models.DiscoveryDataset).where(where_clause))
        await self._session.commit()
        _log.info(f"Deleted {len(deleted)} discovery datasets.")
        return deleted

    async def _withdraw_documents(self, records: Sequence[models.DiscoveryDataset]) -> None:
        """Remove the RAG documents of records that are about to be deleted.

        The documents go first and the rows second, for the reason `DiscoveryPublisher._withdraw`
        gives: a crash in between leaves a record with no document behind it, which the next
        indexing run repairs, while the opposite order strands a live document that nothing
        points at any more and only an orphan sweep would ever find.

        A failure is not swallowed. Deleting the record while its document survives is exactly
        what this exists to prevent, so the caller is left to fail with the rows intact.
        """
        by_channel: dict[int, list[models.DiscoveryDataset]] = defaultdict(list)
        for record in records:
            by_channel[record.channel_id].append(record)

        channels = ChannelService(self._session)
        for channel_id, group in by_channel.items():
            channel = await channels.get_model_by_id(channel_id)
            application_id = ChannelSerializer.db_to_schema(
                channel
            ).details.discovery_application_id
            if application_id is None:
                # No publish target, so nothing was ever published. Refusing the delete here
                # would block it over a document that cannot exist.
                continue

            keys = {record_key(record.agency, record.dataset_id) for record in group}
            async with GenericRagIngestionClient.for_application(application_id) as client:
                withdrawn = await withdraw_documents(client, channel.deployment_id, keys)
            _log.info(
                f"Withdrew {withdrawn} discovery document(s) of channel"
                f" {channel.deployment_id} for {len(group)} record(s) being deleted."
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ upsert ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _upsert(
        self,
        channel_id: int,
        candidates: Sequence[_DiscoveryCandidate],
        delete_absent: bool,
    ) -> schemas.DiscoveryUploadSummary:
        """Reconcile a payload with what the channel already holds.

        Unchanged records are not written at all, so an unedited resubmission keeps its
        statuses instead of being reported as deleted-and-recreated.
        """
        existing = await self._existing_by_key(channel_id)
        summary = schemas.DiscoveryUploadSummary()

        for candidate in candidates:
            item = existing.get(candidate.key)
            if item is None:
                self._session.add(self._new_model(channel_id, candidate.record))
                summary.created += 1
            elif self._apply_values(item, candidate.record, ignore_key_case=True):
                summary.updated += 1
            else:
                summary.unchanged += 1

        if delete_absent:
            submitted = {candidate.key for candidate in candidates}
            stale_ids = [item.id for key, item in existing.items() if key not in submitted]
            if stale_ids:
                # One statement rather than one per record: a full replacement can strand
                # thousands of rows. Safe to run before the inserts above are flushed, since
                # a submitted key is never in this set.
                await self._session.execute(
                    delete(models.DiscoveryDataset).where(models.DiscoveryDataset.id.in_(stale_ids))
                )
                summary.deleted = len(stale_ids)

        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_conflict(e, _conflict())

        # The row counts belong to parsing and are filled in by the caller, so they are
        # left out here rather than logged as zero.
        _log.info(
            f"Upserted discovery datasets for channel {channel_id}: "
            f"{summary.model_dump(include={'created', 'updated', 'unchanged', 'deleted'})}"
        )
        return summary

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ file upload ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def read_upload(self, file: UploadFile) -> bytes:
        """Read an upload into memory, refusing anything over the configured cap.

        Read in chunks so an oversized file is refused without a second full copy of it in
        memory. Note that this is not a limit on what crosses the wire: by the time a handler
        runs, Starlette's multipart parser has already received the whole body and spooled it
        to a temporary file. Bounding the transfer itself belongs at the ingress.
        """
        limit = self._UPLOAD_SETTINGS.max_file_size_bytes
        buffer = bytearray()
        while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise DiscoveryUploadTooLargeError(
                    f"The file is larger than the {limit // (1024 * 1024)} MB limit."
                )
        return bytes(buffer)

    async def upload(
        self,
        channel_id: int,
        data: bytes,
        filename: str | None,
        mode: schemas.DiscoveryUploadMode,
    ) -> schemas.DiscoveryUploadSummary:
        """Load a filled discovery workbook or CSV into a channel."""
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        parsed = await asyncio.to_thread(
            parse_discovery_file, data, filename, self._UPLOAD_SETTINGS.max_rows
        )
        raw = [
            _RawRecord(
                values=row.values,
                row=row.row_number,
                cells={
                    name: reference
                    for name in COLUMN_FIELDS
                    if (reference := parsed.cell(name, row.row_number))
                },
            )
            for row in parsed.rows
        ]
        candidates, problems = _build_candidates_from_rows(raw)
        problems.extend(_duplicate_problems(candidates))
        _raise_for_problems(problems, self._UPLOAD_SETTINGS.max_reported_problems)

        summary = await self._upsert(
            channel.id,
            candidates,
            delete_absent=mode is schemas.DiscoveryUploadMode.REPLACE,
        )
        summary.rows_read = len(parsed.rows)
        summary.rows_skipped = parsed.rows_skipped
        return summary

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ export / import ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def export_discovery_datasets_to_folder(
        self, channel: models.Channel, folder_path: str
    ) -> None:
        records = await self.get_record_models_by_channel(channel.id, limit=None, offset=0)

        if not records:
            _log.info("No discovery datasets found.")
            return

        _log.info(f"Exporting {len(records)} discovery datasets.")
        # Only the descriptive fields travel: validation and indexing state describe this
        # deployment's index, not the dataset.
        rows = [
            schemas.DiscoveryDatasetBase.model_validate(item, from_attributes=True).model_dump(
                mode="json"
            )
            for item in records
        ]

        file_path = os.path.join(folder_path, JobsConfig.DISCOVERY_DATASETS_FILE)
        utils.write_csv_from_dict_list(rows, file_path)
        _log.info(f"Exported discovery datasets to {file_path!r}.")

    async def import_discovery_datasets_from_zip(
        self, zip_file: zipfile.ZipFile, channel_id: int
    ) -> None:
        """Load the archive's records into a channel.

        Always reconciles on the natural key, so this covers both a fresh channel and a
        merge into one that already holds records: re-importing the same archive is
        idempotent instead of colliding on the unique constraint.
        """
        if JobsConfig.DISCOVERY_DATASETS_FILE not in zip_file.namelist():
            _log.info("No discovery datasets found in the zip file.")
            return

        _log.info("Importing discovery datasets from zip file.")
        with zip_file.open(JobsConfig.DISCOVERY_DATASETS_FILE) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding="utf-8", newline=""))
            raw = [_RawRecord(values=row, index=index) for index, row in enumerate(reader)]

        # Validated row by row rather than in a comprehension: an archive whose CSV was edited
        # by hand still has to say which row is at fault, in the failure reason of the import
        # job. A raw pydantic dump there is not something an operator can act on.
        candidates, problems = _build_candidates_from_rows(raw)
        problems.extend(_duplicate_problems(candidates))
        _raise_for_problems(problems, self._UPLOAD_SETTINGS.max_reported_problems)

        summary = await self._upsert(channel_id, candidates, delete_absent=False)
        _log.info(f"Imported {summary.created + summary.updated} discovery datasets.")
