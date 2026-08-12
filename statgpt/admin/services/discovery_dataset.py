import asyncio
import csv
import io
import logging
import os
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.settings.discovery import DiscoveryUploadSettings
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import models, schemas, utils
from statgpt.common.schemas import DiscoveryIndexingStatus, DiscoveryValidationStatus
from statgpt.common.services import ChannelService, DiscoveryDatasetService

from .discovery_upload import COLUMN_FIELDS, FIELD_LABELS, REQUIRED_FIELDS, parse_discovery_file
from .exceptions import (
    DiscoveryPayloadError,
    DiscoveryUploadTooLargeError,
    raise_for_integrity_error,
)

_log = logging.getLogger(__name__)

_UPLOAD_CHUNK_SIZE = 1 << 20

RecordKey = tuple[str, str]
"""The natural key of a record within its channel: lowercased (agency, dataset_id)."""


def record_key(agency: str, dataset_id: str) -> RecordKey:
    """Build the key the database compares on.

    ``.lower()``, never ``.casefold()``: the generated `agency_key` / `dataset_id_key`
    columns hold what PostgreSQL's ``lower()`` produced, and ``casefold()`` is more
    aggressive (``'ß'`` -> ``'ss'``), so it would miss rows the database considers matches.
    """
    return agency.lower(), dataset_id.lower()


def _quote_key(key: RecordKey) -> str:
    agency, dataset_id = key
    return f"agency={agency!r}, dataset id={dataset_id!r}"


def _conflict_detail(keys: Iterable[RecordKey] = ()) -> str:
    """Build a 409 detail naming the colliding record(s), so callers need not bisect."""
    named = sorted({_quote_key(key) for key in keys})
    if not named:
        return "A discovery dataset with the same agency and dataset id already exists in this channel."
    if len(named) == 1:
        return f"A discovery dataset with {named[0]} already exists in this channel."
    return "Discovery datasets already exist in this channel: " + "; ".join(named) + "."


@dataclass(frozen=True)
class DiscoveryCandidate:
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


def build_candidates(
    data: Sequence[schemas.DiscoveryDatasetBase],
) -> list[DiscoveryCandidate]:
    """Wrap a JSON payload so problems can be reported by position."""
    return [DiscoveryCandidate(record=item, index=index) for index, item in enumerate(data)]


def validate_candidates(candidates: Sequence[DiscoveryCandidate], max_problems: int) -> None:
    """Reject a payload that cannot be stored or reconciled, naming every offending record.

    Structural checks only - a record with no natural key cannot be matched, and two records
    that normalize to one key cannot both be stored. These cannot be repaired in the UI, so
    persisting them buys nothing.

    Both duplicate checks compare lowercased keys, exactly like the unique constraint;
    comparing raw strings would let ``'BI'`` and ``'bi '`` through, only to die on an
    IntegrityError at flush - a 500 where the whole point here is a 400 naming the two rows.
    """
    problems: list[schemas.DiscoveryPayloadProblem] = []
    seen: dict[RecordKey, DiscoveryCandidate] = {}

    for candidate in candidates:
        incomplete = False
        for name in REQUIRED_FIELDS:
            if not getattr(candidate.record, name):
                problems.append(candidate.problem(f"{FIELD_LABELS[name]} is empty.", name))
                incomplete = True
        if incomplete:
            continue

        first = seen.get(candidate.key)
        if first is not None:
            problems.append(
                candidate.problem(
                    f"Duplicate of {first.location}: {_quote_key(candidate.key)}."
                    f" Ignoring case and spacing, the two describe the same dataset.",
                    "dataset_id",
                )
            )
        else:
            seen[candidate.key] = candidate

    if problems:
        raise DiscoveryPayloadError(
            problems=problems[:max_problems], truncated=len(problems) > max_problems
        )


class AdminPortalDiscoveryDatasetService(DiscoveryDatasetService):
    """Write access to the discovery dataset records of a channel."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal
        self._settings = DiscoveryUploadSettings()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def _new_model(
        channel_id: int, record: schemas.DiscoveryDatasetBase
    ) -> models.DiscoveryDataset:
        return models.DiscoveryDataset(
            channel_id=channel_id,
            validation_status=DiscoveryValidationStatus.NOT_VALIDATED,
            indexing_status=DiscoveryIndexingStatus.NEW,
            **record.model_dump(),
        )

    @staticmethod
    def _reset_derived_state(item: models.DiscoveryDataset) -> None:
        """Forget verdicts that describe the record as it was before this edit.

        `validation_status` is reset rather than recomputed: recomputing would run the
        semantic check set on the write path, which is exactly what it must not do.
        A record that was published now has a stale document in the index, which is what
        `OUTDATED` exists to make visible.
        """
        item.validation_status = DiscoveryValidationStatus.NOT_VALIDATED
        item.validation_issues = None
        item.validated_at = None
        if item.indexing_status is DiscoveryIndexingStatus.INDEXED:
            item.indexing_status = DiscoveryIndexingStatus.OUTDATED
        item.updated_at = func.now()  # type: ignore[assignment]

    @classmethod
    def _apply_values(
        cls,
        item: models.DiscoveryDataset,
        values: dict[str, str],
        ignore_key_case: bool,
    ) -> bool:
        """Write the changed descriptive fields onto a stored record.

        With `ignore_key_case`, a key field differing only in case is left as it is: a
        workbook resubmitting 'bi' against a stored 'BI' describes the same dataset, and
        rewriting it would mark the record outdated and republish an unchanged document.
        A deliberate single-record edit is a correction, so it does rewrite the casing.
        """
        changed = False
        for name, new_value in values.items():
            current = getattr(item, name)
            if ignore_key_case and name in REQUIRED_FIELDS and current.lower() == new_value.lower():
                continue
            if current != new_value:
                setattr(item, name, new_value)
                changed = True

        if changed:
            cls._reset_derived_state(item)
        return changed

    async def _existing_by_key(self, channel_id: int) -> dict[RecordKey, models.DiscoveryDataset]:
        records = await self.get_record_models_by_channel(channel_id, limit=None, offset=0)
        return {record_key(item.agency, item.dataset_id): item for item in records}

    async def _raise_for_conflicting_records(
        self, channel_id: int, candidates: Sequence[DiscoveryCandidate]
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
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=_conflict_detail(conflicting)
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ single record ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def add_record(
        self, channel_id: int, data: schemas.DiscoveryDatasetBase
    ) -> schemas.DiscoveryDataset:
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        candidates = build_candidates([data])
        validate_candidates(candidates, self._settings.max_reported_problems)

        item = self._new_model(channel.id, data)
        self._session.add(item)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_integrity_error(e, _conflict_detail([candidates[0].key]))

        return self.serialize(item)

    async def update(
        self, item_id: int, data: schemas.DiscoveryDatasetUpdate
    ) -> schemas.DiscoveryDataset:
        item = await self._get_item_or_raise(item_id)
        values: dict[str, str] = data.model_dump(exclude_unset=True, exclude_none=True)

        candidate = DiscoveryCandidate(
            record=schemas.DiscoveryDatasetBase.model_validate(
                {**{name: getattr(item, name) for name in COLUMN_FIELDS}, **values}
            )
        )
        validate_candidates([candidate], self._settings.max_reported_problems)

        if self._apply_values(item, values, ignore_key_case=False):
            try:
                await self._session.commit()
            except IntegrityError as e:
                await self._session.rollback()
                raise_for_integrity_error(e, _conflict_detail([candidate.key]))
            await self._session.refresh(item)

        return self.serialize(item)

    async def delete(self, item_id: int) -> None:
        item = await self._get_item_or_raise(item_id)
        _log.info(f"Deleting {item}")

        await self._session.delete(item)
        await self._session.commit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bulk ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def add_records_bulk(
        self, channel_id: int, data: Sequence[schemas.DiscoveryDatasetBase]
    ) -> list[schemas.DiscoveryDataset]:
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        candidates = build_candidates(data)
        validate_candidates(candidates, self._settings.max_reported_problems)
        await self._raise_for_conflicting_records(channel.id, candidates)

        items = [self._new_model(channel.id, candidate.record) for candidate in candidates]
        self._session.add_all(items)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            # The pre-check above names the conflict in the common case; getting here means
            # a concurrent writer inserted the same key in the meantime.
            raise_for_integrity_error(e, _conflict_detail())

        return [self.serialize(item) for item in items]

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
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"DiscoveryDataset with id {update_request.id} not found.",
                )
            values: dict[str, str] = update_request.model_dump(
                exclude={"id"}, exclude_unset=True, exclude_none=True
            )
            if self._apply_values(item, values, ignore_key_case=False):
                updated_ids.append(update_request.id)

        if updated_ids:
            _log.info(f"Updating {len(updated_ids)} of {len(data)} discovery datasets.")
            try:
                await self._session.commit()
            except IntegrityError as e:
                await self._session.rollback()
                raise_for_integrity_error(e, _conflict_detail())
            refreshed = {item.id: item for item in await self.get_record_models_by_ids(updated_ids)}
            existing.update(refreshed)
        else:
            _log.info(f"All {len(data)} discovery datasets are up-to-date.")

        return [self.serialize(existing[item.id]) for item in data]

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

        query = (
            delete(models.DiscoveryDataset).where(where_clause).returning(models.DiscoveryDataset)
        )
        deleted = (await self._session.execute(query)).scalars().all()
        await self._session.commit()
        _log.info(f"Deleted {len(deleted)} discovery datasets.")
        return [self.serialize(item) for item in deleted]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ upsert ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _upsert(
        self,
        channel_id: int,
        candidates: Sequence[DiscoveryCandidate],
        delete_absent: bool,
    ) -> schemas.DiscoveryUploadSummary:
        """Reconcile a payload with what the channel already holds.

        Unchanged records are not written at all, so an unedited resubmission keeps its
        statuses instead of being reported as deleted-and-recreated.
        """
        existing = await self._existing_by_key(channel_id)
        summary = schemas.DiscoveryUploadSummary()

        for candidate in candidates:
            values = candidate.record.model_dump()
            item = existing.get(candidate.key)
            if item is None:
                self._session.add(self._new_model(channel_id, candidate.record))
                summary.created += 1
            elif self._apply_values(item, values, ignore_key_case=True):
                summary.updated += 1
            else:
                summary.unchanged += 1

        if delete_absent:
            submitted = {candidate.key for candidate in candidates}
            for key, item in existing.items():
                if key not in submitted:
                    await self._session.delete(item)
                    summary.deleted += 1

        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_integrity_error(e, _conflict_detail())

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

        Streamed rather than read whole so an oversized file is rejected without first
        being buffered in full.
        """
        limit = self._settings.max_file_size_bytes
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
            parse_discovery_file, data, filename, self._settings.max_rows
        )
        candidates = [
            DiscoveryCandidate(
                record=schemas.DiscoveryDatasetBase.model_validate(row.values),
                row=row.row_number,
                cells=(
                    {name: parsed.cell(name, row.row_number) or "" for name in COLUMN_FIELDS}
                    if parsed.column_letters
                    else {}
                ),
            )
            for row in parsed.rows
        ]
        validate_candidates(candidates, self._settings.max_reported_problems)

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
            reader = csv.DictReader(io.TextIOWrapper(file, encoding='utf-8', newline=''))
            records = [schemas.DiscoveryDatasetBase.model_validate(row) for row in reader]

        candidates = build_candidates(records)
        validate_candidates(candidates, self._settings.max_reported_problems)

        summary = await self._upsert(channel_id, candidates, delete_absent=False)
        _log.info(f"Imported {summary.created + summary.updated} discovery datasets.")
