import asyncio
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from aidial_client import AsyncDial, DialException
from aidial_client.types.metadata import FileItem, FileMetadata
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services import prepared_downloads as service_module
from statgpt.admin.services.prepared_downloads import (
    PREPARED_DOWNLOAD_PREFIX,
    PreparedDownloadService,
)
from statgpt.common import models
from statgpt.common.utils.dial.attachments import AttachmentsStorage

_NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)


class _Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return _NOW


def _record(number: int, expires_at: datetime = _NOW) -> models.PreparedDownload:
    return models.PreparedDownload(
        id=UUID(int=number),
        owner_user_id="owner",
        channel_id=1,
        file_path=f"files/bucket/{PREPARED_DOWNLOAD_PREFIX}/{UUID(int=number)}.csv",
        filename="discovery_datasets.csv",
        media_type="text/csv",
        expires_at=expires_at,
    )


def _file(
    number: int,
    *,
    age: timedelta | None = timedelta(hours=25),
    extension: str = "csv",
    folder: str = PREPARED_DOWNLOAD_PREFIX,
    node_type: Literal["FOLDER", "ITEM"] = "ITEM",
) -> FileItem:
    return FileItem(
        bucket="bucket",
        url=f"files/bucket/{folder}/{UUID(int=number)}.{extension}",
        node_type=node_type,
        resource_type="FILE",
        updated_at=None if age is None else int((_NOW - age).timestamp() * 1000),
    )


class _Session:
    """Transactional row store which evaluates the cleanup's bound keyset parameters."""

    def __init__(self, rows: dict[UUID, models.PreparedDownload], events: list[str]) -> None:
        self.rows = rows
        self.events = events
        self.in_transaction = False
        self.delete_failures: set[UUID] = set()
        self.commit_failures: set[UUID] = set()
        self.reference_failures: set[str] = set()
        self.batches: list[list[UUID]] = []
        self.queries: list[Any] = []
        self.pending_delete: UUID | None = None

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[None]:
        assert not self.in_transaction
        self.in_transaction = True
        self.pending_delete = None
        try:
            yield
            if self.pending_delete in self.commit_failures:
                raise RuntimeError("row commit failed")
            if self.pending_delete is not None:
                self.rows.pop(self.pending_delete, None)
                self.events.append(f"commit:{self.pending_delete.int}")
        except BaseException:
            self.events.append("rollback")
            raise
        finally:
            self.pending_delete = None
            self.in_transaction = False

    async def execute(self, statement: Any) -> MagicMock:
        assert self.in_transaction
        self.queries.append(statement)
        parameters = statement.compile().params
        result = MagicMock()
        if statement.is_select:
            assert list(statement.selected_columns.keys()) == ["expires_at", "id", "file_path"]
            assert statement._offset_clause is None
            assert statement._for_update_arg is None
            assert str(statement).endswith("LIMIT :param_1")
            cutoff = parameters["expires_at_1"]
            cursor = (
                (parameters["expires_at_2"], parameters["id_1"])
                if "expires_at_2" in parameters
                else None
            )
            eligible = sorted(
                (
                    (r.expires_at, r.id, r.file_path)
                    for r in self.rows.values()
                    if r.expires_at <= cutoff and (cursor is None or (r.expires_at, r.id) > cursor)
                ),
                key=lambda row: row[:2],
            )[: parameters["param_1"]]
            self.batches.append([row[1] for row in eligible])
            result.all.return_value = eligible
        else:
            download_id = parameters["id_1"]
            self.events.append(f"row:{download_id.int}")
            if download_id in self.delete_failures:
                raise RuntimeError("row deletion failed")
            row = self.rows.get(download_id)
            if (
                row is not None
                and row.file_path == parameters["file_path_1"]
                and row.expires_at <= parameters["expires_at_1"]
            ):
                self.pending_delete = download_id
        return result

    async def scalar(self, statement: Any) -> UUID | None:
        assert self.in_transaction
        path = statement.compile().params["file_path_1"]
        self.events.append(f"reference:{path}")
        if path in self.reference_failures:
            raise RuntimeError("reference lookup failed")
        return next((r.id for r in self.rows.values() if r.file_path == path), None)


@pytest.fixture(autouse=True)
def _clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "datetime", _Clock)


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def session(events: list[str]) -> _Session:
    return _Session({}, events)


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch, session: _Session, events: list[str]) -> MagicMock:
    storage = MagicMock(spec=AttachmentsStorage)
    existing: set[str] = set()

    async def delete(path: str) -> None:
        assert not session.in_transaction
        events.append(f"file:{path}")
        if path not in existing:
            raise DialException("missing", status_code=404)
        existing.remove(path)

    async def files(*_: Any, **__: Any) -> AsyncIterator[FileItem]:
        assert not session.in_transaction
        if False:
            yield _file(0)

    storage.delete_file = AsyncMock(side_effect=delete)
    storage.iter_files_in_folder.side_effect = files
    storage.existing = existing

    @asynccontextmanager
    async def factory(api_key: str) -> AsyncIterator[AttachmentsStorage]:
        assert api_key == SystemUserAuthContext().api_key
        yield storage

    monkeypatch.setattr(service_module, "attachments_storage_factory", factory)
    return storage


def _service(session: _Session) -> PreparedDownloadService:
    return PreparedDownloadService(cast(AsyncSession, session))


def _add(session: _Session, storage: MagicMock, *records: models.PreparedDownload) -> None:
    for record in records:
        session.rows[record.id] = record
        storage.existing.add(record.file_path)


async def test_expired_files_precede_row_commits_and_active_rows_are_preserved(
    session: _Session, storage: MagicMock, events: list[str]
) -> None:
    expired = _record(1, _NOW - timedelta(hours=1))
    boundary = _record(2)
    active = _record(3, _NOW + timedelta(seconds=1))
    _add(session, storage, expired, boundary, active)

    await _service(session).cleanup(SystemUserAuthContext(), batch_size=1)

    assert session.rows == {active.id: active}
    assert storage.existing == {active.file_path}
    assert events == [
        f"file:{expired.file_path}",
        "row:1",
        "commit:1",
        f"file:{boundary.file_path}",
        "row:2",
        "commit:2",
    ]
    assert session.batches == [[expired.id], [boundary.id], []]


async def test_missing_file_counts_as_removed(session: _Session, storage: MagicMock) -> None:
    record = _record(1)
    session.rows[record.id] = record

    await _service(session).cleanup(SystemUserAuthContext())

    assert not session.rows
    storage.delete_file.assert_awaited_once_with(record.file_path)


async def test_failed_file_does_not_starve_same_or_later_batches_and_retries(
    session: _Session, storage: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    records = [_record(i) for i in range(1, 5)]
    _add(session, storage, *records)
    original_delete = storage.delete_file.side_effect

    async def delete(path: str) -> None:
        if path == records[0].file_path:
            raise DialException("unavailable", status_code=503)
        await original_delete(path)

    storage.delete_file.side_effect = delete
    await _service(session).cleanup(SystemUserAuthContext(), batch_size=2)

    assert list(session.rows) == [records[0].id]
    assert session.batches == [[records[0].id, records[1].id], [records[2].id, records[3].id], []]
    assert "Failed to clean up prepared download" in caplog.text
    assert "row:1" not in session.events

    storage.delete_file.side_effect = original_delete
    await _service(session).cleanup(SystemUserAuthContext(), batch_size=2)
    assert not session.rows
    assert not storage.existing


@pytest.mark.parametrize("failure", ["delete_failures", "commit_failures"])
async def test_row_failure_retains_metadata_and_retries_with_missing_file(
    failure: str, session: _Session, storage: MagicMock
) -> None:
    first, second = _record(1), _record(2)
    _add(session, storage, first, second)
    failures = getattr(session, failure)
    failures.add(first.id)

    await _service(session).cleanup(SystemUserAuthContext(), batch_size=1)

    assert session.rows == {first.id: first}
    assert not storage.existing
    assert "rollback" in session.events
    assert session.batches == [[first.id], [second.id], []]

    failures.clear()
    await _service(session).cleanup(SystemUserAuthContext())
    assert not session.rows
    assert storage.delete_file.await_args_list.count(((first.file_path,), {})) == 2


async def test_overlapping_runs_are_idempotent(
    session: _Session, storage: MagicMock, events: list[str]
) -> None:
    record = _record(1)
    _add(session, storage, record)
    other_session = _Session(session.rows, events)
    both_selected = asyncio.Event()
    selected = 0
    original_delete = storage.delete_file.side_effect

    async def delete(path: str) -> None:
        nonlocal selected
        assert not session.in_transaction
        assert not other_session.in_transaction
        selected += 1
        if selected == 2:
            both_selected.set()
        await both_selected.wait()
        await original_delete(path)

    storage.delete_file.side_effect = delete
    await asyncio.wait_for(
        asyncio.gather(
            _service(session).cleanup(SystemUserAuthContext()),
            _service(other_session).cleanup(SystemUserAuthContext()),
        ),
        timeout=2,
    )

    assert selected == 2
    assert not session.rows
    assert not storage.existing
    await _service(session).cleanup(SystemUserAuthContext())
    assert selected == 2


async def test_orphan_age_pattern_namespace_and_current_reference_guards(
    session: _Session, storage: MagicMock, events: list[str]
) -> None:
    old_csv, old_xlsx = _file(1), _file(2, extension="xlsx")
    recent = _file(3, age=timedelta(hours=23))
    boundary = _file(4, age=timedelta(hours=24))
    unknown = _file(5, age=None)
    referenced = _file(6)
    unrelated = _file(7, folder="exports")
    nested = _file(8, folder=f"{PREPARED_DOWNLOAD_PREFIX}/nested")
    wrong_format = _file(9, extension="zip")
    folder = _file(10, node_type="FOLDER")
    wrong_name = _file(11)
    wrong_name.url = f"files/bucket/{PREPARED_DOWNLOAD_PREFIX}/manual.csv"
    modified = _file(12)
    modified.created_at = modified.updated_at
    modified.updated_at = int(_NOW.timestamp() * 1000)
    creation_only = _file(13, age=None)
    creation_only.created_at = old_csv.updated_at
    files = [
        old_csv,
        old_xlsx,
        recent,
        boundary,
        unknown,
        referenced,
        unrelated,
        nested,
        wrong_format,
        folder,
        wrong_name,
        modified,
        creation_only,
    ]
    storage.existing.update(file.url for file in files)

    async def listing(*_: Any, **__: Any) -> AsyncIterator[FileItem]:
        for file in files:
            if file is referenced:
                record = _record(6, _NOW + timedelta(hours=1))
                session.rows[record.id] = record
            yield file

    storage.iter_files_in_folder.side_effect = listing
    await _service(session).cleanup(SystemUserAuthContext(), batch_size=2)

    deleted = {call.args[0] for call in storage.delete_file.await_args_list}
    assert deleted == {old_csv.url, old_xlsx.url, creation_only.url}
    assert referenced.url in storage.existing
    assert session.rows[_record(6).id].file_path == referenced.url
    for path in deleted:
        assert events.index(f"reference:{path}") + 1 == events.index(f"file:{path}")
    storage.iter_files_in_folder.assert_called_once_with(PREPARED_DOWNLOAD_PREFIX, page_size=2)


async def test_orphan_item_errors_leave_files_and_continue(
    session: _Session, storage: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    files = [_file(i) for i in range(1, 4)]
    storage.existing.update(file.url for file in files)
    session.reference_failures.add(files[0].url)
    original_delete = storage.delete_file.side_effect

    async def listing(*_: Any, **__: Any) -> AsyncIterator[FileItem]:
        for file in files:
            yield file

    async def delete(path: str) -> None:
        if path == files[1].url:
            raise DialException("unavailable", status_code=503)
        await original_delete(path)

    storage.iter_files_in_folder.side_effect = listing
    storage.delete_file.side_effect = delete
    await _service(session).cleanup(SystemUserAuthContext())

    assert storage.existing == {files[0].url, files[1].url}
    assert "Failed to clean up orphan" in caplog.text
    assert "rollback" in session.events


async def test_orphan_sweep_processes_each_page_before_next_and_releases_old_pages(
    session: _Session, storage: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    page_refs: list[weakref.ReferenceType[FileMetadata]] = []
    file_refs: list[list[weakref.ReferenceType[FileItem]]] = []
    fetched = 0

    async def get_metadata(**kwargs: Any) -> FileMetadata:
        nonlocal fetched
        assert not session.in_transaction
        assert kwargs["limit"] == 2
        assert kwargs["token"] == (str(fetched) if fetched else None)
        assert kwargs["relative_url"] == f"files/bucket/{PREPARED_DOWNLOAD_PREFIX}/"
        assert storage.delete_file.await_count == fetched * 2
        assert all(ref() is None for ref in page_refs)
        if fetched >= 2:
            assert all(ref() is None for ref in file_refs[fetched - 2])
        files = [_file(fetched * 2 + i) for i in (1, 2)]
        storage.existing.update(file.url for file in files)
        fetched += 1
        metadata = FileMetadata(
            bucket="bucket",
            url=kwargs["relative_url"],
            node_type="FOLDER",
            resource_type="FILE",
            items=files,
            next_token=str(fetched) if fetched < 4 else None,
        )
        page_refs.append(weakref.ref(metadata))
        file_refs.append([weakref.ref(file) for file in files])
        return metadata

    dial = MagicMock(spec=AsyncDial)
    dial.files = MagicMock()
    dial.files.metadata.get = AsyncMock(side_effect=get_metadata)
    incremental_storage = AttachmentsStorage(dial)
    incremental_storage.delete_file = storage.delete_file

    async def bucket(_: Any) -> str:
        return "bucket"

    monkeypatch.setattr("statgpt.common.utils.dial.attachments.resolve_bucket", bucket)

    @asynccontextmanager
    async def factory(_: str) -> AsyncIterator[AttachmentsStorage]:
        yield incremental_storage

    monkeypatch.setattr(service_module, "attachments_storage_factory", factory)
    await _service(session).cleanup(SystemUserAuthContext(), batch_size=2)

    assert fetched == 4
    assert storage.delete_file.await_count == 8
    assert all(ref() is None for refs in file_refs for ref in refs)
    assert not session.rows
