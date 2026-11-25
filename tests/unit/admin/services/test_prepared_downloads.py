import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from aidial_client.types.metadata import FileItem
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.auth.user import User
from statgpt.admin.services import prepared_downloads
from statgpt.admin.services.exceptions import (
    PreparedDownloadChannelNotFoundError,
    PreparedDownloadStorageError,
)
from statgpt.common.utils.dial.attachments import AttachmentsStorage

_ID = UUID("10f90c65-3e73-47d5-b1ac-ff51f0a90523")
_NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
_FILE = f"files/test-bucket/prepared-downloads/{_ID}.csv"
_USER = User("admin-a", "admin-a", "Admin A")


@dataclass
class ServiceHarness:
    session: MagicMock
    storage: MagicMock
    service: prepared_downloads.PreparedDownloadService
    now: datetime = _NOW

    async def prepare(self):
        return await self.service.prepare(
            channel_id=7,
            user=_USER,
            auth_context=SystemUserAuthContext(),
            content=b"original snapshot",
            filename="discovery_datasets.csv",
            media_type="text/csv",
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[ServiceHarness]:
    session = MagicMock(spec=AsyncSession)
    session.scalar.return_value = 7
    storage = MagicMock(spec=AttachmentsStorage)
    storage.put_file.return_value = FileItem(
        bucket="test-bucket", url=_FILE, node_type="ITEM", resource_type="FILE"
    )
    state = ServiceHarness(session, storage, prepared_downloads.PreparedDownloadService(session))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return state.now

    @asynccontextmanager
    async def storage_context(*args, **kwargs):
        yield storage

    monkeypatch.setattr(prepared_downloads, "datetime", Clock)
    monkeypatch.setattr(prepared_downloads, "uuid4", lambda: _ID)
    monkeypatch.setattr(prepared_downloads, "attachments_storage_factory", storage_context)
    yield state


@pytest.mark.asyncio
async def test_deadline_and_complete_metadata_are_captured_after_upload(
    harness: ServiceHarness,
) -> None:
    async def upload(*args):
        harness.now += timedelta(minutes=15)
        return harness.storage.put_file.return_value

    harness.storage.put_file.side_effect = upload
    response = await harness.prepare()
    assert response.id == _ID
    assert response.expires_at == _NOW + timedelta(hours=1, minutes=15)
    assert response.expires_at.utcoffset() == timedelta(0)
    harness.storage.put_file.assert_awaited_once_with(
        f"prepared-downloads/{_ID}.csv", "text/csv", b"original snapshot"
    )
    harness.session.commit.assert_awaited_once()
    assert harness.session.add.call_args is not None
    record = harness.session.add.call_args.args[0]
    assert record.id == _ID
    assert record.owner_user_id == _USER.id
    assert record.channel_id == 7
    assert record.file_path == _FILE
    assert record.filename == "discovery_datasets.csv"
    assert record.media_type == "text/csv"
    assert record.created_at == record.updated_at == harness.now
    assert record.expires_at == response.expires_at
    harness.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_failure_never_commits_metadata(harness: ServiceHarness) -> None:
    failure = RuntimeError("upload failed")
    harness.storage.put_file.side_effect = failure
    with pytest.raises(PreparedDownloadStorageError) as caught:
        await harness.prepare()
    assert caught.value.__cause__ is failure
    harness.session.add.assert_not_called()
    harness.session.commit.assert_not_awaited()
    harness.storage.delete_file.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deletion_fails, rollback_fails", [(False, False), (True, False), (False, True)]
)
async def test_commit_failure_compensates_without_masking_original(
    harness: ServiceHarness, deletion_fails: bool, rollback_fails: bool
) -> None:
    failure = RuntimeError("database commit failed")
    harness.session.commit.side_effect = failure
    if deletion_fails:
        harness.storage.delete_file.side_effect = OSError("compensation deletion failed")
    if rollback_fails:
        harness.session.rollback.side_effect = OSError("compensation rollback failed")
    with pytest.raises(RuntimeError) as caught:
        await harness.prepare()
    assert caught.value is failure
    harness.session.rollback.assert_awaited_once()
    harness.storage.delete_file.assert_awaited_once_with(_FILE)


@pytest.mark.asyncio
async def test_cancellation_after_upload_awaits_compensation(harness: ServiceHarness) -> None:
    harness.session.commit.side_effect = asyncio.CancelledError("cancelled during commit")
    with pytest.raises(asyncio.CancelledError):
        await harness.prepare()
    harness.session.rollback.assert_awaited_once()
    harness.storage.delete_file.assert_awaited_once_with(_FILE)


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_abandon_compensation(harness: ServiceHarness) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    harness.session.commit.side_effect = asyncio.CancelledError("initial cancellation")

    async def remove(url: str):
        started.set()
        await release.wait()
        finished.set()

    harness.storage.delete_file.side_effect = remove
    task = asyncio.create_task(harness.prepare())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert finished.is_set()
    harness.storage.delete_file.assert_awaited_once_with(_FILE)


@pytest.mark.asyncio
async def test_channel_disappearing_after_upload_is_not_found_and_compensated(
    harness: ServiceHarness,
) -> None:
    harness.session.scalar.return_value = None
    with pytest.raises(PreparedDownloadChannelNotFoundError):
        await harness.prepare()
    harness.session.add.assert_not_called()
    harness.session.commit.assert_not_awaited()
    harness.storage.delete_file.assert_awaited_once_with(_FILE)


class DatabaseError(Exception):
    def __init__(self, sqlstate: str):
        self.sqlstate = sqlstate
        super().__init__(sqlstate)


@pytest.mark.asyncio
async def test_channel_foreign_key_failure_is_resolved_after_rollback(
    harness: ServiceHarness,
) -> None:
    harness.session.commit.side_effect = IntegrityError("insert", {}, DatabaseError("23503"))
    checks = 0

    async def check_channel(query):
        nonlocal checks
        checks += 1
        if checks == 1:
            return 7
        assert harness.session.rollback.await_count == 1
        return None

    harness.session.scalar.side_effect = check_channel
    with pytest.raises(PreparedDownloadChannelNotFoundError):
        await harness.prepare()
    assert checks == 2
    harness.storage.delete_file.assert_awaited_once_with(_FILE)


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["23505", "23503"])
async def test_unrelated_integrity_errors_propagate_unchanged(
    harness: ServiceHarness, sqlstate: str
) -> None:
    failure = IntegrityError("insert", {}, DatabaseError(sqlstate))
    harness.session.commit.side_effect = failure
    with pytest.raises(IntegrityError) as caught:
        await harness.prepare()
    assert caught.value is failure
    harness.storage.delete_file.assert_awaited_once_with(_FILE)
