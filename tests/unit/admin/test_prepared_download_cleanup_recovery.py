from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aidial_client import DialException
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin import fix_statuses as maintenance
from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services import prepared_downloads as service_module
from statgpt.common.utils.dial.attachments import AttachmentsStorage

_RECOVERY_METHODS = (
    "set_failed_status_for_channel_dataset_version",
    "set_failed_status_for_stuck_jobs",
    "set_failed_status_for_stuck_auto_update_jobs",
    "set_failed_status_for_stuck_deduplication_jobs",
    "set_failed_status_for_stuck_discovery_indexing_jobs",
)


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[MagicMock]:
    sessions: list[MagicMock] = []

    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncSession]:
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = []
        session.execute = AsyncMock(return_value=result)
        sessions.append(session)
        yield session

    monkeypatch.setattr(maintenance, "get_session_context_manager", context)
    return sessions


@pytest.fixture(autouse=True)
def _recovery_services(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    service = SimpleNamespace(
        **{
            name: AsyncMock(side_effect=lambda name=name: events.append(name))
            for name in _RECOVERY_METHODS
        }
    )
    for name in (
        "AdminPortalDataSetService",
        "AdminPortalChannelService",
        "AdminPortalDiscoveryIndexingJobService",
    ):
        monkeypatch.setattr(maintenance, name, lambda _: service)


async def test_cleanup_runs_after_recovery_with_independent_session_and_system_auth(
    monkeypatch: pytest.MonkeyPatch, sessions: list[MagicMock], events: list[str]
) -> None:
    cleanup = AsyncMock(side_effect=lambda **_: events.append("cleanup"))
    constructor = MagicMock(return_value=SimpleNamespace(cleanup=cleanup))
    monkeypatch.setattr(maintenance, "PreparedDownloadService", constructor)

    await maintenance.fix_statuses()

    assert events == [*_RECOVERY_METHODS, "cleanup"]
    assert len(sessions) == 6
    assert len({id(session) for session in sessions}) == 6
    constructor.assert_called_once_with(sessions[-1])
    assert cleanup.await_args is not None
    auth = cleanup.await_args.kwargs["auth_context"]
    assert isinstance(auth, SystemUserAuthContext)
    assert auth.is_system


@pytest.mark.parametrize("failure", ["cleanup", "storage", "listing"])
async def test_cleanup_failures_are_logged_without_changing_recovery_or_msi_lifecycle(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    sessions: list[MagicMock],
    events: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    @asynccontextmanager
    async def msi_context() -> AsyncIterator[None]:
        events.append("msi-enter")
        try:
            yield
        finally:
            events.append("msi-exit")

    monkeypatch.setattr(maintenance, "optional_msi_token_manager_context", msi_context)
    if failure == "cleanup":
        cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        monkeypatch.setattr(
            maintenance, "PreparedDownloadService", lambda _: SimpleNamespace(cleanup=cleanup)
        )
    else:

        async def listing(*_: Any, **__: Any) -> AsyncIterator[Any]:
            raise DialException("listing failed", status_code=503)
            yield

        @asynccontextmanager
        async def storage_factory(_: str) -> AsyncIterator[AttachmentsStorage]:
            if failure == "storage":
                raise DialException("storage failed", status_code=503)
            storage = MagicMock(spec=AttachmentsStorage)
            storage.iter_files_in_folder.side_effect = listing
            yield storage

        monkeypatch.setattr(service_module, "attachments_storage_factory", storage_factory)

    await maintenance.main()

    assert events == ["msi-enter", *_RECOVERY_METHODS, "msi-exit"]
    assert len(sessions) == 6
    assert "Error cleaning up prepared downloads" in caplog.text
    assert "Error in fix_statuses script" not in caplog.text


async def test_recovery_failure_does_not_skip_other_recovery_or_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    sessions: list[MagicMock],
    events: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad_service = SimpleNamespace(
        set_failed_status_for_stuck_deduplication_jobs=AsyncMock(
            side_effect=RuntimeError("recovery failed")
        )
    )
    monkeypatch.setattr(maintenance, "AdminPortalChannelService", lambda _: bad_service)
    cleanup = AsyncMock(side_effect=lambda **_: events.append("cleanup"))
    monkeypatch.setattr(
        maintenance, "PreparedDownloadService", lambda _: SimpleNamespace(cleanup=cleanup)
    )

    await maintenance.fix_statuses()

    assert events == [
        *[name for name in _RECOVERY_METHODS if "deduplication" not in name],
        "cleanup",
    ]
    assert len(sessions) == 6
    assert "Error fixing deduplication job statuses" in caplog.text
    cleanup.assert_awaited_once()
