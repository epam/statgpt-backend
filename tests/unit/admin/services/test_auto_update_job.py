import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.dataset import AdminPortalDataSetService
from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum


class TestProcessAutoUpdateJobCancellation:
    """Regression guard: a ``@background_task`` timeout cancels the coroutine,
    surfacing inside ``process_auto_update_job`` as ``CancelledError`` (which the
    ``except Exception`` branch does NOT catch). The job must still be marked
    FAILED instead of being left stuck in IN_PROGRESS, and the cancellation must
    re-raise so the decorator's timeout machinery keeps working.
    """

    @pytest.mark.asyncio
    async def test_cancellation_marks_job_failed_and_reraises(self) -> None:
        job = MagicMock()
        job.status = StatusEnum.IN_PROGRESS

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=job)

        service = AdminPortalDataSetService()
        # Inject a session so _scoped_session() yields it directly (no real DB).
        service._DbServiceBase__session = mock_session  # type: ignore[attr-defined]
        # Cancel the task right after the initial IN_PROGRESS status write.
        service._set_auto_update_job_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await service.process_auto_update_job(auto_update_job_id=1, auth_context=MagicMock())

        assert job.status == StatusEnum.FAILED
        assert job.reason_for_failure
        mock_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_does_not_clobber_completed_job(self) -> None:
        """A late cancellation must not overwrite an already-terminal COMPLETED
        job (e.g. one that finished with REINDEX_TRIGGERED)."""
        job = MagicMock()
        job.status = StatusEnum.COMPLETED

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=job)

        service = AdminPortalDataSetService()
        # Inject a session so _scoped_session() yields it directly (no real DB).
        service._DbServiceBase__session = mock_session  # type: ignore[attr-defined]
        service._set_auto_update_job_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await service.process_auto_update_job(auto_update_job_id=1, auth_context=MagicMock())

        assert job.status == StatusEnum.COMPLETED
        mock_session.commit.assert_not_awaited()
