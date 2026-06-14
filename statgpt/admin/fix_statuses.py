"""
Script to fix statuses after migrations or crashes.
Sets failed status for any records that were left in a non-final state.
"""

import asyncio
import logging

from statgpt.admin.services import AdminPortalChannelService, AdminPortalDataSetService
from statgpt.common.models import get_session_context_manager, optional_msi_token_manager_context

_log = logging.getLogger(__name__)


async def _fix_channel_dataset_versions() -> None:
    try:
        async with get_session_context_manager() as session:
            service = AdminPortalDataSetService(session)
            await service.set_failed_status_for_channel_dataset_version()
    except Exception:
        _log.exception("Error fixing channel dataset version statuses:")


async def _fix_jobs() -> None:
    try:
        async with get_session_context_manager() as session:
            service = AdminPortalDataSetService(session)
            await service.set_failed_status_for_stuck_jobs()
    except Exception:
        _log.exception("Error fixing job statuses:")


async def _fix_auto_update_jobs() -> None:
    try:
        async with get_session_context_manager() as session:
            service = AdminPortalDataSetService(session)
            await service.set_failed_status_for_stuck_auto_update_jobs()
    except Exception:
        _log.exception("Error fixing auto-update job statuses:")


async def _fix_deduplication_jobs() -> None:
    try:
        async with get_session_context_manager() as session:
            service = AdminPortalChannelService(session)
            await service.set_failed_status_for_stuck_deduplication_jobs()
    except Exception:
        _log.exception("Error fixing deduplication job statuses:")


async def fix_statuses() -> None:
    """Fix statuses from previous runs by setting failed status for stuck records."""
    await _fix_channel_dataset_versions()
    await _fix_jobs()
    await _fix_auto_update_jobs()
    await _fix_deduplication_jobs()


async def main():
    try:
        _log.info("Starting fix_statuses script...")
        async with optional_msi_token_manager_context():
            await fix_statuses()
        _log.info("fix_statuses script completed successfully")
    except Exception:
        _log.exception("Error in fix_statuses script:")
        raise


if __name__ == '__main__':
    asyncio.run(main())
