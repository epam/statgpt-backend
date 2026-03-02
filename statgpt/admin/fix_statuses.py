"""
Script to fix statuses for channel dataset versions after migrations.
Sets failed status for any channel dataset versions that were left in processing state.
"""

import asyncio
import logging

from statgpt.admin.services import AdminPortalDataSetService
from statgpt.common.models import get_session_contex_manager, optional_msi_token_manager_context

_log = logging.getLogger(__name__)


async def fix_statuses():
    """Fix statuses from previous runs by setting failed status for stuck channel dataset versions."""
    async with get_session_contex_manager() as session:
        service = AdminPortalDataSetService(session)
        await service.set_failed_status_for_channel_dataset_version()
    _log.info("Successfully fixed statuses for channel dataset versions")


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
