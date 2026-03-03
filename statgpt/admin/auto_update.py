"""
Batch auto-update script for all datasets in channels with allow_auto_update enabled.
"""

import asyncio
import logging
import sys

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services.channel import AdminPortalChannelService
from statgpt.admin.services.dataset import AdminPortalDataSetService, auto_update_in_background_task
from statgpt.common.models import get_session_contex_manager, optional_msi_token_manager_context

_log = logging.getLogger(__name__)


async def run_auto_update() -> bool:
    """Run batch auto-update for all eligible channels.

    Returns True if all jobs succeeded, False otherwise.
    """
    auth_context = SystemUserAuthContext()

    async with get_session_contex_manager() as session:
        channels = await AdminPortalChannelService(session).get_auto_update_channels()
        _log.info(f"Found {len(channels)} channel(s) with auto-update enabled")

        if not channels:
            return True

        job_ids = await AdminPortalDataSetService(session).create_auto_update_jobs(channels)

    _log.info(f"Created {len(job_ids)} auto-update job(s), starting processing...")

    # Process all jobs concurrently — @background_task semaphore controls concurrency
    await asyncio.gather(
        *(
            auto_update_in_background_task(auto_update_job_id=job_id, auth_context=auth_context)
            for job_id in job_ids
        ),
        return_exceptions=True,
    )

    async with get_session_contex_manager() as session:
        return await AdminPortalDataSetService(session).check_auto_update_results(job_ids)


async def main() -> None:
    try:
        _log.info("Starting batch auto-update script...")
        async with optional_msi_token_manager_context():
            success = await run_auto_update()
        if not success:
            _log.error("Batch auto-update finished with failures")
            sys.exit(1)
        _log.info("Batch auto-update script completed successfully")
    except Exception:
        _log.exception("Error in batch auto-update script:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
