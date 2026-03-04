"""
Batch auto-update script for all datasets in channels with allow_auto_update enabled.
"""

import asyncio
import logging
import sys

import statgpt.common.schemas as schemas
from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services.channel import (
    AdminPortalChannelService,
    deduplicate_dimensions_in_background_task,
)
from statgpt.admin.services.dataset import AdminPortalDataSetService, auto_update_in_background_task
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.models import get_session_contex_manager, optional_msi_token_manager_context

_log = logging.getLogger(__name__)
_SEPARATOR = "-" * 50


async def _discover_and_create_jobs() -> list[schemas.AutoUpdateJob]:
    """Find auto-update channels and create jobs for their datasets."""
    _log.info(_SEPARATOR)
    async with get_session_contex_manager() as session:
        channels = await AdminPortalChannelService(session).get_auto_update_channels()
        _log.info(f"Found {len(channels)} channel(s) with auto-update enabled")

        if not channels:
            return []

        channel_ids = [ch.id for ch in channels]
        return await AdminPortalDataSetService(session).create_auto_update_jobs(channel_ids)


async def _process_jobs(jobs: list[schemas.AutoUpdateJob], auth_context: AuthContext) -> None:
    """Run all auto-update jobs concurrently."""
    _log.info(_SEPARATOR)
    _log.info(f"Created {len(jobs)} auto-update job(s), starting processing...")

    await asyncio.gather(
        *(
            auto_update_in_background_task(auto_update_job_id=job.id, auth_context=auth_context)
            for job in jobs
        ),
        return_exceptions=True,
    )


async def _check_results(job_ids: list[int]) -> tuple[bool, set[int]]:
    """Check job results and return (all_succeeded, channel_ids_with_reindex)."""
    _log.info(_SEPARATOR)
    async with get_session_contex_manager() as session:
        return await AdminPortalDataSetService(session).check_auto_update_results(job_ids)


async def _deduplicate_channels(channel_ids: set[int], auth_context: AuthContext) -> None:
    """Run deduplication for channels that had a reindex."""
    _log.info(_SEPARATOR)
    _log.info(
        f"Running deduplication for {len(channel_ids)} channel(s) "
        f"with reindex: {sorted(channel_ids)}"
    )
    await asyncio.gather(
        *(
            deduplicate_dimensions_in_background_task(
                channel_id=channel_id, auth_context=auth_context
            )
            for channel_id in channel_ids
        ),
        return_exceptions=True,
    )
    _log.info("Deduplication complete")


async def run_auto_update() -> bool:
    """Run batch auto-update for all eligible channels.

    Returns True if all jobs succeeded, False otherwise.
    """
    auth_context = SystemUserAuthContext()

    jobs = await _discover_and_create_jobs()
    if not jobs:
        return True

    await _process_jobs(jobs, auth_context)
    job_ids = [j.id for j in jobs]
    success, reindex_channel_ids = await _check_results(job_ids)

    if reindex_channel_ids:
        await _deduplicate_channels(reindex_channel_ids, auth_context)

    return success


async def main() -> None:
    try:
        _log.info("Starting batch auto-update script...")
        async with optional_msi_token_manager_context():
            success = await run_auto_update()

        _log.info(_SEPARATOR)
        if not success:
            _log.error("Batch auto-update finished with failures")
            sys.exit(1)
        _log.info("Batch auto-update script completed successfully")
    except Exception:
        _log.exception("Error in batch auto-update script:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
