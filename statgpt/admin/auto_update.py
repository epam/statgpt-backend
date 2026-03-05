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
        channel_service = AdminPortalChannelService(session)
        all_channels = await channel_service.get_channels_schemas(limit=None, offset=0)
        channel_ids = [
            ch.id
            for ch in all_channels
            if (dq := ch.details.data_query) is not None and dq.details.allow_auto_update
        ]
        _log.info(f"Found {len(channel_ids)} channel(s) with auto-update enabled")

        if not channel_ids:
            return []

        return await AdminPortalDataSetService(session).create_auto_update_jobs(channel_ids)


async def _process_jobs(jobs: list[schemas.AutoUpdateJob], auth_context: AuthContext) -> None:
    """Run all auto-update jobs concurrently."""
    _log.info(_SEPARATOR)
    _log.info(f"Created {len(jobs)} auto-update job(s), starting processing...")

    results = await asyncio.gather(
        *(
            auto_update_in_background_task(auto_update_job_id=job.id, auth_context=auth_context)
            for job in jobs
        ),
        return_exceptions=True,
    )
    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            _log.error(f"Auto-update job {job.id} failed with exception:", exc_info=result)


async def _get_reindex_channel_ids(job_ids: list[int]) -> set[int]:
    """Get channel IDs that had at least one reindex triggered."""
    async with get_session_contex_manager() as session:
        return await AdminPortalDataSetService(session).get_reindex_channel_ids(job_ids)


async def _log_results(job_ids: list[int]) -> bool:
    """Log per-channel summary and return True if all jobs succeeded."""
    _log.info(_SEPARATOR)
    async with get_session_contex_manager() as session:
        results = await AdminPortalDataSetService(session).get_auto_update_results(job_ids)

    for r in results:
        _log.info(f"channel '{r.deployment_id}' (id={r.channel_id}): {r.summary}")
        for reason in r.failed_reasons:
            _log.error(f"  {reason}")

    total = sum(r.total for r in results)
    failed = sum(r.failed for r in results)
    _log.info(
        f"Auto-update complete: {total - failed} succeeded, {failed} failed "
        f"out of {total} total"
    )
    return failed == 0


async def _deduplicate_channels(channel_ids: set[int], auth_context: AuthContext) -> None:
    """Run deduplication for channels that had a reindex."""
    _log.info(_SEPARATOR)
    _log.info(
        f"Running deduplication for {len(channel_ids)} channel(s) "
        f"with reindex: {sorted(channel_ids)}"
    )
    results = await asyncio.gather(
        *(
            deduplicate_dimensions_in_background_task(
                channel_id=channel_id, auth_context=auth_context
            )
            for channel_id in channel_ids
        ),
        return_exceptions=True,
    )
    for channel_id, result in zip(channel_ids, results):
        if isinstance(result, Exception):
            _log.error(
                f"Deduplication for channel {channel_id} failed with exception:", exc_info=result
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

    reindex_channel_ids = await _get_reindex_channel_ids(job_ids)
    if reindex_channel_ids:
        await _deduplicate_channels(reindex_channel_ids, auth_context)

    return await _log_results(job_ids)


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
