"""Shared status-recovery helper used by fix_statuses-style cleanups.

Lives in its own module so multiple admin services can use it without
depending on each other (avoids circular imports between channel and
dataset services).
"""

import logging
from typing import Any

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum

_log = logging.getLogger(__name__)


async def set_failed_status(
    session: AsyncSession,
    model: Any,
    status_column: Any,
    status_field_name: str,
) -> int:
    """Set the status of all stuck records to FAILED for the given model.

    A record is considered stuck when its status is not a final status and it
    has not been touched for more than 12 hours.

    Returns the number of updated rows.
    """
    table_name = model.__tablename__

    _log.info(f"Setting FAILED status for all non-completed {table_name}...")

    query = (
        update(model)
        .where(
            status_column.notin_(StatusEnum.final_statuses()),
            model.updated_at < text("NOW() - INTERVAL '12 hours'"),
        )
        .values(
            **{status_field_name: StatusEnum.FAILED},
            reason_for_failure=func.coalesce(
                model.reason_for_failure,
                "Stuck in a non-final status with no recorded failure reason."
                " Marked as FAILED by fix_statuses script.",
            ),
            updated_at=func.now(),
        )
    )

    result = await session.execute(query)
    row_count: int = result.rowcount  # type: ignore[attr-defined]

    _log.info(f"Updated {row_count} {table_name} record(s) to FAILED status")
    return row_count
