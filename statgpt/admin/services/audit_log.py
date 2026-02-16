from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas


class AdminAuditLogService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, item_id: int) -> schemas.AuditLogDetails:
        item = await self._session.get(models.AuditLog, item_id)
        if item is None:
            raise ValueError(f"Audit log with id={item_id} not found")
        state_before = None
        if item.entity_id is not None:
            previous_query = (
                select(models.AuditLog)
                .where(
                    models.AuditLog.entity_type == item.entity_type,
                    models.AuditLog.entity_id == item.entity_id,
                    models.AuditLog.id < item.id,
                )
                .order_by(models.AuditLog.id.desc())
                .limit(1)
            )
            previous = (await self._session.execute(previous_query)).scalar_one_or_none()
            if previous is not None:
                state_before = previous.state_after

        return schemas.AuditLogDetails(
            id=item.id,
            entity_type=item.entity_type,
            action_type=item.action_type,
            entity_id=item.entity_id,
            entity_name=item.entity_name,
            performed_by=item.performed_by,
            performed_by_name=item.performed_by_name,
            trace_id=item.trace_id,
            created_at=item.created_at,
            state_before=state_before,
            state_after=item.state_after,
        )

    async def get_logs(
        self,
        *,
        limit: int,
        offset: int,
        entity_type: str | None = None,
        action_type: str | None = None,
        entity_id: str | None = None,
        performed_by: str | None = None,
    ) -> list[schemas.AuditLogListItem]:
        query = select(models.AuditLog).order_by(models.AuditLog.created_at.desc())
        query = self._apply_filters(
            query,
            entity_type=entity_type,
            action_type=action_type,
            entity_id=entity_id,
            performed_by=performed_by,
        )
        query = query.limit(limit).offset(offset)
        result = await self._session.execute(query)
        return [
            schemas.AuditLogListItem(
                id=item.id,
                entity_type=item.entity_type,
                action_type=item.action_type,
                entity_id=item.entity_id,
                entity_name=item.entity_name,
                performed_by=item.performed_by,
                performed_by_name=item.performed_by_name,
                trace_id=item.trace_id,
                created_at=item.created_at,
            )
            for item in result.scalars()
        ]

    async def get_count(
        self,
        *,
        entity_type: str | None = None,
        action_type: str | None = None,
        entity_id: str | None = None,
        performed_by: str | None = None,
    ) -> int:
        query = select(func.count("*")).select_from(models.AuditLog)
        query = self._apply_filters(
            query,
            entity_type=entity_type,
            action_type=action_type,
            entity_id=entity_id,
            performed_by=performed_by,
        )
        return (await self._session.execute(query)).scalar_one()

    @staticmethod
    def _apply_filters(query, **filters):
        if filters["entity_type"]:
            query = query.where(models.AuditLog.entity_type == filters["entity_type"])
        if filters["action_type"]:
            query = query.where(models.AuditLog.action_type == filters["action_type"])
        if filters["entity_id"]:
            query = query.where(models.AuditLog.entity_id == filters["entity_id"])
        if filters["performed_by"]:
            query = query.where(models.AuditLog.performed_by == filters["performed_by"])
        return query
