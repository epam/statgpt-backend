from collections.abc import Awaitable, Callable
from functools import wraps

from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit.context import get_audit_context
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType


async def _persist_audit_log(
    *,
    session: AsyncSession,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
    data: Auditable,
) -> None:
    context = get_audit_context()
    state_after = None if action_type is AuditActionType.DELETE else data.get_state_after()

    item = models.AuditLog(
        entity_type=entity_type,
        action_type=action_type,
        item_id=data.get_item_id(),
        entity_id=data.get_entity_id(),
        entity_name=data.get_entity_name(),
        performed_by=context.performed_by,
        performed_by_name=context.performed_by_name,
        state_after=state_after,
        trace_id=context.trace_id,
    )
    session.add(item)


def audit_action(
    *,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
):
    # Decorated methods must not call session.commit(); this decorator owns transaction commit/rollback.
    def decorator(func: Callable[..., Awaitable[Auditable]]) -> Callable[..., Awaitable[Auditable]]:
        @wraps(func)
        async def wrapped(self, *args, **kwargs) -> Auditable:
            if self._session.in_transaction():
                result: Auditable = await func(self, *args, **kwargs)
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    data=result,
                )
                return result

            async with self._session.begin():
                result = await func(self, *args, **kwargs)
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    data=result,
                )
                return result

        return wrapped

    return decorator
