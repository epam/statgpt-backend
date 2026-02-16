import logging
from collections.abc import Awaitable, Callable
from functools import wraps

from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit.context import get_audit_context
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType

_log = logging.getLogger(__name__)


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
    await session.commit()


def audit_action(
    *,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
):
    def decorator(func: Callable[..., Awaitable[Auditable]]) -> Callable[..., Awaitable[Auditable]]:
        @wraps(func)
        async def wrapped(self, *args, **kwargs) -> Auditable:
            result: Auditable = await func(self, *args, **kwargs)
            try:
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    data=result,
                )
            except Exception:
                _log.exception(
                    f"Failed to persist audit log for {entity_type} action={action_type}"
                )
                # TODO: Probably we should also roll back the session here

            return result

        return wrapped

    return decorator
