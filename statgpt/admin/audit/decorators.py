from collections.abc import Awaitable, Callable
from functools import wraps

from statgpt.admin.audit.service import AuditService
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType, AuditScope


def audit_action(
    *,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
    scope: AuditScope = AuditScope.CONFIG,
):
    """Decorated methods must not call session.commit(); this decorator owns transaction commit/rollback."""

    def decorator(func: Callable[..., Awaitable[Auditable]]) -> Callable[..., Awaitable[Auditable]]:
        @wraps(func)
        async def wrapped(self, *args, **kwargs) -> Auditable:
            if self._session.in_transaction():
                result: Auditable = await func(self, *args, **kwargs)
                await AuditService(self._session).persist(
                    entity_type=entity_type,
                    action_type=action_type,
                    data=result,
                    scope=scope,
                )
                return result

            async with self._session.begin():
                result = await func(self, *args, **kwargs)
                await AuditService(self._session).persist(
                    entity_type=entity_type,
                    action_type=action_type,
                    data=result,
                    scope=scope,
                )
                return result

        return wrapped

    return decorator
