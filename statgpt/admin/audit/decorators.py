from __future__ import annotations

import logging
from functools import wraps
from typing import Any

from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit.context import get_audit_context
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType

_log = logging.getLogger(__name__)


def _normalize_json(value: Any) -> Any:
    if value is None:
        return None
    return jsonable_encoder(value)


async def _persist_audit_log(
    *,
    session: AsyncSession,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
    state_after: Any,
    item_id: int | None,
    entity_id: str | None,
    entity_name: str | None,
) -> None:
    context = get_audit_context()
    item = models.AuditLog(
        entity_type=entity_type,
        action_type=action_type,
        item_id=item_id,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=context.performed_by,
        performed_by_name=context.performed_by_name,
        state_after=_normalize_json(state_after),
        trace_id=context.trace_id,
    )
    session.add(item)
    await session.commit()


def audit_action(
    *,
    entity_type: AuditEntityType,
    action_type: AuditActionType,
):
    def decorator(func):
        @wraps(func)
        async def wrapped(self, *args, **kwargs):
            result: Auditable = await func(self, *args, **kwargs)
            try:
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    state_after=result.get_state_after(),
                    item_id=result.get_item_id(),
                    entity_id=result.get_entity_id(),
                    entity_name=result.get_entity_name(),
                )
            except Exception:
                _log.exception(
                    "Failed to persist audit log for %s action=%s", entity_type, action_type
                )

            return result

        return wrapped

    return decorator
