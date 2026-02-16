from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi.encoders import jsonable_encoder
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit.context import get_audit_context
from statgpt.common.schemas.auditable import Auditable

_log = logging.getLogger(__name__)

StateAfterGetter = Callable[..., Awaitable[Any] | Any]


async def _await_if_needed(value: Awaitable[Any] | Any) -> Any:
    if hasattr(value, "__await__"):
        return await value  # type: ignore[misc]
    return value


def _get_trace_id() -> str | None:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return format(span_context.trace_id, "032x")


def _normalize_json(value: Any) -> Any:
    if value is None:
        return None
    return jsonable_encoder(value)


def _extract_default_entity_ref(result: Any) -> tuple[str | None, str | None]:
    if isinstance(result, Auditable):
        return result.get_entity_id(), result.get_entity_name()

    state = _normalize_json(result)
    if not isinstance(state, dict):
        return None, None
    raw_entity_id = state.get("id")
    raw_name = state.get("title", state.get("name"))
    entity_id = str(raw_entity_id) if raw_entity_id is not None else None
    entity_name = str(raw_name) if raw_name is not None else None
    return entity_id, entity_name


async def _persist_audit_log(
    *,
    session: AsyncSession,
    entity_type: str,
    action_type: str,
    state_after: Any,
    entity_id: str | None,
    entity_name: str | None,
) -> None:
    context = get_audit_context()
    item = models.AuditLog(
        entity_type=entity_type,
        action_type=action_type,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=context.performed_by,
        performed_by_name=context.performed_by_name,
        state_after=_normalize_json(state_after),
        trace_id=_get_trace_id(),
    )
    session.add(item)
    await session.commit()


def audit_action(
    *,
    entity_type: str,
    action_type: str,
    state_after_getter: StateAfterGetter | None = None,
):
    def decorator(func):
        @wraps(func)
        async def wrapped(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)

            state_after = None if action_type == "delete" else result
            if state_after_getter is not None:
                state_after = await _await_if_needed(
                    state_after_getter(self, result, *args, **kwargs)
                )

            entity_id, entity_name = _extract_default_entity_ref(result)

            try:
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    state_after=state_after,
                    entity_id=entity_id,
                    entity_name=entity_name,
                )
            except Exception:
                _log.exception(
                    "Failed to persist audit log for %s action=%s", entity_type, action_type
                )

            return result

        return wrapped

    return decorator
