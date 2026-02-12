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
from statgpt.common.schemas import AuditStateEnum

_log = logging.getLogger(__name__)

BeforeStateGetter = Callable[..., Awaitable[Any] | Any]
AfterStateGetter = Callable[..., Awaitable[Any] | Any]
EntityRefGetter = Callable[
    ..., Awaitable[tuple[str | None, str | None]] | tuple[str | None, str | None]
]


async def _await_if_needed(value: Awaitable[Any] | Any) -> Any:
    if hasattr(value, "__await__"):
        return await value  # type: ignore[misc]
    return value


def _extract_default_entity_ref(
    before_state: Any, after_state: Any
) -> tuple[str | None, str | None]:
    state = after_state if after_state is not None else before_state
    if not isinstance(state, dict):
        return None, None
    raw_entity_id = state.get("id")
    raw_name = state.get("title", state.get("name"))
    entity_id = str(raw_entity_id) if raw_entity_id is not None else None
    entity_name = str(raw_name) if raw_name is not None else None
    return entity_id, entity_name


def _get_trace_id() -> str | None:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return format(span_context.trace_id, "032x")


def _normalize_for_diff(value: Any) -> Any:
    if value is None:
        return None
    return jsonable_encoder(value)


def _resolve_state_transition(
    before_state: Any, after_state: Any
) -> tuple[AuditStateEnum, AuditStateEnum]:
    before_norm = _normalize_for_diff(before_state)
    after_norm = _normalize_for_diff(after_state)

    if before_norm is None and after_norm is None:
        return AuditStateEnum.ABSENT, AuditStateEnum.NOT_CHANGED
    if before_norm is None and after_norm is not None:
        return AuditStateEnum.ABSENT, AuditStateEnum.CREATED
    if before_norm is not None and after_norm is None:
        return AuditStateEnum.EXISTS, AuditStateEnum.DELETED
    if before_norm == after_norm:
        return AuditStateEnum.EXISTS, AuditStateEnum.NOT_CHANGED
    return AuditStateEnum.EXISTS, AuditStateEnum.MODIFIED


async def _persist_audit_log(
    *,
    session: AsyncSession,
    entity_type: str,
    action_type: str,
    before_state: Any,
    after_state: Any,
    entity_id: str | None,
    entity_name: str | None,
) -> None:
    context = get_audit_context()
    state_before, state_after = _resolve_state_transition(before_state, after_state)
    item = models.AuditLog(
        entity_type=entity_type,
        action_type=action_type,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=context.performed_by,
        performed_by_name=context.performed_by_name,
        action_trigger=context.action_trigger,
        state_before=state_before,
        state_after=state_after,
        trace_id=_get_trace_id(),
    )
    session.add(item)
    await session.commit()


def audit_action(
    *,
    entity_type: str,
    action_type: str,
    before_state_getter: BeforeStateGetter | None = None,
    after_state_getter: AfterStateGetter | None = None,
    entity_ref_getter: EntityRefGetter | None = None,
):
    def decorator(func):
        @wraps(func)
        async def wrapped(self, *args, **kwargs):
            before_state = None
            if before_state_getter is not None:
                before_state = await _await_if_needed(before_state_getter(self, *args, **kwargs))

            result = await func(self, *args, **kwargs)

            after_state = result
            if after_state_getter is not None:
                after_state = await _await_if_needed(
                    after_state_getter(self, result, *args, **kwargs)
                )

            if entity_ref_getter is not None:
                entity_id, entity_name = await _await_if_needed(
                    entity_ref_getter(self, result, before_state, after_state, *args, **kwargs)
                )
            else:
                entity_id, entity_name = _extract_default_entity_ref(before_state, after_state)

            try:
                await _persist_audit_log(
                    session=self._session,
                    entity_type=entity_type,
                    action_type=action_type,
                    before_state=before_state,
                    after_state=after_state,
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
