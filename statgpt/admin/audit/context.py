from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from opentelemetry import trace


def _get_trace_id() -> str | None:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return format(span_context.trace_id, "032x")


@dataclass(frozen=True)
class AuditContext:
    performed_by: str | None
    performed_by_name: str | None
    trace_id: str | None = field(default_factory=_get_trace_id)


_audit_context_var: ContextVar[AuditContext] = ContextVar(
    "admin_audit_context",
    default=AuditContext(performed_by=None, performed_by_name="system"),
)


def set_audit_context(*, performed_by: str | None, performed_by_name: str | None) -> None:
    _audit_context_var.set(
        AuditContext(
            performed_by=performed_by,
            performed_by_name=performed_by_name,
        )
    )


def update_audit_context(audit_context: AuditContext) -> None:
    _audit_context_var.set(audit_context)


def get_audit_context() -> AuditContext:
    return _audit_context_var.get()
