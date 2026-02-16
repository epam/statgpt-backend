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
    performed_by: str
    performed_by_name: str
    trace_id: str | None = field(default_factory=_get_trace_id)


_audit_context_var: ContextVar[AuditContext] = ContextVar("admin_audit_context")


def set_audit_context(*, performed_by: str, performed_by_name: str) -> None:
    _audit_context_var.set(
        AuditContext(
            performed_by=performed_by,
            performed_by_name=performed_by_name,
        )
    )


def update_audit_context(audit_context: AuditContext) -> None:
    _audit_context_var.set(audit_context)


def get_audit_context() -> AuditContext:
    if res := _audit_context_var.get(None):
        return res
    raise RuntimeError("Audit context not set")
