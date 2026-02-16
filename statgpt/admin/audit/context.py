from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditContext:
    performed_by: str | None
    performed_by_name: str | None


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


def get_audit_context() -> AuditContext:
    return _audit_context_var.get()
