import datetime
from typing import Any

from pydantic import BaseModel


class AuditLogListItem(BaseModel):
    id: int
    entity_type: str
    action_type: str
    entity_id: str | None
    entity_name: str | None
    performed_by: str | None
    performed_by_name: str | None
    trace_id: str | None
    created_at: datetime.datetime


class AuditLogDetails(AuditLogListItem):
    state_before: dict[str, Any] | list[Any] | str | int | float | bool | None
    state_after: dict[str, Any] | list[Any] | str | int | float | bool | None
