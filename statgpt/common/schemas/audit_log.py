import datetime
from typing import Any

from pydantic import BaseModel

from .enums import AuditActionType, AuditEntityType


class AuditLogListItem(BaseModel):
    id: int
    entity_type: AuditEntityType
    action_type: AuditActionType
    item_id: int | None
    entity_id: str | None
    entity_name: str | None
    performed_by: str | None
    performed_by_name: str | None
    trace_id: str | None
    created_at: datetime.datetime


class AuditLogDetails(AuditLogListItem):
    state_before: dict[str, Any] | list[Any] | str | int | float | bool | None
    state_after: dict[str, Any] | list[Any] | str | int | float | bool | None
