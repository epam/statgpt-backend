import datetime
from typing import Any

from pydantic import BaseModel

from .enums import AuditActionType, AuditEntityType


class AuditLogListItem(BaseModel):
    id: int
    entity_type: AuditEntityType
    action_type: AuditActionType

    item_id: int
    entity_id: str
    entity_name: str

    performed_by: str
    performed_by_name: str
    trace_id: str
    created_at: datetime.datetime


class AuditLogDetails(AuditLogListItem):
    state_before: dict[str, Any] | None
    state_after: dict[str, Any] | None
