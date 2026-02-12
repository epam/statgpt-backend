import datetime

from pydantic import BaseModel

from .enums import AuditStateEnum


class AuditLog(BaseModel):
    id: int
    entity_type: str
    action_type: str
    entity_id: str | None
    entity_name: str | None
    performed_by: str | None
    performed_by_name: str | None
    action_trigger: str
    state_before: AuditStateEnum
    state_after: AuditStateEnum
    trace_id: str | None
    created_at: datetime.datetime
