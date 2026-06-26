from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit import context as audit_context
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def persist(
        self,
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        data: Auditable,
    ) -> None:
        self.persist_batch(
            entity_type=entity_type,
            action_type=action_type,
            items=[data],
        )

    def persist_batch(
        self,
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        items: Iterable[Auditable],
    ) -> None:
        auditable_items = list(items)
        if not auditable_items:
            return

        context = audit_context.get_audit_context()
        audit_logs = [
            self._build_audit_log(
                entity_type=entity_type,
                action_type=action_type,
                data=data,
                context=context,
            )
            for data in auditable_items
        ]
        self._session.add_all(audit_logs)

    @staticmethod
    def _build_audit_log(
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        data: Auditable,
        context: audit_context.AuditContext,
    ) -> models.AuditLog:
        state_after = None if action_type is AuditActionType.DELETE else data.get_state_after()
        return models.AuditLog(
            entity_type=entity_type,
            action_type=action_type,
            item_id=data.get_item_id(),
            entity_id=data.get_entity_id(),
            entity_name=data.get_entity_name(),
            performed_by=context.performed_by,
            performed_by_name=context.performed_by_name,
            state_after=state_after,
            trace_id=context.trace_id,
        )
