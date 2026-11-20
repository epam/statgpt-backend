from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.admin.audit import context as audit_context
from statgpt.admin.settings.app import APP_SETTINGS
from statgpt.common.schemas.auditable import Auditable
from statgpt.common.schemas.enums import AuditActionType, AuditEntityType, AuditScope


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _scope_enabled(scope: AuditScope) -> bool:
        return scope in APP_SETTINGS.enabled_audit_scopes

    async def persist(
        self,
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        data: Auditable,
        scope: AuditScope = AuditScope.CONFIG,
    ) -> None:
        await self.persist_batch(
            entity_type=entity_type,
            action_type=action_type,
            items=[data],
            scope=scope,
        )

    async def persist_batch(
        self,
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        items: Iterable[Auditable],
        scope: AuditScope = AuditScope.CONFIG,
    ) -> None:
        if not self._scope_enabled(scope):
            return

        auditable_items = list(items)
        if not auditable_items:
            return

        context = audit_context.get_audit_context()
        audit_logs = []
        for data in auditable_items:
            if action_type is AuditActionType.UPDATE and await self._is_noop_update(
                entity_type, scope, data
            ):
                continue
            audit_logs.append(
                self._build_audit_log(
                    entity_type=entity_type,
                    action_type=action_type,
                    scope=scope,
                    data=data,
                    context=context,
                )
            )
        self._session.add_all(audit_logs)

    async def log_event(
        self,
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        scope: AuditScope,
        item_id: int,
        entity_id: str,
        entity_name: str,
        state_after: dict | None = None,
    ) -> None:
        """Records a single audit entry from explicit fields.

        Used for events (reindexing, dataset linking) that have no dedicated
        ``Auditable`` domain object.
        """
        if not self._scope_enabled(scope):
            return

        context = audit_context.get_audit_context()
        self._session.add(
            models.AuditLog(
                entity_type=entity_type,
                action_type=action_type,
                scope=scope,
                item_id=item_id,
                entity_id=entity_id,
                entity_name=entity_name,
                performed_by=context.performed_by,
                performed_by_name=context.performed_by_name,
                state_after=state_after,
                trace_id=context.trace_id,
            )
        )

    async def _is_noop_update(
        self, entity_type: AuditEntityType, scope: AuditScope, data: Auditable
    ) -> bool:
        """True when the update leaves the persisted state unchanged.

        Skips empty-diff audit records, e.g. re-importing an unchanged channel
        config, which would otherwise call the audited ``update`` unconditionally.
        Only same-scope records are compared: verbose-scope events (reindex,
        ds_link) store a different ``state_after`` shape and would otherwise mask
        genuine config no-ops.
        """
        query = (
            select(models.AuditLog.state_after)
            .where(
                models.AuditLog.entity_type == entity_type,
                models.AuditLog.scope == scope,
                models.AuditLog.item_id == data.get_item_id(),
            )
            .order_by(models.AuditLog.id.desc())
            .limit(1)
        )
        previous_state = (await self._session.execute(query)).scalar_one_or_none()
        return previous_state is not None and previous_state == data.get_state_after()

    @staticmethod
    def _build_audit_log(
        *,
        entity_type: AuditEntityType,
        action_type: AuditActionType,
        scope: AuditScope,
        data: Auditable,
        context: audit_context.AuditContext,
    ) -> models.AuditLog:
        state_after = None if action_type is AuditActionType.DELETE else data.get_state_after()
        return models.AuditLog(
            entity_type=entity_type,
            action_type=action_type,
            scope=scope,
            item_id=data.get_item_id(),
            entity_id=data.get_entity_id(),
            entity_name=data.get_entity_name(),
            performed_by=context.performed_by,
            performed_by_name=context.performed_by_name,
            state_after=state_after,
            trace_id=context.trace_id,
        )
