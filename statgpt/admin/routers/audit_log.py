from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminAuditLogService
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

router = APIRouter(
    prefix="/audit-logs", tags=["audit-logs"], dependencies=[Depends(require_jwt_auth)]
)


@router.get("")
async def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    entity_type: str | None = None,
    action_type: str | None = None,
    entity_id: str | None = None,
    performed_by: str | None = None,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.AuditLog]:
    service = AdminAuditLogService(session)
    items = await service.get_logs(
        limit=limit,
        offset=offset,
        entity_type=entity_type,
        action_type=action_type,
        entity_id=entity_id,
        performed_by=performed_by,
    )
    total = await service.get_count(
        entity_type=entity_type,
        action_type=action_type,
        entity_id=entity_id,
        performed_by=performed_by,
    )
    return schemas.ListResponse[schemas.AuditLog](
        data=items,
        limit=limit,
        offset=offset,
        count=len(items),
        total=total,
    )


@router.get("/{item_id}")
async def get_audit_log_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.AuditLog:
    service = AdminAuditLogService(session)
    try:
        return await service.get_by_id(item_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
