import datetime
import io
import zipfile
from csv import DictWriter

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminAuditLogService
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

router = APIRouter(
    prefix="/audit-logs", tags=["audit-logs"], dependencies=[Depends(require_jwt_auth)]
)


@router.get("/enum-values")
async def get_audit_log_enum_values() -> dict[str, list[str]]:
    return {
        "entity_types": [entity_type.value for entity_type in schemas.AuditEntityType],
        "action_types": [action_type.value for action_type in schemas.AuditActionType],
    }


def _validate_created_at_range(
    created_at_from: datetime.datetime | None,
    created_at_to: datetime.datetime | None,
) -> None:
    if created_at_from and created_at_to and created_at_from > created_at_to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="created_at_from must be less than or equal to created_at_to",
        )


@router.get("")
async def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    entity_type: schemas.AuditEntityType | None = None,
    action_type: schemas.AuditActionType | None = None,
    item_id: int | None = None,
    entity_id: str | None = None,
    entity_name: str | None = None,
    performed_by: str | None = None,
    performed_by_name: str | None = None,
    trace_id: str | None = None,
    created_at_from: datetime.datetime | None = None,
    created_at_to: datetime.datetime | None = None,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.AuditLogListItem]:
    _validate_created_at_range(created_at_from, created_at_to)

    service = AdminAuditLogService(session)
    items = await service.get_logs(
        limit=limit,
        offset=offset,
        entity_type=entity_type,
        action_type=action_type,
        item_id=item_id,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=performed_by,
        performed_by_name=performed_by_name,
        trace_id=trace_id,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )
    total = await service.get_count(
        entity_type=entity_type,
        action_type=action_type,
        item_id=item_id,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=performed_by,
        performed_by_name=performed_by_name,
        trace_id=trace_id,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )
    return schemas.ListResponse[schemas.AuditLogListItem](
        data=items,
        limit=limit,
        offset=offset,
        count=len(items),
        total=total,
    )


@router.get("/export")
async def export_audit_logs(
    entity_type: schemas.AuditEntityType | None = None,
    action_type: schemas.AuditActionType | None = None,
    item_id: int | None = None,
    entity_id: str | None = None,
    entity_name: str | None = None,
    performed_by: str | None = None,
    performed_by_name: str | None = None,
    trace_id: str | None = None,
    created_at_from: datetime.datetime | None = None,
    created_at_to: datetime.datetime | None = None,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> Response:
    _validate_created_at_range(created_at_from, created_at_to)

    service = AdminAuditLogService(session)
    items = await service.get_logs(
        entity_type=entity_type,
        action_type=action_type,
        item_id=item_id,
        entity_id=entity_id,
        entity_name=entity_name,
        performed_by=performed_by,
        performed_by_name=performed_by_name,
        trace_id=trace_id,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )

    field_names = list(schemas.AuditLogListItem.model_fields.keys())
    csv_buffer = io.StringIO()
    writer = DictWriter(csv_buffer, fieldnames=field_names)
    writer.writeheader()
    for item in items:
        writer.writerow(item.model_dump(mode="json"))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audit_logs.csv", csv_buffer.getvalue())

    zip_data = zip_buffer.getvalue()
    file_name = f"audit_logs_{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H-%M-%SZ}.zip"
    return Response(
        content=zip_data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Length": str(len(zip_data)),
        },
    )


@router.get("/{item_id}")
async def get_audit_log_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.AuditLogDetails:
    service = AdminAuditLogService(session)
    try:
        return await service.get_by_id(item_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
