from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

import common.models as models
import common.schemas as schemas
from admin_portal.auth.auth_context import SystemUserAuthContext
from admin_portal.auth.user import User, require_jwt_auth
from admin_portal.services import AdminPortalDataSetService as DataSetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
async def get_datasets(
    limit: int = 100,
    offset: int = 0,
    data_source_id: int | None = None,
    channel_id: int | None = None,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.ListResponse[schemas.DataSet]:
    """Returns a list of added datasets"""

    service = DataSetService(session)
    datasets = await service.get_datasets_schemas(
        limit=limit,
        offset=offset,
        source_id=data_source_id,
        channel_id=channel_id,
        auth_context=SystemUserAuthContext(),
        allow_offline=True,
    )
    datasets_count = await service.get_datasets_count(
        source_id=data_source_id, channel_id=channel_id
    )

    return schemas.ListResponse[schemas.DataSet](
        data=datasets,
        limit=limit,
        offset=offset,
        count=len(datasets),
        total=datasets_count,
    )


@router.post("")
async def register_dataset(
    data: schemas.DataSetBase,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSet:
    """Register a dataset in the system"""

    return await DataSetService(session).create_dataset(data, auth_context=SystemUserAuthContext())


@router.get("/{item_id}")
async def get_dataset_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSet:
    return await DataSetService(session).get_schema_by_id(
        item_id, auth_context=SystemUserAuthContext(), allow_offline=True
    )


@router.post("/{item_id}")
async def update_dataset(
    item_id: int,
    data: schemas.DataSetUpdate,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSet:
    return await DataSetService(session).update(item_id, data, auth_context=SystemUserAuthContext())


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> None:
    """Delete a dataset by its ID. This is only allowed for datasets that are not used in any channel."""
    await DataSetService(session).delete(item_id)
