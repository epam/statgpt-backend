from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.admin.services.exceptions import DatasetInUseError
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[Depends(require_jwt_auth)])


@router.get("")
async def get_datasets(
    limit: int = 100,
    offset: int = 0,
    data_source_id: int | None = None,
    channel_id: int | None = None,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
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
) -> schemas.DataSet:
    """Register a dataset in the system"""

    return await DataSetService(session).create_dataset(data, auth_context=SystemUserAuthContext())


@router.get("/{item_id}")
async def get_dataset_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.DataSet:
    return await DataSetService(session).get_schema_by_id(
        item_id, auth_context=SystemUserAuthContext(), allow_offline=True
    )


@router.post("/{item_id}")
async def update_dataset(
    item_id: int,
    data: schemas.DataSetUpdateRequest,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.DataSetUpdateResponse:
    """Update a dataset.

    Returns the updated dataset along with the results of propagating config
    changes to all channel datasets that reference this dataset.
    """
    return await DataSetService(session).update(item_id, data, auth_context=SystemUserAuthContext())


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> None:
    """Delete a dataset by its ID. This is only allowed for datasets that are not used in any channel."""
    try:
        await DataSetService(session).delete(item_id)
    except DatasetInUseError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete dataset because it is still used in channels: {e.usage_summary}.",
        ) from e
