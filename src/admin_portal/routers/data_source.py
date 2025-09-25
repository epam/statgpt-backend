from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

import common.models as models
import common.schemas as schemas
from admin_portal.auth.user import User, require_jwt_auth
from admin_portal.services import AdminPortalDataSetService as DataSetService
from admin_portal.services import AdminPortalDataSourceService as DataSourceService
from common.services import DataSourceTypeService

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.get("/types")
async def get_data_source_types(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.ListResponse[schemas.DataSourceType]:
    service = DataSourceTypeService(session)
    data_source_types = await service.get_data_source_types(limit=limit, offset=offset)
    data_source_types_count = await service.get_count()

    return schemas.ListResponse[schemas.DataSourceType](
        data=data_source_types,
        limit=limit,
        offset=offset,
        count=len(data_source_types),
        total=data_source_types_count,
    )


@router.get("/types/{item_id}/config-schema")
async def get_schema_config_of_data_source_type(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
):
    """Returns the JSON schema for a specific data source type."""

    service = DataSourceTypeService(session)
    return await service.get_schema_config(item_id)


@router.get("")
async def get_data_sources(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.ListResponse[schemas.DataSource]:
    """Returns a list of data sources"""

    service = DataSourceService(session)
    data_sources = await service.get_data_sources_schemas(limit=limit, offset=offset)
    data_sources_count = await service.get_data_sources_count()

    return schemas.ListResponse[schemas.DataSource](
        data=data_sources,
        limit=limit,
        offset=offset,
        count=len(data_sources),
        total=data_sources_count,
    )


@router.post("")
async def create_data_source(
    data: schemas.DataSourceBase,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSource:
    """Create a new data source"""

    return await DataSourceService(session).create_data_source(data)


@router.get("/{item_id}")
async def get_data_source_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSource:
    return await DataSourceService(session).get_schema_by_id(item_id)


@router.get("/{item_id}/available-datasets")
async def get_available_datasets(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.ListResponse[schemas.DataSetDescriptor]:
    """Returns a list of datasets that can be loaded from the data source"""

    datasets = await DataSetService(session).load_available_datasets(source_id=item_id)
    datasets_count = len(datasets)

    return schemas.ListResponse[schemas.DataSetDescriptor](
        data=datasets,
        limit=datasets_count,
        offset=0,
        count=datasets_count,
        total=datasets_count,
    )


@router.post("/{item_id}")
async def update_data_source(
    item_id: int,
    data: schemas.DataSourceUpdate,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.DataSource:
    return await DataSourceService(session).update(item_id, data)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> None:
    """Delete data source by id"""

    await DataSourceService(session).delete(item_id)
