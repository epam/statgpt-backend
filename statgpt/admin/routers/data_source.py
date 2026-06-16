from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.admin.services import AdminPortalDataSourceDeletionService as DataSourceDeletionService
from statgpt.admin.services import AdminPortalDataSourceService as DataSourceService
from statgpt.common.services import DataSourceTypeService
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

router = APIRouter(
    prefix="/data-sources", tags=["data-sources"], dependencies=[Depends(require_jwt_auth)]
)


@router.get("/types")
async def get_data_source_types(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
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
    _=Depends(cancel_on_disconnect),
):
    """Returns the JSON schema for a specific data source type."""

    service = DataSourceTypeService(session)
    return await service.get_schema_config(item_id)


@router.get("")
async def get_data_sources(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
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
) -> schemas.DataSource:
    """Create a new data source"""

    return await DataSourceService(session).create_data_source(data)


@router.get("/{item_id}")
async def get_data_source_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.DataSource:
    return await DataSourceService(session).get_schema_by_id(item_id)


@router.get("/{item_id}/providers")
async def get_providers(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.Provider]:
    """Returns a list of providers (maintainer agencies) exposed by the data source."""

    providers = await DataSetService(session).load_available_providers(
        source_id=item_id, auth_context=SystemUserAuthContext()
    )
    providers_count = len(providers)

    return schemas.ListResponse[schemas.Provider](
        data=providers,
        limit=providers_count,
        offset=0,
        count=providers_count,
        total=providers_count,
    )


@router.get("/{item_id}/available-datasets")
async def get_available_datasets(
    item_id: int,
    provider: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_.\-]+$"),
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.DataSetDescriptor]:
    """Returns a list of datasets that exists in the data source and can be added to the system.

    The optional `provider` query parameter restricts the listing to datasets whose
    maintainer agency matches the given id (e.g. `provider=IMF.RES`).

    NOTES:
        * These list does NOT exclude datasets that are already added to the system.
        * The returned datasets contain only some pre-configurations and require manual review
          and updating before being added to the system.
    """

    datasets = await DataSetService(session).load_available_datasets(
        source_id=item_id,
        auth_context=SystemUserAuthContext(),
        provider=provider,
    )
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
) -> schemas.DataSource:
    return await DataSourceService(session).update(item_id, data)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> None:
    """Delete data source by id"""

    await DataSourceDeletionService(session).delete(item_id)
