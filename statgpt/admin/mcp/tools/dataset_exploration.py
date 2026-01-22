from typing import Annotated
from uuid import uuid4

from fastmcp.dependencies import Depends
from fastmcp.server.providers import LocalProvider
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.mcp import schemas as mcp_schemas
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.common.data.base import DataSetValidationResult
from statgpt.common.models.database import get_session_contex_manager
from statgpt.common.services.data_source import DataSourceService

mcp_tools = LocalProvider()


@mcp_tools.tool
async def get_data_sources(
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> list[mcp_schemas.DataSource]:
    """Retrieve a list of data sources"""
    data_source_service = DataSourceService(session)
    data_sources = await data_source_service.get_data_sources_schemas(
        limit=None, offset=0, ids=None
    )
    return [
        mcp_schemas.DataSource(
            id=ds.id, title=ds.title, description=ds.description, type=ds.type.name
        )
        for ds in data_sources
    ]


@mcp_tools.tool
async def get_available_datasets(
    data_source_id: int,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> list[mcp_schemas.DataSetPreview]:
    """Retrieve all available datasets in a given data source."""

    dataset_service = DataSetService(session)
    datasets = await dataset_service.load_available_datasets(
        source_id=data_source_id, auth_context=SystemUserAuthContext()
    )
    return [
        mcp_schemas.DataSetPreview(
            id_in_source=ds.id_in_source, title=ds.title, description=ds.description
        )
        for ds in datasets
    ]


@mcp_tools.tool
async def get_dataset_details_schema(
    data_source_id: int,
    session: AsyncSession = Depends(get_session_contex_manager),
) -> dict:
    """Retrieve the configuration schema of the `details` field for datasets in a given data source."""
    dataset_service = DataSetService(session)
    schema = await dataset_service.get_dataset_config_schema(source_id=data_source_id)
    return schema


@mcp_tools.tool
async def validate_dataset_config(
    data_source_id: int,
    dataset_config: Annotated[dict, "Dataset config dict, must have 'details' key"],
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> DataSetValidationResult:
    """Validate dataset configuration against its structure."""
    dataset_service = DataSetService(session)
    res = await dataset_service.validate_config(
        source_id=data_source_id,
        config=dataset_config['details'],
        auth_context=SystemUserAuthContext(),
    )
    return res


@mcp_tools.tool
def generate_id() -> str:
    """Generate a random UUID string."""
    return str(uuid4())


@mcp_tools.tool
async def get_sdmx_dataset_structure(
    data_source_id: int,
    agency_id: str,
    resource_id: str,
    version: str,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> dict:
    """TODO: add docstring"""
    # NOTE: While other tools are generic, this one is SDMX-specific.

    dataset_service = DataSetService(session)
    config = {
        'urn': {
            'agency_id': agency_id,
            'resource_id': resource_id,
            'version': version,
        }
    }
    response = await dataset_service.get_dataset_structure(
        source_id=data_source_id, config=config, auth_context=SystemUserAuthContext()
    )
    return response
