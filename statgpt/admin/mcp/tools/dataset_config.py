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
from statgpt.common.utils import read_yaml

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
) -> dict:
    """
    Retrieve all datasets available for the specified data source. Note: the list may be large.
    """
    dataset_service = DataSetService(session)
    datasets = await dataset_service.load_available_datasets(
        source_id=data_source_id, auth_context=SystemUserAuthContext()
    )
    return {
        'count': len(datasets),
        'datasets': [
            mcp_schemas.DataSetPreview(urn=ds.id_in_source, title=ds.title) for ds in datasets
        ],
    }


@mcp_tools.tool
async def get_dataset_config_details_schema(
    data_source_id: int,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> dict:
    """
    Retrieve schema for "details" field used in dataset configurations for a specific data source.
    """
    dataset_service = DataSetService(session)
    schema = await dataset_service.get_dataset_config_schema(source_id=data_source_id)
    return schema


@mcp_tools.tool
def generate_id() -> str:
    """Generate a random UUID string."""
    return str(uuid4())


@mcp_tools.tool
async def get_sdmx_dataset_structure(
    data_source_id: Annotated[int, "The ID of the data source containing the SDMX dataset"],
    agency_id: Annotated[
        str, "The agency ID component of the SDMX dataflow URN, e.g., 'ESTAT', 'IMF'"
    ],
    resource_id: Annotated[str, "The resource ID (dataflow identifier) component of the URN"],
    version: Annotated[
        str,
        "The version component of the URN. Use 'latest' for the most recent version,"
        " or a specific version like '1.0' or '1.0.0+' for version ranges.",
    ],
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> dict:
    """Retrieve the structure of an SDMX dataset including its dimensions and attributes.

    This tool fetches metadata about the dataset's structure from the SDMX registry,
    resolving dynamic URN values (e.g., 'latest', wildcards) to actual versions.
    """
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


@mcp_tools.tool
async def validate_dataset_config(
    data_source_id: int,
    dataset_configs_path: Annotated[str, "Absolute path to datasets configs file"],
    dataset_uuid: Annotated[
        str, "Dataset config UUID using which you can retrieve config from datasets files"
    ],
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> DataSetValidationResult:
    """Validate dataset configuration against its structure."""

    datasets_configs = read_yaml(dataset_configs_path).get('dataSets')
    if datasets_configs is None:
        raise ValueError("Datasets configs file does not contain 'dataSets' key")
    dataset_config = next(
        (ds_conf for ds_conf in datasets_configs if ds_conf["id_"] == dataset_uuid), None
    )
    if dataset_config is None:
        raise ValueError(
            f"Dataset config with UUID {dataset_uuid} not found in datasets configs file"
        )

    dataset_service = DataSetService(session)
    res = await dataset_service.validate_config(
        source_id=data_source_id,
        config=dataset_config['details'],
        auth_context=SystemUserAuthContext(),
    )
    return res
