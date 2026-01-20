from typing import Annotated
from uuid import uuid4

from fastmcp.dependencies import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.mcp.utils.dataset_formatter import DetailedDatasetFormatter
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base.datasource import DataSourceHandler
from statgpt.common.data.sdmx.common.config import UrnReference
from statgpt.common.data.sdmx.common.dimension import SdmxDimension
from statgpt.common.data.sdmx.common.urn import UrnParser
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.models.database import get_session_contex_manager
from statgpt.common.schemas.data_source import DataSource
from statgpt.common.schemas.enums import LocaleEnum
from statgpt.common.services.data_source import DataSourceService, DataSourceTypeService


def get_auth_context() -> AuthContext:
    return SystemUserAuthContext()


async def get_data_sources(
    data_source_id: int | None = None,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
) -> list[DataSource]:
    """Retrieve list of data sources with optional filtering and pagination."""
    data_source_service = DataSourceService(session)
    data_source_ids = [data_source_id] if data_source_id is not None else None
    data_sources = await data_source_service.get_data_sources_schemas(
        limit=None, offset=0, ids=data_source_ids
    )
    return data_sources


async def create_data_source_handler(
    data_source_id: int,
    session: AsyncSession,
) -> DataSourceHandler | None:
    """Create and return a data source handler for the given data source ID."""
    data_sources = await get_data_sources(data_source_id=data_source_id, session=session)

    if len(data_sources) == 0:
        return None
    data_source = data_sources[0]

    handler_cls = await DataSourceTypeService.get_data_source_handler_class(
        data_source.type  # type: ignore[arg-type]
    )
    handler = handler_cls(handler_cls.parse_config(data_source.details))
    return handler


async def get_datasets(
    data_source_id: int,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
    auth_context: AuthContext = Depends(get_auth_context),
) -> list[dict] | None:
    """Retrieve all datasets for a given data source."""
    handler = await create_data_source_handler(data_source_id=data_source_id, session=session)
    if handler is None:
        return None
    datasets = await handler.list_datasets(auth_context=auth_context)
    datasets_dict = []
    for dataset in datasets:
        datasets_dict.append(
            {
                'name': dataset.name,
                'description': dataset.description,
                'urn': dataset.details.urn.short_urn(),  # type: ignore[attr-defined]
            }
        )
    return datasets_dict


async def get_dataset_dimensions_and_attributes(
    data_source_id: int,
    urn: str,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
    auth_context: AuthContext = Depends(get_auth_context),
) -> tuple[list[SdmxDimension], list[Sdmx21Attribute]]:
    """Retrieve dimensions and attributes for a dataset by URN."""
    handler = await create_data_source_handler(data_source_id=data_source_id, session=session)
    if handler is None:
        return [], []
    dimensions, attributes = await handler.get_dimensions_and_attributes(urn, auth_context)
    return list(dimensions), list(attributes)  # type: ignore[arg-type]


async def validate_dataset_config(
    data_source_id: int,
    dataset_config: Annotated[dict, "Dataset config dict, must have 'details' key"],
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
    auth_context: AuthContext = Depends(get_auth_context),
) -> str:
    """Validate dataset configuration against its structure."""
    handler = await create_data_source_handler(data_source_id=data_source_id, session=session)
    if handler is None:
        return "DATA_SOURCE_HANDLER_NOT_FOUND"

    parsed_urn = UrnParser.create_default().parse(dataset_config['details']['urn'])  # type: ignore[attr-defined]
    dataset_config['details']['urn'] = UrnReference.model_validate(parsed_urn, from_attributes=True)

    dataset_config_parsed = handler.parse_data_set_config(dataset_config['details'])
    await handler.validate_dataset_config(dataset_config_parsed, auth_context=auth_context)
    return "VALIDATION_PASSED"


def generate_id() -> str:
    """Generate a random UUID string."""
    return str(uuid4())


async def get_dataset_structure(
    data_source_id: int,
    urn: str,
    session: AsyncSession = Depends(get_session_contex_manager),  # type: ignore[arg-type]
    auth_context: AuthContext = Depends(get_auth_context),
) -> str | None:
    """Get formatted dataset structure as a string."""
    dimensions, attributes = await get_dataset_dimensions_and_attributes(
        data_source_id=data_source_id, urn=urn, session=session, auth_context=auth_context
    )
    if len(dimensions) == 0 and len(attributes) == 0:
        return "No dimensions or attributes found for the dataset, check the URN and try again."
    formatter = DetailedDatasetFormatter(
        include_name=True,
        list_level=0,
        add_source_id=True,
        locale=LocaleEnum.EN,
    )
    response = await formatter.format(
        name=urn,
        dimensions=dimensions,
        attributes=attributes,
    )
    return response
