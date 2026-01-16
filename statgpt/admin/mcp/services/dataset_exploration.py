from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.mcp.utils.dataset_formatter import DetailedDatasetFormatter
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base.datasource import DataSourceHandler
from statgpt.common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from statgpt.common.data.sdmx.common.config import SdmxDataSourceConfig
from statgpt.common.data.sdmx.common.dimension import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import Urn
from statgpt.common.models.database import get_session_contex_manager
from statgpt.common.schemas.enums import LocaleEnum
from statgpt.common.services.data_source import DataSourceService, DataSourceTypeService

auth_context = SystemUserAuthContext()


async def get_data_sources_list(
    session: AsyncSession,
    limit: int | None = None,
    offset: int = 0,
    data_source_id: int | None = None,
):
    """Retrieve list of data sources with optional filtering and pagination."""
    data_source_service = DataSourceService(session)
    ids = [data_source_id] if data_source_id is not None else None
    data_sources = await data_source_service.get_data_sources_schemas(
        limit=limit, offset=offset, ids=ids
    )
    return data_sources


async def create_data_source_handler(data_source_id: int) -> DataSourceHandler | None:
    """Create and return a data source handler for the given data source ID."""
    async with get_session_contex_manager() as session:
        data_sources = await get_data_sources_list(session=session, data_source_id=data_source_id)

    if len(data_sources) == 0:
        return None
    data_source = data_sources[0]

    handler_cls = await DataSourceTypeService.get_data_source_handler_class(
        data_source.type  # pyright: ignore[reportArgumentType]
    )
    handler = handler_cls(handler_cls.parse_config(data_source.details))
    return handler


async def get_datasets(data_source_id: int):
    """Retrieve all datasets for a given data source."""
    handler = await create_data_source_handler(data_source_id)
    if handler is None:
        return None
    datasets = await handler.list_datasets(auth_context=auth_context)
    return datasets


async def create_sdmx_client(handler_config: SdmxDataSourceConfig, auth_context: AuthContext):
    """Create an SDMX client with rate limiting from handler configuration."""
    rate_limiter = await SdmxRateLimiterFactory.get(
        handler_config.get_id(), handler_config.rate_limits
    )
    return AsyncQuanthubClient.from_config(
        handler_config, auth_context, rate_limiter  # type: ignore[arg-type]
    )


async def get_dataset_dimensions_and_attributes(
    data_source_id: int, urn: str
) -> tuple[list[SdmxDimension], list[Sdmx21Attribute]] | tuple[None, None]:
    """Retrieve dimensions and attributes for a dataset by URN."""
    handler = await create_data_source_handler(data_source_id=data_source_id)
    if handler is None:
        return None, None
    sdmx_client = await create_sdmx_client(handler._config, auth_context)

    parsed_urn = handler._urn_parser.parse(urn)  # type: ignore[attr-defined]
    urn_obj = Urn(
        agency_id=parsed_urn.agency_id,
        resource_id=parsed_urn.resource_id,
        version=parsed_urn.version if parsed_urn.version else "latest",
    )

    dataflow_loader = DataflowLoader(sdmx_client)
    structure_message = await dataflow_loader.load_structure_message(urn_obj, mode="full")

    dims_creator = DimensionsCreator(structure_message, urn_obj, handler._config.locale, {})
    dimensions = await dims_creator.create_dimensions()

    attributes_creator = Sdmx21AttributesCreator(structure_message, urn_obj, handler._config.locale)
    attributes = await attributes_creator.create_attributes()
    return dimensions, attributes


from typing import Annotated


async def validate_dataset_config(
    data_source_id: int,
    dataset_config: Annotated[dict, "Dataset config dict, must have 'details' key"],
) -> str:
    """Validate dataset configuration against its structure."""
    handler = await create_data_source_handler(data_source_id=data_source_id)
    if handler is None:
        return "DATA_SOURCE_HANDLER_NOT_FOUND"
    dimensions, _ = await get_dataset_dimensions_and_attributes(
        data_source_id=data_source_id, urn=dataset_config['details']['urn']
    )
    if dimensions is None:
        return "DIMENSIONS_NOT_FOUND"
    dataset_config_parsed = handler.parse_data_set_config(dataset_config['details'])
    handler.validate_dataset_config(dataset_config_parsed, dimensions)
    return "VALIDATION_PASSED"


def generate_id() -> str:
    """Generate a random UUID string."""
    return str(uuid4())


async def get_dataset_structure(data_source_id: int, urn: str) -> str | None:
    """Get formatted dataset structure as a string."""
    dimensions, attributes = await get_dataset_dimensions_and_attributes(
        data_source_id=data_source_id, urn=urn
    )
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
