import uuid
from uuid import uuid4

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base.query import DataSetAvailabilityQuery
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from statgpt.common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from statgpt.common.data.sdmx.common.config import SdmxDataSourceConfig
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import Urn
from statgpt.common.services.data_source import DataSourceService, DataSourceTypeService

auth_context = SystemUserAuthContext()


async def load_series_df(
    dataset: QuanthubSdmx21DataSet, indicator_ids: list[str], auth_context
) -> pd.DataFrame:
    query = DataSetAvailabilityQuery(dimensions_queries_dict={})
    avail_query_resp = await dataset.availability_query(query=query, auth_context=auth_context)

    avail_values = {
        dim: avail_query_resp.dimensions_queries_dict[dim].values for dim in indicator_ids
    }

    dim_2_avail_values_cnt_sorted = sorted(
        ((k, len(v)) for k, v in avail_values.items()), key=lambda x: x[1]
    )
    order = [x[0] for x in dim_2_avail_values_cnt_sorted]

    series, _ = await dataset._get_available_series(
        cur_query={},
        cur_dim=order[0],
        cur_dim_avail_values=avail_values[order[0]],
        other_dims_to_fill=order[1:],
        queries_count=1,
        auth_context=auth_context,
    )
    series_df = pd.DataFrame(series)
    return series_df


async def get_dataset_combinations(
    data_source_id: int,
    urn: str,
    n: int,
) -> pd.DataFrame | None:
    # Create handler
    handler = await create_data_source_handler(data_source_id=data_source_id)
    if handler is None:
        return None
    sdmx_client = await create_sdmx_client(handler._config, auth_context)

    # Parse URN
    parsed_urn = handler._urn_parser.parse(urn)
    urn_obj = Urn(
        agency_id=parsed_urn.agency_id,
        resource_id=parsed_urn.resource_id,
        version=parsed_urn.version if parsed_urn.version else "latest",
    )

    # Load structure
    dataflow_loader = DataflowLoader(sdmx_client)
    structure_message = await dataflow_loader.load_structure_message(urn_obj, mode="full")
    dataflow = structure_message.dataflow[urn_obj]
    # Create dimensions and attributes
    dimensions_creator = DimensionsCreator(structure_message, urn_obj, handler._config.locale, {})
    dimensions = await dimensions_creator.create_dimensions()

    attributes_creator = Sdmx21AttributesCreator(structure_message, urn_obj, handler._config.locale)
    attributes = await attributes_creator.create_attributes()

    # Get attribute values and annotations
    attribute_values = await sdmx_client.dataset_level_attributes(
        agency_id=urn_obj.agency_id, resource_id=urn_obj.resource_id, version=urn_obj.version
    )
    annotations = await sdmx_client.dynamic_dataflow_annotations(
        agency_id=urn_obj.agency_id, resource_id=urn_obj.resource_id, version=urn_obj.version
    )

    # Create deafult dataset config - convert SdmxDataSetConfigTemplate to QuanthubDataSetConfig
    config_template = handler._create_config_for(dataflow, short_urn=urn_obj.short_urn())
    config = QuanthubDataSetConfig(**config_template.model_dump())

    dataset = QuanthubSdmx21DataSet(
        entity_id=uuid.uuid4(),
        title="",
        config=config,
        handler=handler,  # type: ignore
        dataflow=dataflow,
        locale=handler._config.locale,
        dimensions=dimensions,
        attributes=attributes,
        attribute_values=attribute_values,
        annotations=annotations,
    )

    # Load indicator combinations
    indicator_ids = [x.entity_id for x in dataset.dimensions()]
    if "TIME_PERIOD" in indicator_ids:
        indicator_ids.pop(indicator_ids.index("TIME_PERIOD"))
    df = await load_series_df(
        dataset=dataset, auth_context=auth_context, indicator_ids=indicator_ids
    )
    n = min(n, len(df))
    sampled_df = df.sample(n) if n > 0 else df.head()
    mapping = dataset._get_dim_values_id_2_name_mapping()

    result_df = sampled_df.copy()
    for dim_id in sampled_df.columns:
        if dim_id in mapping:
            result_df[dim_id] = sampled_df[dim_id].map(lambda x: mapping[dim_id].get(x, x))

    return result_df


async def get_data_sources_list(
    session: AsyncSession, limit=100, offset=0, data_source_id: int | None = None
):
    data_source_service = DataSourceService(session)
    ids = [data_source_id] if data_source_id is not None else None
    data_sources = await data_source_service.get_data_sources_schemas(
        limit=limit, offset=offset, ids=ids
    )
    return data_sources


async def create_data_source_handler(data_source_id: int) -> Sdmx21DataSourceHandler | None:
    data_sources = await get_data_sources(data_source_id=data_source_id)
    if len(data_sources) == 0:
        return None
    data_source = data_sources[0]

    handler_cls = await DataSourceTypeService.get_data_source_handler_class(
        data_source.type  # pyright: ignore[reportArgumentType]
    )
    handler = handler_cls(handler_cls.parse_config(data_source.details))
    return handler  # type: ignore[return-value]


async def get_datasets(data_source_id: int):
    handler = await create_data_source_handler(data_source_id)
    if handler is None:
        return None
    datasets = await handler.list_datasets(auth_context=auth_context)
    return datasets


async def create_sdmx_client(handler_config: SdmxDataSourceConfig, auth_context: AuthContext):
    rate_limiter = await SdmxRateLimiterFactory.get(
        handler_config.get_id(), handler_config.rate_limits
    )
    return AsyncQuanthubClient.from_config(
        handler_config, auth_context, rate_limiter  # type: ignore[arg-type]
    )


async def get_dataset_dimensions(data_source_id: int, urn: str):
    handler = await create_data_source_handler(data_source_id=data_source_id)
    if handler is None:
        return None
    sdmx_client = await create_sdmx_client(handler._config, auth_context)

    parsed_urn = handler._urn_parser.parse(urn)
    urn_obj = Urn(  # pyright: ignore[reportAssignmentType]
        agency_id=parsed_urn.agency_id,
        resource_id=parsed_urn.resource_id,
        version=parsed_urn.version if parsed_urn.version else "latest",
    )

    dataflow_loader = DataflowLoader(sdmx_client)
    structure_message = await dataflow_loader.load_structure_message(urn_obj, mode="full")

    dims_creator = DimensionsCreator(structure_message, urn_obj, handler._config.locale, {})
    dimensions = await dims_creator.create_dimensions()
    return dimensions


async def validate_dataset_config(data_source_id: int, config_dict: dict) -> bool | None:
    handler = await create_data_source_handler(data_source_id=data_source_id)
    if handler is None:
        return None
    dimensions = await get_dataset_dimensions(
        data_source_id=data_source_id, urn=config_dict['details']['urn']
    )
    dataset_config = handler.parse_data_set_config(config_dict['details'])
    handler._validate_dataset_config(dataset_config, dimensions)
    return True


def generate_id() -> str:
    """Create random UUID..."""
    return str(uuid4())
