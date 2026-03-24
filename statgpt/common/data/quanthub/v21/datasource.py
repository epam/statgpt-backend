import uuid

import httpx
from httpx import HTTPStatusError

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig, QuanthubSdmxDataSourceConfig
from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from statgpt.common.data.sdmx import Sdmx21DataSourceHandler
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dataset import InvalidConfigurationError, SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import Urn
from statgpt.common.schemas.dataset import Status
from statgpt.common.utils import AsyncLoadingCache
from statgpt.common.utils.timer import debug_timer

from .qh_sdmx_client import AsyncQuanthubClient


# (DataSourceHandler[SdmxDataSourceConfig, InMemorySdmx21DataSet | SdmxOfflineDataSet, QuanthubDataSetConfig],ABC)
# todo: add generic typing with QuanthubInMemorySdmx21DataSet
class QuanthubSdmx21DataSourceHandler(Sdmx21DataSourceHandler):

    _dataset_cache: AsyncLoadingCache[QuanthubSdmx21DataSet | SdmxOfflineDataSet] = (
        AsyncLoadingCache()
    )

    def __init__(self, config: QuanthubSdmxDataSourceConfig):
        super().__init__(config)
        self._config: QuanthubSdmxDataSourceConfig = config  # for type hinting

    async def create_sdmx_client(self, auth_context: AuthContext) -> AsyncQuanthubClient:
        rate_limiter = await SdmxRateLimiterFactory.get(
            self._config.get_id(), self._config.rate_limits
        )
        return AsyncQuanthubClient.from_config(self._config, auth_context, rate_limiter)

    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        if auth_context.is_system:
            return True
        elif not self._config.auth_enabled:
            logger.debug(
                f"Skipping availability check for dataset {config['urn']} as auth is disabled."
            )
            return True
        else:
            try:
                conf = self.parse_data_set_config(config)
                client = await self.create_sdmx_client(auth_context)
                await client.availableconstraint(
                    agency_id=conf.urn.agency_id,
                    resource_id=conf.urn.resource_id,
                    version=conf.urn.version,
                    params={"references": "none"},
                    use_cache=False,
                )
                return True
            except HTTPStatusError as e:
                # availability endpoint returns 400 with NotFound instead of 403
                # treat 400 as Forbidden as well
                if e.response.status_code in [403, 400]:
                    # 403 means user doesn't have access to dataset
                    return False
                else:
                    raise e

    async def _get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> QuanthubSdmx21DataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

        logger.info(f"Loading dataset urn={dataset_config.urn!r}.")

        sdmx_client = await self.create_sdmx_client(auth_context)

        try:
            urn = Urn(
                agency_id=dataset_config.urn.agency_id,
                resource_id=dataset_config.urn.resource_id,
                version=dataset_config.urn.version,
            )
            dataflow_loader = DataflowLoader(sdmx_client)
            urn, structure_message = await dataflow_loader.load_structure_message(urn, mode="full")
        except Exception as e:
            if allow_offline:
                msg = f"Failed to load the dataflow or its associated structures. urn={dataset_config.urn!r}"
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e

        try:
            dimensions_creator = DimensionsCreator(
                structure_message, urn, self._config.locale, dataset_config.get_dimension_aliases()
            )
            dimensions = await dimensions_creator.create_dimensions()
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dimensions from the loaded structure message."
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e

        try:
            attributes_creator = Sdmx21AttributesCreator(
                structure_message, urn, self._config.locale
            )
            attributes = await attributes_creator.create_attributes()
        except Exception as e:
            if allow_offline:
                msg = "Failed to create attributes from the loaded structure message."
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e

        try:
            attribute_values = await sdmx_client.dataset_level_attributes(
                agency_id=urn.agency_id, resource_id=urn.resource_id, version=urn.version
            )
        except Exception:
            logger.exception(f"Failed to load dataset-level attributes for the dataflow({urn}).")
            attribute_values = {}

        try:
            annotations = await sdmx_client.dynamic_dataflow_annotations(
                agency_id=urn.agency_id, resource_id=urn.resource_id, version=urn.version
            )
        except httpx.RequestError as e:
            logger.exception(
                f"Failed to load annotations for the dataflow({urn})."
                f"\nRequest: {e.request.method} {e.request.url}"
                + (f"\nContent: {e.request.content!r}" if e.request.content else "")
            )
            annotations = []
        except Exception:
            logger.exception(f"Failed to load annotations for the dataflow({urn}).")
            annotations = []

        try:
            dataflow = structure_message.dataflow[urn]
            res = QuanthubSdmx21DataSet(
                entity_id=entity_id,
                title=title,
                config=dataset_config,
                handler=self,
                dataflow=dataflow,
                locale=self._config.locale,
                dimensions=dimensions,
                attributes=attributes,
                attribute_values=attribute_values,
                annotations=annotations,
            )
        except InvalidConfigurationError as e:
            if allow_offline:
                msg = f"Invalid dataset(urn={dataset_config.urn!r}) configuration: {e}"
                logger.warning(msg)
                status = Status(status='invalid_config', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dataset class."
                logger.exception(f"{msg}. See exception details below.")
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e

        return res

    async def get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> QuanthubSdmx21DataSet | SdmxOfflineDataSet:
        with debug_timer(f"QuanthubSdmx21DataSourceHandler.get_dataset: {title}"):
            if allow_cached and not self._config.auth_enabled:
                dataset_config = self.parse_data_set_config(config)
                return await self._dataset_cache.get(
                    key=str(entity_id),
                    loader=lambda: self._get_dataset(
                        entity_id, title, config, auth_context, allow_offline=allow_offline
                    ),
                    # NOTE: OfflineDataset may end up in the cache when allow_offline=True
                    # and loading fails. The validator rejects it on the next access,
                    # triggering a fresh load attempt (in case the upstream recovered).
                    validator=lambda ds: (
                        not isinstance(ds, SdmxOfflineDataSet) and ds.config == dataset_config
                    ),
                )
            return await self._get_dataset(
                entity_id, title, config, auth_context, allow_offline=allow_offline
            )

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="QH_SDMX21",
            name="Quanthub SDMX 2.1 Registry",
            description="Quanthub SDMX 2.1 Registry data source",
        )

    @staticmethod
    def parse_config(d: dict) -> QuanthubSdmxDataSourceConfig:
        return QuanthubSdmxDataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> QuanthubDataSetConfig:
        return QuanthubDataSetConfig.model_validate(d)

    @staticmethod
    def get_data_set_config_schema() -> dict:
        return QuanthubDataSetConfig.model_json_schema(by_alias=True)
