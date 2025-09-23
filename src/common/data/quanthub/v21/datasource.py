import httpx
from httpx import HTTPStatusError

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import DataSourceType
from common.data.quanthub.config import QuanthubDataSetConfig, QuanthubSdmxDataSourceConfig
from common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from common.data.sdmx import Sdmx21DataSourceHandler
from common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from common.data.sdmx.v21.dataflow_loader import DataflowLoader
from common.data.sdmx.v21.dataset import SdmxOfflineDataSet
from common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from common.data.sdmx.v21.schemas import Urn
from common.settings.sdmx import quanthub_settings
from common.utils import Cache

from .qh_sdmx_client import AsyncQuanthubClient


# (DataSourceHandler[SdmxDataSourceConfig, InMemorySdmx21DataSet | SdmxOfflineDataSet, QuanthubDataSetConfig],ABC)
# todo: add generic typing with QuanthubInMemorySdmx21DataSet
class QuanthubSdmx21DataSourceHandler(Sdmx21DataSourceHandler):

    # TEMP fix:
    _dataset_cache: Cache[QuanthubSdmx21DataSet] = Cache(ttl=quanthub_settings.dataset_cache_ttl)

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
            logger.info(
                f"Skipping availability check for dataset {config['urn']} as auth is disabled."
            )
            return True
        else:
            try:
                urn = self._urn_parser.parse(config["urn"])
                client = await self.create_sdmx_client(auth_context)
                await client.availableconstraint(
                    agency_id=urn.agency_id,
                    resource_id=urn.resource_id,
                    version=urn.version if urn.version else "latest",
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

    async def get_dataset(
        self,
        entity_id: str,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> QuanthubSdmx21DataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

        if allow_cached and not self._config.auth_enabled:
            # If auth is disabled, we can cache datasets for all users
            if ds := self._dataset_cache.get(entity_id):
                logger.info(
                    f"Returning cached dataset(id={entity_id}, urn={dataset_config.urn!r})."
                )
                return ds

        try:
            _urn = self._urn_parser.parse(dataset_config.urn)
        except Exception as e:
            if allow_offline:
                msg = f"Failed to parse the URN={dataset_config.urn!r} from the dataset configuration."
                logger.exception(msg)
                return SdmxOfflineDataSet(
                    entity_id, title, dataset_config, self, status_details=msg
                )
            else:
                raise e

        urn = Urn(
            agency_id=_urn.agency_id,
            resource_id=_urn.resource_id,
            version=_urn.version if _urn.version else "latest",
        )

        sdmx_client = await self.create_sdmx_client(auth_context)

        try:
            dataflow_loader = DataflowLoader(sdmx_client)
            structure_message = await dataflow_loader.load_structure_message(urn)
        except Exception as e:
            if allow_offline:
                msg = "Failed to load the dataflow or its associated structures."
                logger.exception(msg)
                return SdmxOfflineDataSet(
                    entity_id, title, dataset_config, self, status_details=msg
                )
            else:
                raise e

        try:
            dimensions_creator = DimensionsCreator(structure_message, urn, self._config.locale)
            dimensions = await dimensions_creator.create_dimensions()
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dimensions from the loaded structure message."
                logger.exception(msg)
                return SdmxOfflineDataSet(
                    entity_id, title, dataset_config, self, status_details=msg
                )
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
                return SdmxOfflineDataSet(
                    entity_id, title, dataset_config, self, status_details=msg
                )
            else:
                raise e

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
                annotations=annotations,
            )
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dataset class."
                logger.exception(f"{msg}. See exception details below.")
                msg += (
                    " Probably there is a mistake in configuration."
                    " For example, the indicator dimension name is incorrect."
                )
                return SdmxOfflineDataSet(
                    entity_id, title, dataset_config, self, status_details=msg
                )
            else:
                raise e

        if allow_cached and not self._config.auth_enabled:
            # If auth is disabled, cache the dataset for all users
            # NOTE: we do not cache offline datasets
            self._dataset_cache.set(entity_id, res)
            logger.info(f"Cached dataset(id={entity_id}, urn={dataset_config.urn!r}).")

        return res

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
