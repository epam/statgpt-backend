import uuid
from collections.abc import Mapping
from typing import Any

import httpx
from httpx import HTTPStatusError
from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.common import SdmxAugmentedDataSourceHandler
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig, QuanthubSdmxDataSourceConfig
from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.dataset import SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn
from statgpt.common.settings.sdmx import quanthub_settings
from statgpt.common.utils import Cache
from statgpt.common.utils.timer import debug_timer

from .qh_sdmx_client import AsyncQuanthubClient


class QuanthubSdmx21DataSourceHandler(SdmxAugmentedDataSourceHandler):

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
                    raise

    async def _load_extra_dataset_data(
        self, sdmx_client: Any, urn: Urn, structure_message: StructureMessage21 | None = None
    ) -> Mapping[str, Any]:
        client = sdmx_client
        try:
            attribute_values = await client.dataset_level_attributes(
                agency_id=urn.agency_id,
                resource_id=urn.resource_id,
                version=urn.version,
            )
        except Exception:
            logger.exception(f"Failed to load dataset-level attributes for the dataflow({urn}).")
            attribute_values = {}

        try:
            annotations = await client.dynamic_dataflow_annotations(
                agency_id=urn.agency_id,
                resource_id=urn.resource_id,
                version=urn.version,
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

        return {
            "attribute_values": attribute_values,
            "annotations": annotations,
        }

    def _build_dataset(
        self,
        *,
        entity_id: uuid.UUID,
        title: str,
        dataset_config: Any,
        dataflow: DataFlow,
        dimensions: list[Any],
        attributes: list[Sdmx21Attribute],
        extra_data: Mapping[str, Any],
    ) -> QuanthubSdmx21DataSet:
        config = dataset_config
        return QuanthubSdmx21DataSet(
            entity_id=entity_id,
            title=title,
            config=config,
            handler=self,
            dataflow=dataflow,
            locale=self._config.locale,
            dimensions=dimensions,
            attributes=attributes,
            attribute_values=extra_data.get("attribute_values", {}),
            annotations=extra_data.get("annotations", []),
        )

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
            return await self._get_dataset(
                entity_id,
                title,
                config,
                auth_context,
                allow_offline=allow_offline,
                allow_cached=allow_cached,
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
