import uuid
from collections.abc import Mapping
from typing import Any

from sdmx.model.common import BaseAnnotation
from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.proxy.config import ProxySdmx30DataSourceConfig
from statgpt.common.data.proxy.sdmx_schemas.structure_message import ProxyAnnotation
from statgpt.common.data.proxy.v30.dataset import Sdmx30ProxyDataSet
from statgpt.common.data.proxy.v30.sdmx_client import AsyncProxySdmxClient
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dataset import InvalidConfigurationError, SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn
from statgpt.common.data.sdmx.v21.urn_utils import is_wildcarded_version, lookup_urn
from statgpt.common.schemas.dataset import Status
from statgpt.common.settings.sdmx import proxy_sdmx_settings
from statgpt.common.utils import Cache
from statgpt.common.utils.timer import debug_timer


class ProxySdmx30DataSourceHandler(Sdmx21DataSourceHandler):
    """SDMX 3.0 proxy source that is parsed via sdmx1 (SDMX 2.1) models."""

    _dataset_cache: Cache[Sdmx30ProxyDataSet] = Cache(ttl=proxy_sdmx_settings.dataset_cache_ttl)

    def __init__(self, config: ProxySdmx30DataSourceConfig):
        super().__init__(config)
        self._config: ProxySdmx30DataSourceConfig = config  # for type hinting

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="PROXY_SDMX30",
            name="Proxy SDMX 3.0",
            description="SDMX 3.0 proxy data source (parsed with sdmx1)",
        )

    @staticmethod
    def parse_config(d: dict) -> ProxySdmx30DataSourceConfig:
        return ProxySdmx30DataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> QuanthubDataSetConfig:
        return QuanthubDataSetConfig.model_validate(d)

    async def create_sdmx_client(self, auth_context: AuthContext) -> AsyncProxySdmxClient:
        rate_limiter = await SdmxRateLimiterFactory.get(
            self._config.get_id(), self._config.rate_limits
        )
        return AsyncProxySdmxClient.from_config(self._config, auth_context, rate_limiter)

    def _should_use_cache(self, allow_cached: bool) -> bool:
        auth_enabled = getattr(self._config, "auth_enabled", False)
        return allow_cached and not auth_enabled

    async def _load_extra_dataset_data(
        self, sdmx_client: Any, urn: Urn, structure_message: StructureMessage21 | None = None
    ) -> Mapping[str, Any]:
        if structure_message is None:
            return {}

        dataflow = structure_message.dataflow.get(urn)
        if dataflow is None:
            return {}

        annotations = [self._to_proxy_annotation(a) for a in dataflow.annotations]
        return {"annotations": annotations}

    @staticmethod
    def _to_proxy_annotation(annotation: BaseAnnotation) -> ProxyAnnotation:
        text = str(annotation.text) if annotation.text else None
        return ProxyAnnotation(
            id=annotation.id,
            title=annotation.title,
            type=annotation.type,
            value=annotation.value,
            text=text,
        )

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
    ) -> Sdmx30ProxyDataSet:
        return Sdmx30ProxyDataSet(
            entity_id=entity_id,
            title=title,
            config=dataset_config,
            handler=self,
            dataflow=dataflow,
            locale=self._config.locale,
            dimensions=dimensions,
            attributes=attributes,
            annotations=extra_data.get("annotations", []),
        )

    async def _get_dataset(  # type: ignore[override]
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> Sdmx30ProxyDataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

        if self._should_use_cache(allow_cached):
            if ds := self._dataset_cache.get(str(entity_id)):
                logger.debug(
                    f"Returning cached dataset(id={entity_id}, urn={dataset_config.urn!r})"
                )
                return ds

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
        except Exception:
            if allow_offline:
                msg = f"Failed to load the dataflow or its associated structures. urn={dataset_config.urn!r}"
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise

        try:
            dimensions_creator = DimensionsCreator(
                structure_message, urn, self._config.locale, dataset_config.get_dimension_aliases()
            )
            dimensions = await dimensions_creator.create_dimensions()
        except Exception:
            if allow_offline:
                msg = "Failed to create dimensions from the loaded structure message."
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise

        try:
            attributes_creator = Sdmx21AttributesCreator(
                structure_message, urn, self._config.locale
            )
            attributes = await attributes_creator.create_attributes()
        except Exception:
            if allow_offline:
                msg = "Failed to create attributes from the loaded structure message."
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise

        try:
            extra_data = await self._load_extra_dataset_data(
                sdmx_client=sdmx_client, urn=urn, structure_message=structure_message
            )
        except Exception:
            if allow_offline:
                msg = "Failed to load additional dataset metadata."
                logger.exception(f"{msg}. See exception details below.")
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise

        try:
            dataflow = structure_message.dataflow[urn]
            if is_wildcarded_version(dataflow.structure.version):
                dataflow.structure = lookup_urn(
                    structure_message.structure, Urn.for_artifact(dataflow.structure)
                )
            result = self._build_dataset(
                entity_id=entity_id,
                title=title,
                dataset_config=dataset_config,
                dataflow=dataflow,
                dimensions=dimensions,
                attributes=attributes,
                extra_data=extra_data,
            )
        except InvalidConfigurationError as e:
            if allow_offline:
                msg = f"Invalid dataset(urn={dataset_config.urn!r}) configuration: {e}"
                logger.warning(msg)
                status = Status(status='invalid_config', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise
        except Exception:
            if allow_offline:
                msg = "Failed to create dataset class."
                logger.exception(f"{msg}. See exception details below.")
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            raise

        if self._should_use_cache(allow_cached):
            self._dataset_cache.set(str(entity_id), result)
            logger.info(f"Cached dataset(id={entity_id}, urn={dataset_config.urn!r}).")

        return result

    async def get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> Sdmx30ProxyDataSet | SdmxOfflineDataSet:
        with debug_timer(f"ProxySdmx30DataSourceHandler.get_dataset: {title}"):
            return await self._get_dataset(
                entity_id,
                title,
                config,
                auth_context,
                allow_offline=allow_offline,
                allow_cached=allow_cached,
            )
