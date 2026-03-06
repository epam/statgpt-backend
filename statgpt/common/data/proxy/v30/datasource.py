import uuid
from collections.abc import Mapping
from typing import Any

from sdmx.model.common import BaseAnnotation
from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.common import SdmxAugmentedDataSourceHandler
from statgpt.common.data.proxy.config import ProxySdmx30DataSourceConfig
from statgpt.common.data.proxy.sdmx_schemas.structure_message import ProxyAnnotation
from statgpt.common.data.proxy.v30.dataset import Sdmx30ProxyDataSet
from statgpt.common.data.proxy.v30.sdmx_client import AsyncProxySdmxClient
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.dataset import SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn
from statgpt.common.settings.sdmx import proxy_sdmx_settings
from statgpt.common.utils import Cache
from statgpt.common.utils.timer import debug_timer


class ProxySdmx30DataSourceHandler(SdmxAugmentedDataSourceHandler):
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
        config = dataset_config
        return Sdmx30ProxyDataSet(
            entity_id=entity_id,
            title=title,
            config=config,
            handler=self,
            dataflow=dataflow,
            locale=self._config.locale,
            dimensions=dimensions,
            attributes=attributes,
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
