import uuid
from collections.abc import Iterable

from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.quanthub.sdmx_schemas.v30 import QhAnnotation
from statgpt.common.data.quanthub.v21.datasource import QuanthubSdmx21DataSourceHandler
from statgpt.common.data.sdmx.common import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v30.config import ProxySdmx30DataSourceConfig
from statgpt.common.data.sdmx.v30.dataset import Sdmx30ProxyDataSet
from statgpt.common.data.sdmx.v30.sdmx_client import AsyncProxySdmxClient


class ProxySdmx30DataSourceHandler(QuanthubSdmx21DataSourceHandler):
    """SDMX 3.0 proxy source that is parsed via sdmx1 (SDMX 2.1) models."""

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
        logger.info(
            "Creating SDMX client: type=PROXY_SDMX30 id=%s name=%s url=%s",
            self._config.get_id(),
            self._config.get_name(),
            self._config.sdmx_config.url,
        )
        return AsyncProxySdmxClient.from_config(self._config, auth_context, rate_limiter)

    def _build_dataset(
        self,
        *,
        entity_id: uuid.UUID,
        title: str,
        config: QuanthubDataSetConfig,
        dataflow: DataFlow,
        dimensions: Iterable[SdmxDimension],
        attributes: Iterable[Sdmx21Attribute],
        attribute_values: dict[str, str | None],
        annotations: Iterable[QhAnnotation],
    ) -> Sdmx30ProxyDataSet:
        return Sdmx30ProxyDataSet(
            entity_id=entity_id,
            title=title,
            config=config,
            handler=self,
            dataflow=dataflow,
            locale=self._config.locale,
            dimensions=dimensions,
            attributes=attributes,
            attribute_values=attribute_values,
            annotations=annotations,
        )
