import uuid
from collections.abc import Mapping
from typing import Any

from sdmx.model.common import BaseAnnotation
from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSourceType
from statgpt.common.data.base.sdmx_schemas import Sdmx30AnnotationModel
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dataset import InvalidConfigurationError, SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler, SdmxDataSetDescriptor
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiterFactory
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn
from statgpt.common.data.sdmx.v21.urn_utils import is_wildcarded_version, lookup_urn
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.v30.dataset import StatGptSdmxProxyDataSet
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import (
    AsyncStatGptSdmxProxyClient as SdmxClient,
)
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import proxy_structure_extra_headers
from statgpt.common.schemas.dataset import Status
from statgpt.common.settings.sdmx import statgpt_sdmx_proxy_settings
from statgpt.common.utils import TtlCache
from statgpt.common.utils.timer import debug_timer


class StatGptSdmxProxyDataSourceHandler(Sdmx21DataSourceHandler):
    """StatGPT SDMX proxy data source (SDMX 3.0 API, parsed as SDMX 2.1 models)."""

    _dataset_cache: TtlCache[StatGptSdmxProxyDataSet] = TtlCache(
        ttl=statgpt_sdmx_proxy_settings.dataset_cache_ttl
    )

    def __init__(self, config: StatGptSdmxProxyDataSourceConfig):
        super().__init__(config)
        self._config: StatGptSdmxProxyDataSourceConfig = config  # for type hinting

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="PROXY_SDMX30",
            name="StatGPT SDMX Proxy",
            description="StatGPT SDMX 3.0 proxy API (parsed with SDMX 2.1 models)",
        )

    @staticmethod
    def parse_config(d: dict) -> StatGptSdmxProxyDataSourceConfig:
        return StatGptSdmxProxyDataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> QuanthubDataSetConfig:
        return QuanthubDataSetConfig.model_validate(d)

    async def create_sdmx_client(self, auth_context: AuthContext) -> SdmxClient:
        rate_limiter = await SdmxRateLimiterFactory.get(
            self._config.get_id(), self._config.rate_limits
        )
        return SdmxClient.from_config(self._config, auth_context, rate_limiter)

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
    def _to_proxy_annotation(annotation: BaseAnnotation) -> Sdmx30AnnotationModel:
        text = str(annotation.text) if annotation.text else None
        return Sdmx30AnnotationModel(
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
    ) -> StatGptSdmxProxyDataSet:
        return StatGptSdmxProxyDataSet(
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
    ) -> StatGptSdmxProxyDataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

        if allow_cached:
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
            dataflow_loader = DataflowLoader(
                sdmx_client, structure_extra_headers=proxy_structure_extra_headers
            )
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

        if allow_cached:
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
    ) -> StatGptSdmxProxyDataSet | SdmxOfflineDataSet:
        with debug_timer(f"StatGptSdmxProxyDataSourceHandler.get_dataset: {title}"):
            return await self._get_dataset(
                entity_id,
                title,
                config,
                auth_context,
                allow_offline=allow_offline,
                allow_cached=allow_cached,
            )

    async def list_datasets(self, auth_context: AuthContext) -> list[SdmxDataSetDescriptor]:
        """
        NOTE: The proxy SDMX 3.0 data source does not support querying across "all" agencies,
        and the use of "latest" version is not supported by all registries.
        Listing all datasets is unavailable in this handler.
        """
        raise NotImplementedError(
            "Listing all datasets is unavailable for StatGPT SDMX proxy handler"
        )
