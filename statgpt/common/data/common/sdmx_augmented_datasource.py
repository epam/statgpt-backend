import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dataset import InvalidConfigurationError, SdmxOfflineDataSet
from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.schemas import StructureMessage21, Urn
from statgpt.common.data.sdmx.v21.urn_utils import is_wildcarded_version, lookup_urn
from statgpt.common.schemas.dataset import Status
from statgpt.common.utils import Cache


class SdmxAugmentedDataSourceHandler(Sdmx21DataSourceHandler, ABC):
    """Shared SDMX dataset loading flow for handlers that add custom metadata."""

    _dataset_cache: Cache[Any]

    async def _get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> Any | SdmxOfflineDataSet:
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

    def _should_use_cache(self, allow_cached: bool) -> bool:
        # If auth is disabled, we can cache datasets for all users.
        # Not all SDMX datasource configs support auth; treat missing flag as disabled.
        auth_enabled = getattr(self._config, "auth_enabled", False)
        return allow_cached and not auth_enabled

    async def _load_extra_dataset_data(
        self, sdmx_client: Any, urn: Urn, structure_message: StructureMessage21 | None = None
    ) -> Mapping[str, Any]:
        return {}

    @abstractmethod
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
    ) -> Any:
        pass
