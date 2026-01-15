import typing as t
import uuid
from abc import ABC
from operator import itemgetter

from langchain_core.documents import Document
from sdmx.message import StructureMessage
from sdmx.model.common import TimeDimension
from sdmx.model.v21 import DataflowDefinition as Dataflow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import (
    BaseDimensionConfig,
    DataSetDescriptor,
    DatasetHierarchy,
    DataSourceHandler,
    DataSourceType,
    DefaultDatasetHierarchyCreator,
    DimensionType,
    IndicatorDimensionConfig,
    NonIndicatorDimensionConfig,
    SpecialNonIndicatorDimensions,
    TimePeriodDimensionConfig,
    VirtualDimensionCategory,
)
from statgpt.common.data.sdmx.common import (
    ComplexIndicator,
    DimensionCodeCategory,
    SdmxConstants,
    SdmxDataSetConfig,
    SdmxDataSetConfigTemplate,
    SdmxDataSourceConfig,
    SdmxDimension,
    UrnReference,
)
from statgpt.common.data.sdmx.v21.dataflow_loader import DataflowLoader
from statgpt.common.data.sdmx.v21.dataset import (
    InvalidConfigurationError,
    Sdmx21DataSet,
    SdmxOfflineDataSet,
)
from statgpt.common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient
from statgpt.common.schemas.dataset import Status
from statgpt.common.utils import crc32_hash
from statgpt.common.utils.timer import debug_timer

from .dataset_hierarchy import CategorySchemaDataSetHierarchyCreator
from .ratelimiter import SdmxRateLimiterFactory
from .schemas import StructureMessage21, Urn


class Sdmx21DataSourceHandler(
    DataSourceHandler[SdmxDataSourceConfig, Sdmx21DataSet | SdmxOfflineDataSet, SdmxDataSetConfig],
    ABC,
):
    _HIERARCHY_CREATORS = {
        'config': DefaultDatasetHierarchyCreator,
        'category_scheme': CategorySchemaDataSetHierarchyCreator,
    }

    def __init__(self, config: SdmxDataSourceConfig):
        super().__init__(config)

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="SDMX21", name="SDMX 2.1 Registry", description="SDMX 2.1 Registry data source"
        )

    @staticmethod
    def parse_config(d: dict) -> SdmxDataSourceConfig:
        return SdmxDataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> SdmxDataSetConfig:
        return SdmxDataSetConfig.model_validate(d)

    @property
    def source_id(self) -> str:
        return self._config.get_id()

    @property
    def name(self) -> str:
        return self._config.get_name()

    @property
    def description(self) -> t.Optional[str]:
        return self._config.description

    async def create_sdmx_client(self, auth_context: AuthContext) -> AsyncSdmxClient:
        rate_limiter = await SdmxRateLimiterFactory.get(
            self._config.get_id(), self._config.rate_limits
        )
        return AsyncSdmxClient.from_config(self._config, auth_context, rate_limiter)

    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        # There is no authorization for SDMX datasets, so they are always available
        return True

    async def list_datasets(self, auth_context: AuthContext) -> list[DataSetDescriptor]:
        client = await self.create_sdmx_client(auth_context)

        message: StructureMessage = await client.dataflow(
            agency_id="all",
            resource_id="all",
            version="latest",
            params={"references": "datastructure"},
        )
        dataflows: list[Dataflow] = list(message.dataflow.values())

        res = [self._get_dataset_descriptor(dataflow) for dataflow in dataflows]
        return res

    def _get_dataset_descriptor(self, dataflow: Dataflow) -> DataSetDescriptor:
        urn = Urn.for_artifact(dataflow)
        urn_ref = UrnReference.model_validate(urn, from_attributes=True)
        try:
            config = self._create_config_for(dataflow, urn_ref)
        except Exception:
            logger.warning(
                f"Failed to create dataset config for dataflow urn={dataflow.urn!r}", exc_info=True
            )
            config = SdmxDataSetConfigTemplate(urn=urn_ref)

        return DataSetDescriptor(
            name=dataflow.name[self._config.locale],
            description=dataflow.description.localizations.get(self._config.locale),
            details=config,
        )

    @staticmethod
    def _create_config_for(dataflow: Dataflow, urn_ref: UrnReference) -> SdmxDataSetConfigTemplate:
        """We do our best to create a valid dataset configuration from the dataflow structure."""

        dimensions: dict[str, BaseDimensionConfig] = {}
        for dim in dataflow.structure.dimensions:
            entity_id = dim.id
            if isinstance(dim, TimeDimension):
                dimensions[entity_id] = TimePeriodDimensionConfig()
            elif dim.id.upper() in ["FREQ", "FREQUENCY"]:
                dimensions[entity_id] = NonIndicatorDimensionConfig(
                    subtype=SpecialNonIndicatorDimensions.FREQUENCY
                )
            elif dim.id.upper() in ["INDICATOR", "SERIES"]:
                dimensions[entity_id] = IndicatorDimensionConfig(is_required=True)
            else:
                dimensions[entity_id] = BaseDimensionConfig(dimension_type=None)

        if not dimensions:
            raise ValueError(f"Could not find any dimensions in dataflow {dataflow.urn!r}")

        return SdmxDataSetConfigTemplate(urn=urn_ref, dimensions=dimensions)

    @property
    def entity_id(self) -> str:
        return self._config.get_id()

    def _validate_dataset_config(
        self, config: SdmxDataSetConfig, dimensions: list[SdmxDimension]
    ) -> None:
        problems = []

        dimensions_dict = {dim.entity_id: dim for dim in dimensions}

        for dim_id, dim_config in config.dimensions.items():
            if dim_config.virtual:
                continue  # Skip virtual dimensions

            dimension = dimensions_dict.get(dim_id)
            if dimension is None:
                problems.append(
                    f"{dim_config.dimension_type} dimension with id={dim_id!r} not found in the dataflow."
                )
                continue

            if dim_config.type is DimensionType.TIME_PERIOD and not dimension.is_time_dimension:
                problems.append(
                    f"Dimension with id={dim_id!r} is configured as time period dimension,"
                    f" but it is not a time dimension in the dataflow."
                )

        for dim in dimensions_dict.keys():
            if dim not in config.dimensions:
                problems.append(
                    f"Dimension with id={dim!r} is present in the dataflow but not configured in the dataset configuration."
                )

        if problems:
            raise ValueError("Dataset configuration validation failed:\n" + "\n".join(problems))

    async def _get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> Sdmx21DataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

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
            dataflow = structure_message.dataflow[urn]
            return Sdmx21DataSet(
                entity_id=entity_id,
                title=title,
                config=dataset_config,
                handler=self,
                dataflow=dataflow,
                locale=self._config.locale,
                dimensions=dimensions,
                attributes=dataflow.structure.attributes,
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

    async def get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> Sdmx21DataSet | SdmxOfflineDataSet:
        with debug_timer(f"Sdmx21DataSourceHandler.get_dataset: {title}"):
            return await self._get_dataset(
                entity_id,
                title,
                config,
                auth_context,
                allow_offline=allow_offline,
            )

    async def get_structure_hash_and_metadata(
        self, dataset_config: dict, auth_context: AuthContext
    ) -> tuple[str, dict]:
        config = self.parse_data_set_config(dataset_config)
        urn = Urn(
            agency_id=config.urn.agency_id,
            resource_id=config.urn.resource_id,
            version=config.urn.version or "latest",
        )

        sdmx_client = await self.create_sdmx_client(auth_context)

        dataflow_loader = DataflowLoader(sdmx_client)
        urn, structure_message = await dataflow_loader.load_structure_message(urn, mode="shallow")

        meta_json = {
            "dimensions": self._get_dimensions_from(structure_message, urn),
        }
        return str(crc32_hash(str(meta_json))), meta_json

    def _get_dimensions_from(
        self, structure_message: StructureMessage21, dataflow_urn: Urn
    ) -> list[dict[str, str]]:
        locale = self.config.locale
        dsd = structure_message.dataflow[dataflow_urn].structure

        res: list[dict[str, str]] = []
        for dimension in dsd.dimensions.components:
            if dimension.concept_identity is None:
                continue
            scheme_urn = Urn.for_artifact(dimension.concept_identity.parent)  # type: ignore[arg-type]
            scheme = structure_message.concept_scheme[scheme_urn]
            concept_id = dimension.concept_identity.id
            dimension_name = scheme.items[concept_id].name.localized_default(locale)
            res.append({"entity_id": dimension.id, "name": dimension_name})
        res = sorted(res, key=itemgetter("entity_id"))
        return res

    def get_structure_metadata_diff(self, old_metadata: dict | None, new_metadata: dict) -> dict:
        if old_metadata is None:
            return {'message': 'No previous metadata to compare.'}

        try:
            old_dimensions = {dim['entity_id']: dim for dim in old_metadata.get('dimensions', [])}
            new_dimensions = {dim['entity_id']: dim for dim in new_metadata['dimensions']}
            return self._compare_dimension_meta(old_dimensions, new_dimensions)
        except Exception as e:
            logger.warning(f"Cannot compute structure metadata diff: {e}", exc_info=True)
            return {'message': 'Could not compute diff due to error.'}

    @staticmethod
    def _compare_dimension_meta(old: dict[str, dict], current: dict[str, dict]) -> dict:
        result: dict[str, t.Any] = {}

        new = set(current.keys()).difference(old.keys())
        if new:
            result['new_dimensions'] = [current[dim_id] for dim_id in new]

        removed = set(old.keys()).difference(current.keys())
        if removed:
            result['removed_dimensions'] = [old[dim_id] for dim_id in removed]

        modified = {}
        for dim_id in set(old.keys()).intersection(current.keys()):
            old_dim = old[dim_id]
            current_dim = current[dim_id]
            changes = {}
            for field in ['name']:
                old_value = getattr(old_dim, field, None)
                current_value = getattr(current_dim, field, None)
                if old_value != current_value:
                    changes[field] = {'old': old_value, 'new': current_value}
            if changes:
                modified[dim_id] = changes
        if modified:
            result['modified_dimensions'] = modified

        return result

    async def close(self):
        # do nothing
        pass

    async def get_indicator_from_document(self, document: Document) -> ComplexIndicator:
        return ComplexIndicator.from_document(document)

    async def document_to_dimension_category(
        self, document: Document
    ) -> DimensionCodeCategory | VirtualDimensionCategory:
        if SdmxConstants.METADATA_DIMENSION_ID in document.metadata:
            return DimensionCodeCategory.from_document(document)
        else:
            return VirtualDimensionCategory.from_document(document)

    async def get_dataset_hierarchy(self, auth_context: AuthContext) -> DatasetHierarchy | None:
        if self.config.dataset_hierarchy is None:
            return None

        try:
            if self.config.dataset_hierarchy.type not in self._HIERARCHY_CREATORS:
                raise ValueError(
                    f"Unsupported dataset hierarchy type: {self.config.dataset_hierarchy.type}"
                )

            creator_class = self._HIERARCHY_CREATORS[self.config.dataset_hierarchy.type]
            creator = creator_class(self, auth_context)
            return await creator.create_hierarchy()
        except Exception:
            logger.exception(
                "Failed to create dataset hierarchy. Returning None. See exception details below."
            )
            return None

    def merge_config_with_resolved(
        self,
        current_config: dict,
        resolved_config: dict,
    ) -> dict:
        """Uses the new configuration except for the resolved URN."""
        result = current_config.copy()
        if 'urn' in resolved_config:
            result['urn'] = resolved_config['urn']
        return result
