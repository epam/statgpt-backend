import random
import typing as t
import uuid
from abc import ABC
from operator import itemgetter

from langchain_core.documents import Document
from pydantic import BaseModel, Field, SerializeAsAny, ValidationError, computed_field
from sdmx.message import StructureMessage
from sdmx.model.common import TimeDimension
from sdmx.model.v21 import DataflowDefinition as Dataflow

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import (
    BaseDimensionConfig,
    CategoricalDimension,
    DataSetDescriptor,
    DatasetHierarchy,
    DataSetStructure,
    DataSetValidationResult,
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
    UrnReference,
)
from statgpt.common.data.sdmx.common.dimension import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.attributes_creator import Sdmx21AttributesCreator
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


class SdmxDataSetDescriptor(DataSetDescriptor):
    details: SdmxDataSetConfigTemplate = Field(
        description="Preliminary details defined by the SDMX data source."
    )

    @property
    def id_in_source(self) -> str:
        return self.details.urn.short_urn()


class SdmxEntityMetadata(BaseModel):
    entity_id: str
    name: str
    desciption: str | None
    type: str | None = None


class Item(BaseModel):
    id: str
    name: str


class CategoryDimensionMetadata(SdmxEntityMetadata):
    values_count: int
    values_sample: list[Item]


class SdmxDataSetStructure(DataSetStructure):
    provided_urn: UrnReference = Field(
        description=(
            "The URN of the SDMX dataflow as provided by the user."
            " May contain dynamic values or wildcards. E.g., 'latest', 'all', 'X.Y.0+', etc."
        )
    )
    resolved_urn: UrnReference = Field(
        description="The actual URN of the dataflow as resolved from the SDMX registry."
    )

    dimensions: list[SerializeAsAny[SdmxEntityMetadata]]
    attributes: list[SerializeAsAny[SdmxEntityMetadata]]
    description: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dimension_count(self) -> int:
        return len(self.dimensions)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def attribute_count(self) -> int:
        return len(self.attributes)


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

    @staticmethod
    def get_data_set_config_schema() -> dict:
        return SdmxDataSetConfig.model_json_schema(by_alias=True)

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

    @staticmethod
    def _get_dimension_metadata(dim: SdmxDimension) -> SdmxEntityMetadata:
        if isinstance(dim, CategoricalDimension):
            values = dim.available_values
            sample_size = min(10, len(values))
            sample_values = [
                Item(id=value.query_id, name=value.name)
                for value in random.sample(values, sample_size)
            ]
            return CategoryDimensionMetadata(
                entity_id=dim.entity_id,
                name=dim.name,
                desciption=dim.description,
                type=dim.dimension_type.value if dim.dimension_type else None,
                values_count=len(values),
                values_sample=sample_values,
            )

        return SdmxEntityMetadata(
            entity_id=dim.entity_id,
            name=dim.name,
            desciption=dim.description,
            type=dim.dimension_type.value if dim.dimension_type else None,
        )

    @staticmethod
    def _get_attribute_metadata(attr: Sdmx21Attribute) -> SdmxEntityMetadata:
        return SdmxEntityMetadata(
            entity_id=attr.entity_id,
            name=attr.name,
            desciption=attr.description,
            type=attr.attribute_type.value if attr.attribute_type else None,
        )

    @staticmethod
    def _get_description(
        structure_message: StructureMessage21, urn: Urn, locale: str = "en"
    ) -> str | None:
        return structure_message.dataflow[urn].description.localizations.get(locale)

    async def get_dataset_structure(
        self, config: dict, auth_context: AuthContext
    ) -> SdmxDataSetStructure:

        urn_ref = UrnReference.model_validate(config['urn'])
        actual_urn, structure_message = await self._load_dataset_structure_message(
            urn_ref=urn_ref, auth_context=auth_context
        )

        dimensions = await self._get_dimensions(actual_urn, structure_message, {})
        attributes = await self._get_attributes(actual_urn, structure_message)

        return SdmxDataSetStructure(
            provided_urn=urn_ref,
            resolved_urn=UrnReference.model_validate(actual_urn, from_attributes=True),
            dimensions=[self._get_dimension_metadata(dim) for dim in dimensions],
            attributes=[self._get_attribute_metadata(attr) for attr in attributes],
            description=self._get_description(structure_message, actual_urn, self._config.locale),
        )

    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        # There is no authorization for SDMX datasets, so they are always available
        return True

    async def list_datasets(self, auth_context: AuthContext) -> list[SdmxDataSetDescriptor]:
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

    def _get_dataset_descriptor(self, dataflow: Dataflow) -> SdmxDataSetDescriptor:
        urn = Urn.for_artifact(dataflow)
        urn_ref = UrnReference.model_validate(urn, from_attributes=True)
        try:
            config = self._create_config_for(dataflow, urn_ref)
        except Exception:
            logger.warning(
                f"Failed to create dataset config for dataflow urn={dataflow.urn!r}", exc_info=True
            )
            config = SdmxDataSetConfigTemplate(urn=urn_ref)

        return SdmxDataSetDescriptor(
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
            elif dim.id.upper() in ["COUNTRY", "REGION", "REF_AREA"]:
                dimensions[entity_id] = NonIndicatorDimensionConfig(
                    subtype=SpecialNonIndicatorDimensions.REGION
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

    async def validate_dataset_config(
        self, config: dict, auth_context: AuthContext, mode: t.Literal["raise", "return"] = "raise"
    ) -> DataSetValidationResult:

        try:
            dataset_config = self.parse_data_set_config(config)
        except ValidationError as e:
            if mode == "raise":
                raise
            else:
                return DataSetValidationResult(
                    is_valid=False,
                    errors=[str(err) for err in e.errors()],
                )

        urn, structure_message = await self._load_dataset_structure_message(
            urn_ref=dataset_config.urn, auth_context=auth_context
        )
        dimensions = await self._get_dimensions(
            urn, structure_message, aliases=dataset_config.get_dimension_aliases()
        )

        problems = self._validate_dimension_config(dimensions, dataset_config)
        if problems:
            if mode == "raise":
                raise ValueError("Dataset configuration validation failed:\n" + "\n".join(problems))
            else:
                return DataSetValidationResult(is_valid=False, errors=problems)

        return DataSetValidationResult(is_valid=True)

    def _validate_dimension_config(
        self, dimensions: list[SdmxDimension], dataset_config: SdmxDataSetConfig
    ) -> list[str]:
        problems = []

        dimensions_dict = {dim.entity_id: dim for dim in dimensions}

        for dim_id, dim_config in dataset_config.dimensions.items():
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
            if dim not in dataset_config.dimensions:
                problems.append(
                    f"Dimension with id={dim!r} is present in the dataflow"
                    f" but not configured in the dataset configuration."
                )

        return problems

    async def _load_dataset_structure_message(
        self, urn_ref: UrnReference, auth_context: AuthContext
    ) -> tuple[Urn, StructureMessage21]:
        """Load the structure message for the dataset's dataflow."""
        urn = Urn(
            agency_id=urn_ref.agency_id, resource_id=urn_ref.resource_id, version=urn_ref.version
        )
        sdmx_client = await self.create_sdmx_client(auth_context)
        dataflow_loader = DataflowLoader(sdmx_client)
        urn, structure_message = await dataflow_loader.load_structure_message(urn, mode="full")
        return urn, structure_message

    async def _get_dimensions(
        self, urn: Urn, structure_message: StructureMessage21, aliases: dict[str, str]
    ) -> list[SdmxDimension]:
        """Create dimensions from the structure message based on the dataset configuration."""
        dimensions_creator = DimensionsCreator(structure_message, urn, self._config.locale, aliases)
        dimensions = await dimensions_creator.create_dimensions()
        return dimensions

    async def _get_attributes(
        self, urn: Urn, structure_message: StructureMessage21
    ) -> list[Sdmx21Attribute]:
        attributes_creator = Sdmx21AttributesCreator(structure_message, urn, self._config.locale)
        attributes = await attributes_creator.create_attributes()
        return attributes

    async def _get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> Sdmx21DataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)

        try:
            urn, structure_message = await self._load_dataset_structure_message(
                urn_ref=dataset_config.urn, auth_context=auth_context
            )
        except Exception as e:
            if allow_offline:
                msg = f"Failed to load the dataflow or its associated structures. urn={dataset_config.urn!r}"
                logger.exception(msg)
                status = Status(status='offline', details=msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, self, status)
            else:
                raise e

        try:
            dimensions = await self._get_dimensions(
                urn, structure_message, aliases=dataset_config.get_dimension_aliases()
            )
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
