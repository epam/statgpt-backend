import typing as t
import uuid
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from pydantic import Field, SerializeAsAny

from statgpt.common.auth.auth_context import AuthContext

from .base import BaseEntity, BaseModel, EntityType
from .category import DimensionCategory
from .config import DataSetConfigTemplate
from .dataset import DataSet, DataSetConfig
from .dataset_hierarchy import DatasetHierarchy
from .indicator import BaseIndicator


class DataSetDescriptor(BaseModel, ABC):
    name: str = Field(description="The name of the dataset")
    description: str | None = Field(description="The description of the dataset")

    details: SerializeAsAny[DataSetConfigTemplate] = Field(
        description="Preliminary details defined by the data source."
    )

    @property
    @abstractmethod
    def id_in_source(self) -> str:
        """The unique identifier of the dataset within its data source."""


class DataSetValidationResult(BaseModel):
    is_valid: bool = Field(description="Indicates whether the dataset configuration is valid.")
    errors: list[str] = Field(
        default_factory=list,
        description="A list of error messages if the configuration is invalid.",
    )


class DataSetStructure(BaseModel, ABC):
    """Abstract base class for dataset structure representation."""


class DataSetHierarchyConfig(BaseModel):
    """Default configuration of a dataset hierarchy using the configuration as the source of the hierarchy."""

    # TODO: add fields and implement hierarchy creation logic


class DataSourceConfig(BaseModel, ABC):
    pass


DataSourceConfigType = t.TypeVar("DataSourceConfigType", bound=DataSourceConfig)
DataSetType = t.TypeVar("DataSetType", bound=DataSet)
DataSetConfigType = t.TypeVar("DataSetConfigType", bound=DataSetConfig)


class DataSourceType(BaseModel):
    type_id: str = Field(description="The ID of the data source type")
    name: str = Field(description="The name of the data source type")
    description: t.Optional[str] = Field(description="The description of the data source type")

    def __hash__(self):
        return hash(self.type_id)


class DataSourceHandler(
    BaseEntity, t.Generic[DataSourceConfigType, DataSetType, DataSetConfigType], ABC
):
    def __init__(self, config: DataSourceConfigType):
        super().__init__()
        self._config = config

    @property
    def entity_type(self) -> EntityType:
        return EntityType.DATA_SOURCE

    @property
    def config(self) -> DataSourceConfigType:
        return self._config

    @staticmethod
    @abstractmethod
    def parse_config(d: dict) -> DataSourceConfigType:
        pass

    @staticmethod
    @abstractmethod
    def get_data_set_config_schema() -> dict:
        """Get the JSON schema for the dataset configuration specific to this data source type."""

    @staticmethod
    @abstractmethod
    def parse_data_set_config(d: dict) -> DataSetConfigType:
        pass

    @staticmethod
    @abstractmethod
    def data_source_type() -> DataSourceType:
        pass

    @abstractmethod
    async def list_datasets(self, auth_context: AuthContext) -> t.Sequence[DataSetDescriptor]:
        pass

    @abstractmethod
    async def get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
        allow_cached: bool = False,
    ) -> DataSetType:
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def get_structure_hash_and_metadata(
        self, dataset_config: dict, auth_context: AuthContext
    ) -> tuple[str, dict]:
        """Get a hash calculated based on the part of the dataset structure that is important for indexing.
        If this hash changes, the dataset should be re-indexed.

        Additionally, return metadata about the structure that can be used to understand what has changed.
        The metadata is a JSON-serializable dictionary.
        """

    @abstractmethod
    def get_structure_metadata_diff(self, old_metadata: dict | None, new_metadata: dict) -> dict:
        """Get the difference between the old and new structure metadata of a dataset.

        Return a JSON-serializable dictionary describing what has changed.
        """

    @abstractmethod
    async def get_indicator_from_document(self, documents: Document) -> BaseIndicator:
        pass

    @abstractmethod
    async def document_to_dimension_category(self, documents: Document) -> DimensionCategory:
        pass

    @abstractmethod
    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        pass

    async def get_dataset_hierarchy(self, auth_context: AuthContext) -> DatasetHierarchy | None:
        return None

    @abstractmethod
    def merge_config_with_resolved(
        self,
        current_config: dict,
        resolved_config: dict,
    ) -> dict:
        """Merge current config with resolved config from indexing time.

        Takes current_config and merges with resolved_config to preserve
        any dynamically resolved values (implementation-specific).
        """

    @abstractmethod
    async def validate_dataset_config(
        self, config: dict, auth_context: AuthContext, mode: t.Literal["raise", "return"] = "raise"
    ) -> DataSetValidationResult:
        pass

    @abstractmethod
    async def get_dataset_structure(
        self, config: dict, auth_context: AuthContext
    ) -> DataSetStructure:
        pass
