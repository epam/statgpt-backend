import typing as t
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, alias_generators

from common.auth.auth_context import AuthContext

from .base import BaseEntity, EntityType
from .category import DimensionCategory
from .dataset import DataSet, DataSetConfig
from .indicator import BaseIndicator


class DataSetDescriptor(BaseModel):
    source_id: str = Field(description="The ID in the source of the dataset")
    name: str = Field(description="The name of the dataset")
    description: t.Optional[str] = Field(description="The description of the dataset")

    details: dict = Field(
        description="Preliminary details defined by the data source.", default_factory=dict
    )


class DataSourceConfig(BaseModel, ABC):
    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


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

    @staticmethod
    @abstractmethod
    def parse_config(d: dict) -> DataSourceConfigType:
        pass

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
        entity_id: str,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> DataSetType:
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def get_indicator_from_document(self, documents: Document) -> BaseIndicator:
        pass

    @abstractmethod
    async def document_to_dimension_category(self, documents: Document) -> DimensionCategory:
        pass

    @abstractmethod
    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        pass
