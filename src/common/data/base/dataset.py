from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field, StrictStr, alias_generators

from common.auth.auth_context import AuthContext
from common.config.utils import replace_env
from common.schemas.dataset import Status

from .attribute import Attribute
from .base import BaseEntity, EntityType
from .dimension import Dimension, VirtualDimension, VirtualDimensionConfig
from .indicator import BaseIndicator
from .query import DataSetAvailabilityQuery, DataSetQuery, Query

if t.TYPE_CHECKING:
    from common.data.base.datasource import DataSourceHandler


class DatasetCitation(BaseModel):
    provider: StrictStr | None = Field(default=None)
    last_updated: StrictStr | None = Field(default=None)
    url: StrictStr | None = Field(default=None)
    description: StrictStr | None = Field(default=None)

    def get_url(self) -> str | None:
        if self.url:
            return replace_env(self.url)
        return None


class IndexerIndicatorAnnotationConfig(BaseModel):
    description: str = Field(description="annotation name to get indicator description", default="")

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class IndexerIndicatorConfig(BaseModel):
    unpack: bool = Field(default=False)
    use_code_list_description: bool = Field(default=False)
    super_primary: bool = Field(default=False)

    annotations: IndexerIndicatorAnnotationConfig | None = Field(default=None)


model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class IndexerConfig(BaseModel):
    description: str = Field(description="dataset_description", default="")

    indicator: IndexerIndicatorConfig = Field(
        description="indicator_config", default_factory=IndexerIndicatorConfig
    )

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class SpecialDimension(BaseModel):
    dimension_id: str = Field()
    processor_id: str = Field()


class DataSetConfig(BaseModel, ABC):
    is_official: bool = Field(default=False)
    dimension_default_queries: t.Dict[str, t.List[Query]] = Field(
        description="Default queries for each dimension if any",
        default_factory=dict,
    )
    citation: DatasetCitation | None = Field(default=None)
    indexer: IndexerConfig | None = Field(default=None)
    special_dimensions: list[SpecialDimension] = Field(
        default_factory=list, description="The list of dimensions which require a special handling"
    )
    virtual_dimensions: t.List[VirtualDimensionConfig] = Field(
        description="The list of virtual dimensions (e.g. Country for datasets by national agencies)",
        default_factory=list,
    )
    pinned_columns: t.List[str] = Field(
        description="Column names and order to pin in the data in grid", default_factory=list
    )

    @abstractmethod
    def get_source_id(self) -> str:
        pass

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class DataResponse(ABC):
    """Base class for data responses from datasets."""

    @property
    @abstractmethod
    def file_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def visual_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame suitable for visualization and export (Plotly grid, CSV file)"""

    @abstractmethod
    def enrich_attachment_name(self, value: str) -> str:
        """Replace placeholders in the attachment name with actual values."""

    @abstractmethod
    def merge(self, other: "DataResponse") -> "DataResponse":
        """Merge another DataResponse into a new DataResponse instance.

        NOTE: This method can be used only for responses from the same dataset.
        """

    @property
    @abstractmethod
    def custom_table_dict(self) -> dict | None:
        """Return a dictionary in format suitable for displaying custom table attachment (AI DIAL Custom Visualizer)."""

    @property
    @abstractmethod
    def plotly_grid(self) -> go.Figure | None:
        """Return a Plotly grid figure"""

    @abstractmethod
    def get_plotly_graphs_with_names(self, template: str) -> list[tuple[str, go.Figure]]:
        """Return a list of Plotly graphs with their names formatted according to the template."""

    @property
    @abstractmethod
    def url_query(self) -> str | None:
        """Return the URL query to receive the data in this response."""

    @property
    @abstractmethod
    def json_query_old(self) -> dict | None:
        """Return the query in JSON format. [Deprecated, use `json_query` instead]"""

    @property
    @abstractmethod
    def json_query(self) -> dict | None:
        """Return the query in JSON format."""

    @property
    @abstractmethod
    def python_code(self) -> str | None:
        """Return the Python code to query the data source."""


DataSetConfigType = t.TypeVar("DataSetConfigType", bound=DataSetConfig)
DataSourceHandlerType = t.TypeVar("DataSourceHandlerType", bound='DataSourceHandler')


class DataSet(BaseEntity, t.Generic[DataSetConfigType, DataSourceHandlerType], ABC):
    _config: DataSetConfigType
    _datasource: DataSourceHandlerType

    def __init__(
        self,
        entity_id: str,
        title: str,
        config: DataSetConfigType,
        datasource: DataSourceHandlerType,
    ):
        BaseEntity.__init__(self)
        self._entity_id = entity_id
        self._title = title
        self._config = config
        self._datasource = datasource

    @abstractmethod
    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        raise NotImplementedError()

    @property
    def entity_type(self) -> EntityType:
        return EntityType.DATA_SET

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def name(self) -> str:
        return self._title

    @property
    def config(self) -> DataSetConfigType:
        return self._config

    @property
    @abstractmethod
    def status(self) -> Status:
        pass

    @property
    @abstractmethod
    def default_value_codes(self) -> list[str]:
        pass

    @abstractmethod
    def dimensions(self) -> t.Sequence[Dimension]:
        pass

    @abstractmethod
    def dimension(self, dimension_id: str) -> Dimension:
        pass

    @abstractmethod
    def attributes(self) -> t.Sequence[Attribute]:
        pass

    def non_virtual_dimensions(self) -> t.Sequence[Dimension]:
        return [dim for dim in self.dimensions() if not isinstance(dim, VirtualDimension)]

    @abstractmethod
    def non_indicator_dimensions(self) -> t.Sequence[Dimension]:
        pass

    @abstractmethod
    def special_dimensions(self) -> dict[str, Dimension]:
        pass

    @abstractmethod
    def indicator_dimensions(self) -> t.Sequence[Dimension]:
        pass

    @abstractmethod
    def indicator_dimensions_required_for_query(self) -> list[str]:
        pass

    @abstractmethod
    async def get_indicators(self, auth_context: AuthContext) -> t.Sequence[BaseIndicator]:
        pass

    @abstractmethod
    async def availability_query(
        self, query: DataSetAvailabilityQuery, auth_context: AuthContext
    ) -> DataSetAvailabilityQuery:
        pass

    @abstractmethod
    async def query(self, query: DataSetQuery, auth_context: AuthContext) -> DataResponse | None:
        pass


class OfflineDataSet(DataSet, t.Generic[DataSetConfigType, DataSourceHandlerType], ABC):
    """Class for cases where dataset loading failed"""

    def __init__(
        self,
        entity_id: str,
        title: str,
        config: DataSetConfigType,
        datasource: DataSourceHandlerType,
        status_details: str = "",
    ):
        super().__init__(entity_id, title, config, datasource)
        self._status_details = status_details

    @property
    def status(self) -> Status:
        return Status(status='offline', details=self._status_details)

    @property
    def default_value_codes(self) -> list[str]:
        return []

    def dimensions(self) -> list[Dimension]:
        return []

    def dimension(self, dimension_id: str) -> Dimension:
        raise RuntimeError("No dimensions for offline datasets")

    def attributes(self) -> list[Attribute]:
        return []

    def dimensions_by_concept_name(self, concept_name) -> list[Dimension]:
        return []

    def non_virtual_dimensions(self) -> list[Dimension]:
        return []

    def non_indicator_dimensions(self) -> list[Dimension]:
        return []

    def special_dimensions(self) -> dict[str, Dimension]:
        return {}

    def indicator_dimensions(self) -> list[Dimension]:
        return []

    def indicator_dimensions_required_for_query(self) -> list[str]:
        return []

    async def get_indicators(self, auth_context: AuthContext) -> t.Sequence[BaseIndicator]:
        return []

    async def availability_query(
        self, query: DataSetAvailabilityQuery, auth_context: AuthContext
    ) -> DataSetAvailabilityQuery:
        return query

    async def query(self, query: DataSetQuery, auth_context: AuthContext) -> DataResponse:
        raise RuntimeError("Query not supported for offline datasets")

    async def updated_at(self, auth_context: AuthContext | None) -> datetime | None:
        return None
