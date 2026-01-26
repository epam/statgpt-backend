from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Generic, TypeVar

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.dataset import Status
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

from .attribute import Attribute
from .base import BaseEntity, EntityType
from .config import DataSetConfig
from .dimension import Dimension
from .indicator import BaseIndicator
from .query import DataSetAvailabilityQuery, DataSetQuery

if TYPE_CHECKING:
    from statgpt.common.data.base.datasource import DataSourceHandler


class DataResponseStatus(BaseModel):
    request_status: DataRequestStatus = Field(description="status of the data request")
    parsing_status: DataParsingStatus = Field(description="status of data parsing")

    def merge(self, other: DataResponseStatus) -> DataResponseStatus:
        if (
            self.request_status == DataRequestStatus.FAILED
            and other.request_status == DataParsingStatus.FAILED
        ):
            request_status = DataRequestStatus.FAILED
        elif (
            self.request_status == DataRequestStatus.SUCCESS
            and other.request_status == DataRequestStatus.SUCCESS
        ):
            request_status = DataRequestStatus.SUCCESS
        else:
            request_status = DataRequestStatus.PARTIALLY_FAILED

        if (
            self.parsing_status == DataParsingStatus.FAILED
            and other.parsing_status == DataParsingStatus.FAILED
        ):
            parsing_status = DataParsingStatus.FAILED
        elif (
            self.parsing_status == DataParsingStatus.SUCCESS
            and other.parsing_status == DataParsingStatus.SUCCESS
        ):
            parsing_status = DataParsingStatus.SUCCESS
        elif self.parsing_status == DataParsingStatus.NA:
            parsing_status = other.parsing_status
        elif other.parsing_status == DataParsingStatus.NA:
            parsing_status = self.parsing_status
        else:
            parsing_status = DataParsingStatus.PARTIALLY_FAILED
        return DataResponseStatus(
            request_status=request_status,
            parsing_status=parsing_status,
        )


class DataResponse(ABC):
    """Base class for data responses from datasets."""

    @property
    @abstractmethod
    def status(self) -> DataResponseStatus:
        pass

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

    @property
    @abstractmethod
    def time_period(self) -> tuple[str, str] | None:
        """Return the time period covered by the data in this response as a tuple of (start, end)."""


DataSetConfigType = TypeVar("DataSetConfigType", bound=DataSetConfig)
DataSourceHandlerType = TypeVar("DataSourceHandlerType", bound='DataSourceHandler')


class DataSet(BaseEntity, Generic[DataSetConfigType, DataSourceHandlerType], ABC):
    _config: DataSetConfigType
    _datasource: DataSourceHandlerType

    def __init__(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: DataSetConfigType,
        datasource: DataSourceHandlerType,
    ):
        BaseEntity.__init__(self)
        self._entity_id = entity_id
        self._title = title
        self._config = config
        self._datasource = datasource

    @property
    def id(self) -> uuid.UUID:
        return self._entity_id

    @abstractmethod
    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        raise NotImplementedError()

    @property
    def entity_type(self) -> EntityType:
        return EntityType.DATA_SET

    @property
    def entity_id(self) -> str:
        return str(self._entity_id)

    @property
    @abstractmethod
    def source_id(self) -> str:
        """ID of the dataset in the source system."""

    @property
    def name(self) -> str:
        return self._title

    @property
    @abstractmethod
    def dataset_url(self) -> str | None:
        pass

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
    def dimensions(self) -> Sequence[Dimension]:
        pass

    @abstractmethod
    def dimension(self, dimension_id: str) -> Dimension:
        pass

    @abstractmethod
    def get_time_dimension(self) -> Dimension:
        pass

    @abstractmethod
    def get_frequency_dimension(self) -> Dimension:
        pass

    @abstractmethod
    def attributes(self) -> Sequence[Attribute]:
        pass

    @abstractmethod
    def non_indicator_dimensions(self) -> Sequence[Dimension]:
        pass

    @abstractmethod
    def special_dimensions(self) -> dict[str, Dimension]:
        pass

    @abstractmethod
    def indicator_dimensions(self, non_virtual: bool = False) -> Sequence[Dimension]:
        pass

    @abstractmethod
    def virtual_indicator_dimensions(self) -> Sequence[Dimension]:
        pass

    @property
    @abstractmethod
    def required_dimensions(self) -> list[str]:
        pass

    @abstractmethod
    async def get_indicators(
        self, auth_context: AuthContext, allow_cached: bool
    ) -> Sequence[BaseIndicator]:
        pass

    @abstractmethod
    async def availability_query(
        self, query: DataSetAvailabilityQuery, auth_context: AuthContext
    ) -> DataSetAvailabilityQuery:
        pass

    @abstractmethod
    async def query(self, query: DataSetQuery, auth_context: AuthContext) -> DataResponse | None:
        pass


class OfflineDataSet(DataSet, Generic[DataSetConfigType, DataSourceHandlerType], ABC):
    """Class for cases where dataset loading failed"""

    def __init__(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: DataSetConfigType,
        datasource: DataSourceHandlerType,
        status: Status,
    ):
        super().__init__(entity_id, title, config, datasource)
        self._status = status

    @property
    def status(self) -> Status:
        return self._status

    @property
    def default_value_codes(self) -> list[str]:
        return []

    @property
    def dataset_url(self) -> str | None:
        return None

    def dimensions(self) -> list[Dimension]:
        return []

    def dimension(self, dimension_id: str) -> Dimension:
        raise RuntimeError("No dimensions for offline datasets")

    def get_time_dimension(self) -> Dimension:
        raise RuntimeError("No dimensions for offline datasets")

    def get_frequency_dimension(self) -> Dimension:
        raise RuntimeError("No dimensions for offline datasets")

    def attributes(self) -> list[Attribute]:
        return []

    def dimensions_by_concept_name(self, concept_name) -> list[Dimension]:
        return []

    def non_indicator_dimensions(self) -> list[Dimension]:
        return []

    def special_dimensions(self) -> dict[str, Dimension]:
        return {}

    def indicator_dimensions(self, non_virtual: bool = False) -> list[Dimension]:
        return []

    def virtual_indicator_dimensions(self) -> list[Dimension]:
        return []

    @property
    def required_dimensions(self) -> list[str]:
        return []

    async def get_indicators(
        self, auth_context: AuthContext, allow_cached: bool
    ) -> Sequence[BaseIndicator]:
        return []

    async def availability_query(
        self, query: DataSetAvailabilityQuery, auth_context: AuthContext
    ) -> DataSetAvailabilityQuery:
        return query

    async def query(self, query: DataSetQuery, auth_context: AuthContext) -> DataResponse:
        raise RuntimeError("Query not supported for offline datasets")

    async def updated_at(self, auth_context: AuthContext | None) -> datetime | None:
        return None
