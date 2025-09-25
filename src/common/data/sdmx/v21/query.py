import typing as t
from enum import StrEnum

from pydantic import BaseModel, Field

from common.data.sdmx.common import UrnParseError, UrnParser
from common.schemas.base import BaseYamlModel


class SdmxQueryReadinessStatus(StrEnum):
    MISSING_REQUIRED_DIMENSIONS = "MISSING_REQUIRED_DIMENSIONS"
    READY = "READY"


class TimeDimensionQuery(BaseModel):
    time_dimension_id: str = Field(description="The id of the time dimension")
    start_period: t.Optional[str] = Field(description="The start period of the time dimension")
    end_period: t.Optional[str] = Field(description="The end period of the time dimension")


class SdmxDataSetQuery(BaseModel):
    # TODO: remove `datetime_dimensions` or `time_dimension_query` as they have the same purpose
    # See issue #124
    status: SdmxQueryReadinessStatus = Field(description="The readiness status of the query")
    categorical_dimensions: dict[str, list[str]] = Field(
        description="Categorical dimensions queries"
    )
    time_dimension_query: TimeDimensionQuery | None = Field(description="Time dimension query")
    missing_dimensions: list[str] = Field(description="Missing dimensions")

    def __contains__(self, item):
        return item in self.categorical_dimensions or (
            self.time_dimension_query is not None
            and item == self.time_dimension_query.time_dimension_id
        )

    @classmethod
    def empty(cls):
        return cls(
            status=SdmxQueryReadinessStatus.READY,
            categorical_dimensions={},
            time_dimension_query=None,
            missing_dimensions=[],
        )

    def get_key(self) -> dict[str, list[str]]:
        urn_parser = UrnParser.create_default()
        key = {}
        for dimension_id, category_values in self.categorical_dimensions.items():
            values = []
            for category_value in category_values:
                try:
                    parsed_urn = urn_parser.parse(category_value)
                    item = parsed_urn.item_id
                except UrnParseError:
                    item = category_value
                values.append(item)
            key[dimension_id] = values
        return key

    def get_params(self) -> dict:
        result = {
            "detail": "full",
        }
        if self.time_dimension_query:
            if self.time_dimension_query.start_period:
                result['startPeriod'] = self.time_dimension_query.start_period
            if self.time_dimension_query.end_period:
                result['endPeriod'] = self.time_dimension_query.end_period
        return result

    def merge(self, other: 'SdmxDataSetQuery') -> 'SdmxDataSetQuery':
        """Create a new query by merging two queries.

        NOTE: This method only applies to queries on the same dataset, but we do not verify it here.
        """

        merged_query = SdmxDataSetQuery(
            status=SdmxQueryReadinessStatus.READY,
            categorical_dimensions=self._merge_categorical_dimensions(other),
            time_dimension_query=self._merge_time_dimension_query(other),
            missing_dimensions=self._merge_missing_dimensions(other),
        )
        return merged_query

    def _merge_categorical_dimensions(self, other: 'SdmxDataSetQuery') -> dict[str, list[str]]:
        res = {}

        for dimension_id, values in self.categorical_dimensions.items():
            if dimension_id in other.categorical_dimensions:
                merged_values = set(values + other.categorical_dimensions[dimension_id])
                res[dimension_id] = list(merged_values)
            else:
                res[dimension_id] = values

        for dimension_id, values in other.categorical_dimensions.items():
            if dimension_id not in res:
                res[dimension_id] = values

        return res

    def _merge_time_dimension_query(self, other: 'SdmxDataSetQuery') -> TimeDimensionQuery | None:
        if self.time_dimension_query and other.time_dimension_query:
            start1 = self.time_dimension_query.start_period
            end1 = self.time_dimension_query.end_period
            start2 = other.time_dimension_query.start_period
            end2 = other.time_dimension_query.end_period

            return TimeDimensionQuery(
                time_dimension_id=self.time_dimension_query.time_dimension_id,
                start_period=min(start1, start2) if (start1 and start2) else (start1 or start2),
                end_period=max(end1, end2) if (end1 and end2) else (end1 or end2),
            )
        return self.time_dimension_query or other.time_dimension_query

    def _merge_missing_dimensions(self, other: 'SdmxDataSetQuery') -> list[str]:
        set1 = set(self.missing_dimensions)
        set2 = set(other.missing_dimensions)
        merged_missing = set1.intersection(set2)
        return list(merged_missing)


class SdmxDataSetAvailabilityQuery(BaseModel):
    time_dimension_query: t.Optional[TimeDimensionQuery] = Field(
        None, description="Time dimension query"
    )
    categorical_dimensions: t.Dict[str, t.Any] = Field(
        description="Categorical dimensions queries", default_factory=dict
    )
    datetime_dimensions: t.Dict[str, t.Any] = Field(
        description="Datetime dimensions queries", default_factory=dict
    )

    def get_key(self) -> dict[str, list[str]]:
        urn_parser = UrnParser.create_default()
        key = {}
        for dimension_id, category_values in self.categorical_dimensions.items():
            values = []
            for category_value in category_values:
                try:
                    parsed_urn = urn_parser.parse(category_value)
                    item = parsed_urn.item_id
                except UrnParseError:
                    item = category_value
                values.append(item)
            key[dimension_id] = values
        return key

    def get_params(self) -> dict[str, str]:
        result = {}
        if self.time_dimension_query:
            if self.time_dimension_query.start_period:
                result['startPeriod'] = self.time_dimension_query.start_period
            if self.time_dimension_query.end_period:
                result['endPeriod'] = self.time_dimension_query.end_period
        return result


class JsonQueryOperator(StrEnum):
    IN = "in"
    BETWEEN = "between"
    GE = "ge"
    """Greater than or equal"""
    LE = "le"
    """Less than or equal"""
    GT = "gt"
    """Greater than"""
    LT = "lt"
    """Less than"""


class JsonComponentQuery(BaseYamlModel):
    component_code: str = Field(description="The code of the component")
    operator: JsonQueryOperator = Field(description="The operator of the query")
    values: list[str] = Field(description="The values of the query")


class JsonQueryMetadata(BaseYamlModel):
    country_dimension: str = Field(description="The country dimension code")
    indicator_dimensions: list[str] = Field(description="The indicator dimension codes")


class JsonQuery(BaseYamlModel):
    urn: str = Field(description="The urn of the dataset")
    filters: list[JsonComponentQuery] = Field(description="The list of component queries")


class JsonQueryWithMetadata(JsonQuery):
    metadata: JsonQueryMetadata = Field(description="The metadata of the query")
