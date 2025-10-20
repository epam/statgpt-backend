from enum import StrEnum

from pydantic import Field

from .base import BaseYamlModel


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
    dataset_url: str | None = Field(default=None, description="URL of the dataset")


class JsonQuery(BaseYamlModel):
    urn: str = Field(description="The urn of the dataset")
    filters: list[JsonComponentQuery] = Field(description="The list of component queries")


class JsonQueryWithMetadata(JsonQuery):
    metadata: JsonQueryMetadata = Field(description="The metadata of the query")

    @classmethod
    def from_query(cls, query: JsonQuery, metadata: JsonQueryMetadata) -> "JsonQueryWithMetadata":
        return cls(urn=query.urn, filters=query.filters, metadata=metadata)
