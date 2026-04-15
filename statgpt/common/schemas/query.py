import re
from enum import StrEnum

from pydantic import Field, field_validator

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
    values: list[str] = Field(
        description="The values of the query (also accepts a comma-separated string)"
    )

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return v.split(",")
        return v  # type: ignore[return-value]


class JsonQueryMetadata(BaseYamlModel):
    country_dimension: str = Field(description="The country dimension code")
    indicator_dimensions: list[str] = Field(description="The indicator dimension codes")
    time_period_dimension: str = Field(
        default="TIME_PERIOD",
        description="The time period dimension code",
    )
    dataset_url: str | None = Field(default=None, description="URL of the dataset")


class JsonQuery(BaseYamlModel):
    urn: str = Field(description="The urn of the dataset")
    filters: list[JsonComponentQuery] = Field(description="The list of component queries")

    _URN_PATTERN = re.compile(r"^(?P<agency>[^:]+):(?P<resource>[^(]+)\((?P<version>[^)]+)\)$")

    @field_validator("urn")
    @classmethod
    def _validate_urn(cls, value: str) -> str:
        if not cls._URN_PATTERN.match(value):
            raise ValueError("URN must match 'AGENCY:RESOURCE(VERSION)' format")
        return value


class JsonQueryWithMetadata(JsonQuery):
    metadata: JsonQueryMetadata = Field(description="The metadata of the query")
    sdmx1_source: str | None = Field(default=None, description="The sdmx1 library source id")

    @classmethod
    def from_query(
        cls,
        query: JsonQuery,
        metadata: JsonQueryMetadata,
        sdmx1_source: str | None = None,
    ) -> "JsonQueryWithMetadata":
        return cls(
            urn=query.urn,
            filters=query.filters,
            metadata=metadata,
            sdmx1_source=sdmx1_source,
        )
