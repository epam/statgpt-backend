import re
from enum import StrEnum
from functools import cached_property
from typing import ClassVar

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
        description=(
            "The values of the query. A single string is treated as one value "
            "(not split on commas)."
        )
    )

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: object) -> list[str]:
        if isinstance(v, str):
            stripped = v.strip()
            return [] if not stripped else [stripped]
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v]
        raise ValueError(f"values must be a str or list[str]; got {type(v).__name__}")


class JsonQueryMetadata(BaseYamlModel):
    country_dimension: str = Field(description="The country dimension code")
    indicator_dimensions: list[str] = Field(description="The indicator dimension codes")
    time_period_dimension: str = Field(
        description="The time period dimension code (must match the dataset DSD / filters)."
    )
    dataset_url: str | None = Field(default=None, description="URL of the dataset")
    rest_key_dimension_codes: list[str] | None = Field(
        default=None,
        description=(
            "Non-time dimension codes in DSD order for SDMX 2.1 REST series keys; "
            "dimensions without a filter appear as empty key segments."
        ),
    )


class JsonQuery(BaseYamlModel):
    urn: str = Field(description="The urn of the dataset")
    filters: list[JsonComponentQuery] = Field(description="The list of component queries")

    _URN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<agency>[^:]+):(?P<resource>[^(]+)\((?P<version>[^)]+)\)$"
    )

    @field_validator("urn")
    @classmethod
    def _validate_urn(cls, value: str) -> str:
        if not cls._URN_PATTERN.match(value):
            raise ValueError("URN must match 'AGENCY:RESOURCE(VERSION)' format")
        return value

    @cached_property
    def _urn_parts(self) -> tuple[str, str, str]:
        agency, _, tail = self.urn.partition(":")
        open_paren = tail.index("(")
        return agency, tail[:open_paren], tail[open_paren + 1 : -1]

    @property
    def agency_id(self) -> str:
        return self._urn_parts[0]

    @property
    def resource_id(self) -> str:
        return self._urn_parts[1]

    @property
    def version(self) -> str:
        return self._urn_parts[2]


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
