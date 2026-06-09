import re
from datetime import datetime
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
    EXCLUDED = "excluded"
    """User selected a value valid in another dataset, but, given the other
    selected filters, no such value exists for this dimension here."""


class JsonComponentQuery(BaseYamlModel):
    component_code: str = Field(description="The id of the component")
    operator: JsonQueryOperator = Field(description="The operator of the query")
    values: list[str] = Field(description="The values of the query")


class JsonQueryMetadata(BaseYamlModel):
    country_dimension: str = Field(description="The country dimension id")
    indicator_dimensions: list[str] = Field(description="The indicator dimension ids")
    time_period_dimension: str = Field(
        description="The time period dimension id (must match the dataset DSD / filters)."
    )
    dataset_url: str | None = Field(default=None, description="URL of the dataset")
    key_dimension_ids_in_dsd_order: list[str] | None = Field(
        default=None,
        description=(
            "Non-time dimension ids in DSD order for SDMX 2.1 REST series keys; "
            "dimensions without a filter appear as empty key segments."
        ),
    )


class JsonQuery(BaseYamlModel):
    urn: str = Field(description="The urn of the dataset")
    filters: list[JsonComponentQuery] = Field(description="The list of component queries")

    _URN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<agency>[A-Za-z0-9_.-]+):(?P<resource>[A-Za-z0-9_.-]+)"
        r"\((?P<version>[A-Za-z0-9_.-]+)\)$"
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
    last_updated_at: datetime | None = Field(
        default=None, description="When the dataset's data was last updated"
    )

    @classmethod
    def from_query(
        cls,
        query: JsonQuery,
        metadata: JsonQueryMetadata,
        sdmx1_source: str | None = None,
        last_updated_at: datetime | None = None,
    ) -> "JsonQueryWithMetadata":
        return cls(
            urn=query.urn,
            filters=query.filters,
            metadata=metadata,
            sdmx1_source=sdmx1_source,
            last_updated_at=last_updated_at,
        )
