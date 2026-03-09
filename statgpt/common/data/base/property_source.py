from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, alias_generators


class PropertySourceEnum(StrEnum):
    ANNOTATION = "annotation"
    """Retrieve the property from an SDMX annotation."""
    ATTRIBUTE = "attribute"
    """Retrieve the property from an SDMX attribute."""
    CITATION = "citation"
    """Retrieve the property from the dataset citation configuration."""
    VALUE = "value"
    """Use a fixed value specified in the `field`."""


class PropertySource(BaseModel):
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel, populate_by_name=True, extra="allow"
    )

    source: PropertySourceEnum = Field()
    field: str = Field(description="The field name in the source")
    formats: list[str] | None = Field(
        default=None, description="The list of non-default formats to try when parsing the value"
    )
