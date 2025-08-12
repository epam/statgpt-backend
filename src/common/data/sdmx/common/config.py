from typing import Any

from pydantic import BaseModel, ConfigDict, Field, alias_generators

from common.config import utils as config_utils
from common.data.base import DataSetConfig, DataSourceConfig


class SdmxHeaders(BaseModel):
    agencyscheme: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    availableconstraint: dict[str, str] = Field(
        default_factory=lambda: {"accept": "application/xml"}
    )
    categoryscheme: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    codelist: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    conceptscheme: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    data: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    dataflow: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    datastructure: dict[str, str] = Field(default_factory=lambda: {"accept": "application/xml"})
    hierarchicalcodelist: dict[str, str] = Field(
        default_factory=lambda: {"accept": "application/xml"}
    )
    provisionagreement: dict[str, str] = Field(
        default_factory=lambda: {"accept": "application/xml"}
    )


class SdmxSupport(BaseModel):
    agencyscheme: bool = Field(default=True)
    availableconstraint: bool = Field(default=True)
    categoryscheme: bool = Field(default=True)
    codelist: bool = Field(default=True)
    conceptscheme: bool = Field(default=True)
    data: bool = Field(default=True)
    dataflow: bool = Field(default=True)
    datastructure: bool = Field(default=True)
    hierarchicalcodelist: bool = Field(default=True)
    preview: bool = Field(default=True)
    provisionagreement: bool = Field(default=True)


class SdmxConfig(BaseModel):
    id: str = Field()
    data_content_type: str = Field(default="JSON")
    url: str = Field()
    name: str = Field()
    headers: SdmxHeaders = Field(default_factory=SdmxHeaders)
    supports: SdmxSupport = Field(default_factory=SdmxSupport)

    def get_url(self) -> str:
        """Return the URL for the SDMX data source, replacing environment variables if necessary."""
        return config_utils.replace_env(self.url)

    def to_sdmx1_dict(self) -> dict[str, Any]:
        """Convert the SdmxConfig to a dictionary suitable for creating an `sdmx1` data source."""
        config_dict = self.model_dump()
        config_dict["url"] = self.get_url()
        return config_dict


class SdmxDataSourceConfig(DataSourceConfig):
    description: str = Field(default="", description="The description of the data source")
    sdmx_config: SdmxConfig = Field(description="The configuration for the SDMX data source")
    locale: str = Field(default="en", description="Locale")

    def get_id(self) -> str:
        return self.sdmx_config.id

    def get_name(self) -> str:
        return self.sdmx_config.name


class FixedItem(BaseModel):
    # TODO: probably use other existing abstraction
    id: str
    name: str
    description: str | None


class SdmxDataSetConfig(DataSetConfig):
    use_title_from_src: bool = Field(
        default=False, description="Whether to use the title obtained from the source"
    )
    urn: str = Field(description="The URN of the dataset")
    indicator_dimensions: list[str] = Field(description="The list of indicator dimensions")
    indicator_dimensions_required_for_query: list[str] = Field(
        default=[],
        description=(
            "The list of indicator dimensions required to build a query. "
            "Used to filter out queries without these dimensions. See the detailed logic in the code"
        ),
    )
    country_dimension: str | None = Field(None, description="The main country dimension")
    country_dimension_alias: str | None = Field(
        None, description="The alias of the main country dimension"
    )
    dimension_all_values: dict[str, FixedItem] = Field(
        default_factory=dict,
        description=(
            "Dictionary of special dimension values - 'All-values' which are used to set '*' filter in the query "
            "for the dimension. Keys are dimension IDs, values are FixedItem objects."
        ),
    )
    fixed_indicator: FixedItem | None = Field(default=None)
    include_attributes: list[str] | None = Field(
        default=None, description="List of attributes to add to the query results table"
    )

    def get_source_id(self) -> str:
        return self.urn

    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)
