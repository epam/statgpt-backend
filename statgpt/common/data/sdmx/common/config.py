import logging
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from statgpt.common.config import utils as config_utils
from statgpt.common.data.base import (
    BaseModel,
    DataSetConfig,
    DataSetConfigTemplate,
    DataSetHierarchyConfig,
    DataSourceConfig,
)
from statgpt.common.data.sdmx.common.data_explorer_url import DataExplorerUrlConfig

_log = logging.getLogger(__name__)


class ProviderDiscoveryMode(StrEnum):
    """Strategy for enumerating providers (maintainer agencies) exposed by a data source."""

    AGENCYSCHEME = "agencyscheme"
    DATAFLOWS = "dataflows"


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
    versions: set[str] = Field(default_factory=lambda: {"2.1"})

    def get_url(self) -> str:
        """Return the URL for the SDMX data source, replacing environment variables if necessary."""
        return config_utils.replace_env(self.url)

    def to_sdmx1_dict(self) -> dict[str, Any]:
        """Convert the SdmxConfig to a dictionary suitable for creating an `sdmx1` data source."""
        config_dict = self.model_dump()
        config_dict["url"] = self.get_url()
        return config_dict


class SdmxRateLimitsConfig(BaseModel):

    structure_requests_concurrency: str | None = Field(
        default=None, description="The maximum number of concurrent structure requests"
    )
    availability_and_data_requests_concurrency: str | None = Field(
        default=None, description="The maximum number of concurrent availability and data requests"
    )

    def get_structure_requests_concurrency(self) -> int | None:
        if self.structure_requests_concurrency is None:
            return None

        try:
            limit_str = config_utils.replace_env(self.structure_requests_concurrency)
            limit = int(limit_str)
            if limit <= 0:
                raise ValueError(f"Concurrency limit must be positive, got: {limit}")
            return limit
        except (ValueError, TypeError) as e:
            _log.warning(
                f"Invalid structure_requests_concurrency '{self.structure_requests_concurrency}': {e}. "
                f"Returning None (no limit)"
            )
            return None

    def get_availability_and_data_requests_concurrency(self) -> int | None:
        if self.availability_and_data_requests_concurrency is None:
            return None

        try:
            limit_str = config_utils.replace_env(self.availability_and_data_requests_concurrency)
            limit = int(limit_str)
            if limit <= 0:
                raise ValueError(f"Concurrency limit must be positive, got: {limit}")
            return limit
        except (ValueError, TypeError) as e:
            _log.warning(
                f"Invalid availability_and_data_requests_concurrency '{self.availability_and_data_requests_concurrency}': {e}. "
                f"Returning None (no limit)"
            )
            return None


class DefaultDataSetHierarchyConfig(DataSetHierarchyConfig):
    type: Literal['config']


class CategorySchemaDataSetHierarchyConfig(BaseModel):
    """Configuration of a dataset hierarchy using the SDMX category scheme as the source of the hierarchy."""

    type: Literal['category_scheme']

    # The URN of the category scheme:
    agency_id: str
    resource_id: str
    version: str = Field(default='latest')


class UrnReference(BaseModel):
    """This class represents a URN reference for SDMX datasets and allows dynamic values."""

    agency_id: str
    resource_id: str
    version: str = Field(default='latest')

    def short_urn(self) -> str:
        """Return a short URN representation."""
        return f"{self.agency_id}:{self.resource_id}({self.version})"

    def __repr__(self) -> str:
        return f"UrnReference(agency_id={self.agency_id!r}, resource_id={self.resource_id!r}, version={self.version!r})"

    def __str__(self) -> str:
        return self.short_urn()


class SdmxDataSourceConfig(DataSourceConfig):
    description: str = Field(default="", description="The description of the data source")
    sdmx_config: SdmxConfig = Field(description="The configuration for the SDMX data source")
    locale: str = Field(default="en", description="Locale")
    default_value_codes: list[str] = Field(
        default_factory=list,
        description=(
            "List of default value codes to use when a dimension is not specified in a query. "
            "E.g. in SDMX standard codes are `_Z` - Not Applicable/Not Available, `_T` - Total, etc."
        ),
    )
    sdmx1_source: str | None = Field(
        default=None, description="The name of the sdmx1 source to use in python code (if any)"
    )
    rate_limits: SdmxRateLimitsConfig = Field(
        default_factory=SdmxRateLimitsConfig,
        description="The rate limits configuration for the SDMX data source",
    )
    data_explorer_url: str | None = Field(
        default=None,
        description=(
            "Default base URL of the data explorer UI. Used when a dataset does not set "
            "data_explorer_url, for example to build preset explorer links instead of raw API URLs. "
            "When set, query-result URLs always use this explorer link. Whether dataset-level "
            "`dataset_url` also uses it is controlled by `use_data_explorer_for_dataset_url`."
        ),
    )
    use_data_explorer_for_dataset_url: bool = Field(
        default=False,
        description=(
            "If true and a data explorer URL is resolved (dataset or data source), dataset URLs "
            "will point to the data explorer instead of citation URLs."
        ),
    )
    data_explorer_url_config: DataExplorerUrlConfig | None = Field(
        default=None,
        description=(
            "Default rules for building data-explorer query links and dataset-only explorer URLs. "
            "Overridden by the same field on a dataset when set."
        ),
    )
    dataset_hierarchy: (
        DefaultDataSetHierarchyConfig | CategorySchemaDataSetHierarchyConfig | None
    ) = Field(
        default=None,
        discriminator='type',
        description="The configuration of the dataset hierarchy for this data source",
    )
    provider_discovery: ProviderDiscoveryMode = Field(
        default=ProviderDiscoveryMode.AGENCYSCHEME,
        description=(
            "How to enumerate providers (maintainer agencies) exposed by this data source. "
            "`agencyscheme` queries the SDMX agencyscheme endpoint and returns localized names. "
            "`dataflows` lists all dataflows and groups them by agency_id (ids only, no display "
            "names). Use `dataflows` for SDMX 2.1 registries that don't expose agency schemes."
        ),
    )

    def get_data_explorer_url(self) -> str | None:
        if self.data_explorer_url:
            return config_utils.replace_env(self.data_explorer_url)
        return None

    def get_id(self) -> str:
        return self.sdmx_config.id

    def get_name(self) -> str:
        return self.sdmx_config.name


class SdmxDataSetConfigMixin:
    use_title_from_src: bool = Field(
        default=False, description="Whether to use the title obtained from the source"
    )
    urn: UrnReference = Field(description="The URN of the dataset")
    include_attributes: list[str] | None = Field(
        default=None, description="List of attributes to add to the query results table"
    )
    default_value_codes: list[str] | None = Field(
        default=None,
        description=(
            "List of default value codes to use when a dimension is not specified in a query. "
            "E.g. in SDMX standard codes are `_Z` - Not Applicable/Not Available, `_T` - Total, etc. "
            "If not set, the default value codes from the data source configuration will be used. "
            "If set to an empty list, no default value codes will be used."
        ),
    )
    sdmx1_source: str | None = Field(
        default=None,
        description=(
            "Optional sdmx1 library source id for generated Python attachment code for this dataset. "
            "When unset, the data source `sdmx1_source` is used; if that is also unset, the dataflow maintainer id is used."
        ),
    )
    data_explorer_url_config: DataExplorerUrlConfig | None = Field(
        default=None,
        description=(
            "How to build data-explorer links for this dataset. When unset, the data source default "
            "is used; if that is also unset, the legacy format applies (urn + filter + time parameters)."
        ),
    )


class SdmxDataSetConfigTemplate(SdmxDataSetConfigMixin, DataSetConfigTemplate):
    pass


class SdmxDataSetConfig(SdmxDataSetConfigMixin, DataSetConfig):
    pass
