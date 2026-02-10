from pydantic import Field

from statgpt.common.data.sdmx.common.config import SdmxDataSourceConfig


class ProxySdmx30DataSourceConfig(SdmxDataSourceConfig):
    """Configuration for SDMX 3.0 proxy data sources that still use sdmx1 parsing."""

    versions: set[str] = Field(
        default_factory=set,
        description="The versions of the SDMX standard supported by the data source",
    )
    headers: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="The headers for the data source"
    )
