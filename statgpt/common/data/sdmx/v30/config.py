from pydantic import Field

from statgpt.common.data.quanthub.config import QuanthubSdmxDataSourceConfig


class ProxySdmx30DataSourceConfig(QuanthubSdmxDataSourceConfig):
    """Configuration for SDMX 3.0 proxy data sources that still use sdmx1 parsing."""
    versions: set[str] = Field(
        default_factory=set,
        description="The versions of the SDMX standard supported by the data source",
    )
    headers: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="The headers for the data source"
    )