from pydantic import Field

from statgpt.common.data.sdmx.common.config import SdmxDataSourceConfig


class StatGptSdmxProxyDataSourceConfig(SdmxDataSourceConfig):
    """Configuration for StatGPT SDMX proxy data sources (SDMX 3.0 API, parsed as SDMX 2.1)."""

    versions: set[str] = Field(
        default_factory=set,
        description="The versions of the SDMX standard supported by the data source",
    )
    headers: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="The headers for the data source"
    )
