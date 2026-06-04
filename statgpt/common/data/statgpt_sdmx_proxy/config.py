from pydantic import model_validator

from statgpt.common.data.sdmx.common.config import ProviderDiscoveryMode, SdmxDataSourceConfig


class StatGptSdmxProxyDataSourceConfig(SdmxDataSourceConfig):
    """Configuration for StatGPT SDMX proxy data sources (SDMX 3.0 API, parsed as SDMX 2.1)."""

    @model_validator(mode='after')
    def _validate_provider_discovery(self) -> 'StatGptSdmxProxyDataSourceConfig':
        if self.provider_discovery is ProviderDiscoveryMode.DATAFLOWS:
            raise ValueError(
                "provider_discovery='dataflows' is not supported for StatGPT SDMX proxy data "
                "sources: the proxy cannot list dataflows across all agencies. Use 'agencyscheme'."
            )
        return self
