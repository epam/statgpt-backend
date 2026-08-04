from typing import Any

from pydantic import Field, model_validator

from statgpt.common.config import utils as config_utils
from statgpt.common.data.base.datasource import DataSourceConfig
from statgpt.common.data.sdmx.common.config import ProviderDiscoveryMode, SdmxDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.config_client import (
    fetch_proxy_config,
    push_proxy_config,
)

# Serialization alias of `StatGptSdmxProxyDataSourceConfig.proxy_config`: `details` is stored
# camelCase, so this is the key the admin portal sees and sends back.
PROXY_CONFIG_KEY = "proxyConfig"


class StatGptSdmxProxyDataSourceConfig(SdmxDataSourceConfig):
    """Configuration for StatGPT SDMX proxy data sources (SDMX 3.0 API, parsed as SDMX 2.1)."""

    config_url: str = Field(
        default="$env:{SDMX_PROXY_CONFIG_SERVER_HOST}/statgpt/sdmx-proxy-config-server/api/v0/config",
        description="The URL of the StatGPT SDMX proxy configuration server.",
    )
    proxy_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The proxy configuration served by the config server at `config_url`. "
            "It is fetched when the data source is read and pushed back when the data source "
            "is created or updated. The config server owns this value: it is never stored in "
            "the database."
        ),
    )

    @model_validator(mode='after')
    def _validate_provider_discovery(self) -> 'StatGptSdmxProxyDataSourceConfig':
        if self.provider_discovery is ProviderDiscoveryMode.DATAFLOWS:
            raise ValueError(
                "provider_discovery='dataflows' is not supported for StatGPT SDMX proxy data "
                "sources: the proxy cannot list dataflows across all agencies. Use 'agencyscheme'."
            )
        return self

    def get_config_url(self) -> str:
        return config_utils.replace_env(self.config_url)

    def dump_for_storage(self) -> dict[str, Any]:
        return self.model_dump(mode='json', by_alias=True, exclude={'proxy_config'})

    def matches_stored(self, stored: DataSourceConfig) -> bool:
        if self.dump_for_storage() != stored.dump_for_storage():
            return False
        if self.proxy_config is None:
            # The incoming config leaves the proxy configuration to the config server, so the
            # value currently served by it is not a difference.
            return True
        return (
            isinstance(stored, StatGptSdmxProxyDataSourceConfig)
            and self.proxy_config == stored.proxy_config
        )

    def external_details_key(self) -> str | None:
        return self.get_config_url()

    async def load_external_details(self) -> dict[str, Any]:
        return {PROXY_CONFIG_KEY: await fetch_proxy_config(self.get_config_url())}

    async def push_external_details(self) -> dict[str, Any] | None:
        if self.proxy_config is None:
            # The config server owns the value, so an update that does not carry one leaves it be.
            return None
        stored = await push_proxy_config(self.get_config_url(), self.proxy_config)
        return {PROXY_CONFIG_KEY: stored}
