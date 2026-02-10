from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SdmxSettings(BaseSettings):
    """
    SDMX client and cache configuration
    """

    model_config = SettingsConfigDict(env_prefix="SDMX_")

    cache_dir: str | None = Field(default=None, description="Directory for SDMX cache")

    client_retry_count: int = Field(
        default=5, description="Maximum number of retries for SDMX client"
    )

    client_retry_delay: int = Field(default=3, description="Delay between retries in seconds")

    # Constants
    indicator_combinations_subdir: str = Field(
        default='available_indicator_combinations',
        description="Subdirectory name for indicator combinations",
    )


class QuantHubSettings(BaseSettings):
    """
    QuantHub specific settings
    """

    model_config = SettingsConfigDict(env_prefix="quanthub_")

    dataset_cache_ttl: int = Field(
        default=3600,
        description="Cache TTL for QuantHub datasets in seconds",
    )


class ProxySdmxSettings(BaseSettings):
    """
    Proxy SDMX specific settings
    """

    model_config = SettingsConfigDict(env_prefix="proxy_sdmx_")

    dataset_cache_ttl: int = Field(
        default=3600,
        description="Cache TTL for Proxy SDMX datasets in seconds",
    )


# Create singleton instances
sdmx_settings = SdmxSettings()
quanthub_settings = QuantHubSettings()
proxy_sdmx_settings = ProxySdmxSettings()
