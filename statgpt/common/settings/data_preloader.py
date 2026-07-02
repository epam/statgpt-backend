from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataPreloaderSettings(BaseSettings):
    """
    Settings for the dataset preloader (startup warm-up + periodic cache refresh)
    """

    model_config = SettingsConfigDict(env_prefix="data_preload_")

    refresh_interval_seconds: int = Field(
        default=2880,
        description=(
            "Interval in seconds between periodic dataset cache refreshes that keep the "
            "dataset caches warm. Keep it below the configured dataset cache TTLs "
            "(QUANTHUB_DATASET_CACHE_TTL, PROXY_SDMX_DATASET_CACHE_TTL; the default is "
            "0.8 x their 3600 s defaults), so entries are replaced before they expire — "
            "a larger interval re-opens the cold-reload window. Set to 0 to disable the "
            "refresh loop. Note: refreshed dataset objects may be rebuilt from still-cached "
            "SDMX structure messages (client cache TTL), so structural staleness is bounded "
            "by ~2x the structure cache TTL in the worst case."
        ),
    )


data_preloader_settings = DataPreloaderSettings()
