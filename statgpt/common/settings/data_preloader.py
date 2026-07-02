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
            "dataset caches warm (default is 0.8 x the 3600 s dataset cache TTL, so entries "
            "are replaced before they expire). Set to 0 to disable the refresh loop. "
            "Note: refreshed dataset objects may be rebuilt from still-cached SDMX structure "
            "messages (client cache TTL), so structural staleness is bounded by ~2x the "
            "structure cache TTL in the worst case."
        ),
    )


data_preloader_settings = DataPreloaderSettings()
