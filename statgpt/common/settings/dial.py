from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DialSettings(BaseSettings):
    """
    DIAL Core API connection settings
    """

    model_config = SettingsConfigDict(env_prefix="DIAL_")

    url: str = Field(
        default="http://localhost:8080",
        description="URL of the DIAL Core API where this app is deployed",
    )

    api_key: SecretStr = Field(default=SecretStr(""), description="API key for the DIAL Core API")


# Create singleton instance
dial_settings = DialSettings()
