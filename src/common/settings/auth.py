from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """
    General authentication settings
    """

    model_config = SettingsConfigDict(env_prefix="")

    token_expiration_buffer: int = Field(
        default=120,
        alias="TOKEN_EXPIRATION_BUFFER",
        description="Buffer time in seconds before token expiration",
    )


class DialChatSettings(BaseSettings):
    """
    Dial Chat OAuth2 settings
    """

    model_config = SettingsConfigDict(env_prefix="")

    oauth2_token_endpoint_url: str = Field(
        default="", alias="OAUTH2_TOKEN_ENDPOINT_URL", description="OAuth2 token endpoint URL"
    )

    scope: str = Field(
        default="", alias="SERVICES_CHAT_SCOPE", description="OAuth2 scope for services chat"
    )

    client_id: str = Field(
        default="", alias="SERVICES_CHAT_CLIENT_ID", description="Client ID for services chat"
    )

    client_secret: str = Field(
        default="",
        alias="SERVICES_CHAT_CLIENT_SECRET",
        description="Client secret for services chat",
    )


class ClientsSpaChatSettings(BaseSettings):
    """
    Clients SPA Chat OAuth2 settings
    """

    model_config = SettingsConfigDict(env_prefix="")

    oauth2_token_endpoint_url: str = Field(
        default="", alias="OAUTH2_TOKEN_ENDPOINT_URL", description="OAuth2 token endpoint URL"
    )

    client_id: str = Field(
        default="", alias="CLIENTS_SPA_CHAT_CLIENT_ID", description="Client ID for SPA chat"
    )

    client_secret: str = Field(
        default="", alias="CLIENTS_SPA_CHAT_CLIENT_SECRET", description="Client secret for SPA chat"
    )


# Create singleton instances
auth_settings = AuthSettings()
dial_chat_settings = DialChatSettings()
clients_spa_chat_settings = ClientsSpaChatSettings()
