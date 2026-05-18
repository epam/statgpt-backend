from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DialRagSettings(BaseSettings):
    """
    DIAL RAG configuration settings
    """

    model_config = SettingsConfigDict(env_prefix="DIAL_RAG_")

    pgvector_url: str | None = Field(
        default=None, description="URL for remote DIAL RAG PGVector service"
    )

    pgvector_api_key: SecretStr | None = Field(
        default=None, description="API key for remote DIAL RAG PGVector service"
    )


# Create singleton instance
dial_rag_settings = DialRagSettings()
