from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """
    PostgreSQL database connection settings
    """

    model_config = SettingsConfigDict(env_prefix="PGVECTOR_")

    host: str = Field(default="", description="PostgreSQL host")

    port: str = Field(default="5432", description="PostgreSQL port")

    database: str = Field(default="", description="PostgreSQL database name")

    user: str = Field(default="", description="PostgreSQL user")

    password: str = Field(default="", description="PostgreSQL password")

    use_msi: bool = Field(
        default=False, description="Use Managed Service Identity for authentication"
    )

    msi_scope: str = Field(
        default="https://ossrdbms-aad.database.windows.net/.default",
        description="MSI scope for Azure PostgreSQL",
    )

    msi_token_refresh_timeout: int = Field(
        default=23 * 3600, description="MSI token refresh timeout in seconds"
    )

    batch_size: int = Field(default=1000, description="Batch size for vector store operations")

    @model_validator(mode="after")
    def validate_config(self):
        """Validate that required fields are present based on authentication type"""
        if self.use_msi:
            if not all(
                [
                    self.host,
                    self.port,
                    self.database,
                    self.user,
                    self.msi_scope,
                    self.msi_token_refresh_timeout,
                ]
            ):
                raise ValueError(
                    "For MSI DB authentication, the following env vars must be configured: "
                    "PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DATABASE, PGVECTOR_USER, "
                    "PGVECTOR_MSI_SCOPE, PGVECTOR_MSI_TOKEN_REFRESH_TIMEOUT"
                )
        else:
            if not all([self.host, self.port, self.database, self.user, self.password]):
                raise ValueError(
                    "For user/password DB authentication, the following env vars must be configured: "
                    "PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DATABASE, PGVECTOR_USER, PGVECTOR_PASSWORD"
                )
        return self

    def create_default_uri(self) -> str:
        """Create PostgreSQL connection URI for standard authentication"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def create_msi_uri(self) -> str:
        """Create PostgreSQL connection URI for MSI authentication"""
        return f"postgresql+asyncpg://{self.user}@{self.host}:{self.port}/{self.database}"
