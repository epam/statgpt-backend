import os

from .utils import get_bool_env, get_int_env


class PostgresConfig:
    _HOST_VAR = "PGVECTOR_HOST"
    _PORT_VAR = "PGVECTOR_PORT"
    _DATABASE_VAR = "PGVECTOR_DATABASE"
    _USER_VAR = "PGVECTOR_USER"
    _PASSWORD_VAR = "PGVECTOR_PASSWORD"
    _MSI_SCOPE_VAR = "PGVECTOR_MSI_SCOPE"
    _MSI_TOKEN_REFRESH_TIMEOUT_VAR = "PGVECTOR_MSI_TOKEN_REFRESH_TIMEOUT"

    HOST = os.getenv(_HOST_VAR, "")
    PORT = os.getenv(_PORT_VAR, "5432")
    DATABASE = os.getenv(_DATABASE_VAR, "")
    USER = os.getenv(_USER_VAR, "")
    PASSWORD = os.getenv(_PASSWORD_VAR, "")

    USE_MSI: bool = get_bool_env("PGVECTOR_USE_MSI", False)  # type: ignore
    MSI_SCOPE = os.getenv(_MSI_SCOPE_VAR, "https://ossrdbms-aad.database.windows.net/.default")
    MSI_TOKEN_REFRESH_TIMEOUT: int = get_int_env(_MSI_TOKEN_REFRESH_TIMEOUT_VAR, 23 * 3600)  # type: ignore

    @classmethod
    def validate_default_config(cls):
        if cls.USE_MSI and not all(
            [
                cls.HOST,
                cls.PORT,
                cls.DATABASE,
                cls.USER,
                cls.MSI_SCOPE,
                cls.MSI_TOKEN_REFRESH_TIMEOUT,
            ]
        ):
            required_env_vars = [
                cls._HOST_VAR,
                cls._PORT_VAR,
                cls._DATABASE_VAR,
                cls._USER_VAR,
                cls._MSI_SCOPE_VAR,
                cls._MSI_TOKEN_REFRESH_TIMEOUT_VAR,
            ]
            raise ValueError(
                f"For MSI DB authentication, the following env vars must be configured: {', '.join(required_env_vars)}"
            )
        elif not all([cls.HOST, cls.PORT, cls.DATABASE, cls.USER, cls.PASSWORD]):
            required_env_vars = [
                cls._HOST_VAR,
                cls._PORT_VAR,
                cls._DATABASE_VAR,
                cls._USER_VAR,
                cls._PASSWORD_VAR,
            ]
            raise ValueError(
                f"For user/password DB authentication, the following env vars must be configured: {', '.join(required_env_vars)}"
            )

    @classmethod
    def create_default_uri(cls):
        return (
            f"postgresql+asyncpg://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"
        )

    @classmethod
    def validate_msi_config(cls):
        if not all([cls.HOST, cls.PORT, cls.DATABASE, cls.USER]):
            required_env_vars = [cls._HOST_VAR, cls._PORT_VAR, cls._DATABASE_VAR, cls._USER_VAR]
            raise ValueError(f"All env vars should be configured: {', '.join(required_env_vars)}.")

    @classmethod
    def create_msi_uri(cls):
        return f"postgresql+asyncpg://{cls.USER}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"


PG_VECTOR_STORE_BATCH_SIZE = int(os.environ.get("PGVECTOR_BATCH_SIZE", 1000))
