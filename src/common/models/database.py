from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from common.auth import msi
from common.config.database import PostgresConfig
from common.utils.value_tools import ValueUpdater


class Base(AsyncAttrs, DeclarativeBase):
    pass


metadata = Base.metadata  # for Alembic migrations

# The MSI token manager is used to store and update the MSI token for Postgres in the background.
msi_token_manager: ValueUpdater[msi.MsiTokenResponse] = ValueUpdater(
    msi.MsiGrant(msi.Config(scope=PostgresConfig.MSI_SCOPE)).authorize,
    PostgresConfig.MSI_TOKEN_REFRESH_TIMEOUT,
)


@asynccontextmanager
async def optional_msi_token_manager_context():
    """Initializes MSI token manager if MSI is enabled."""
    if PostgresConfig.USE_MSI:
        await msi_token_manager.initialize()
    try:
        yield
    finally:
        if PostgresConfig.USE_MSI:
            await msi_token_manager.close()


class SessionMaker:
    _DEFAULT_ENGINE_CONFIG = {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 300,
        "pool_pre_ping": True,  # connection may become invalid after token expiration
    }

    def __init__(self, engine_config: dict | None = None):
        self._engine_config = (
            engine_config if engine_config is not None else self._DEFAULT_ENGINE_CONFIG
        )

    async def _create_default_engine(self) -> AsyncEngine:
        PostgresConfig.validate_default_config()
        return create_async_engine(PostgresConfig.create_default_uri(), **self._engine_config)

    async def _create_msi_engine(self) -> AsyncEngine:
        PostgresConfig.validate_msi_config()
        engine = create_async_engine(PostgresConfig.create_msi_uri(), **self._engine_config)

        if not msi_token_manager.is_initialized:
            raise RuntimeError("Cannot create engine before MSI token manager is initialized")

        # event not supported for async engine - provide token must be synchronous
        @event.listens_for(engine.sync_engine, "do_connect")
        def provide_token(dialect, conn_rec, cargs, cparams):
            cparams["password"] = msi_token_manager.value.access_token

        return engine

    async def create_engine(self) -> AsyncEngine:
        if PostgresConfig.USE_MSI:
            engine = await self._create_msi_engine()
        else:
            engine = await self._create_default_engine()
        return engine

    async def create(self) -> async_sessionmaker[AsyncSession]:
        engine = await self.create_engine()
        return async_sessionmaker(engine, expire_on_commit=False)


class SessionMakerSingleton:
    instance = None

    @classmethod
    async def get_or_create(
        cls, engine_config: dict | None = None
    ) -> async_sessionmaker[AsyncSession]:
        if not cls.instance:
            cls.instance = await SessionMaker(engine_config).create()
        return cls.instance


# Dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    session_maker = await SessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        yield session


@asynccontextmanager
async def get_session_contex_manager() -> AsyncGenerator[AsyncSession, None]:
    session_maker = await SessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        yield session
