import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from statgpt.common.auth import msi
from statgpt.common.settings.application import application_settings
from statgpt.common.settings.database import PostgresSettings
from statgpt.common.utils.value_tools import ValueUpdater

_log = logging.getLogger(__name__)


class Base(AsyncAttrs, DeclarativeBase):
    pass


metadata = Base.metadata  # for Alembic migrations


class DatabaseConnectionError(RuntimeError):
    pass


# The MSI token manager is used to store and update the MSI token for Postgres in the background.
_MSI_TOKEN_MANAGER: ValueUpdater[msi.MsiTokenResponse] | None = None


def _get_msi_token_manager() -> ValueUpdater[msi.MsiTokenResponse] | None:
    global _MSI_TOKEN_MANAGER
    if _MSI_TOKEN_MANAGER is None:
        postgres_settings = PostgresSettings()
        _MSI_TOKEN_MANAGER = ValueUpdater(
            msi.MsiGrant(msi.Config(scope=postgres_settings.msi_scope)).authorize,
            postgres_settings.msi_token_refresh_timeout,
        )
    return _MSI_TOKEN_MANAGER


def _track_session(session: AsyncSession) -> None:
    if application_settings.gc_debug:
        from statgpt.common.utils.gc_debug import gc_debugger

        session_id = id(session)
        gc_debugger.track_object(session, f"db_session_{session_id}")


@asynccontextmanager
async def optional_msi_token_manager_context():
    """Initializes MSI token manager if MSI is enabled."""
    postgres_settings = PostgresSettings()
    if postgres_settings.use_msi:
        _log.debug("Initializing MSI token manager")
        msi_token_manager = _get_msi_token_manager()
        await msi_token_manager.initialize()
        _log.debug("MSI token manager initialized successfully")
    try:
        yield
    finally:
        if postgres_settings.use_msi:
            _log.debug("Closing MSI token manager")
            msi_token_manager = _get_msi_token_manager()
            await msi_token_manager.close()
            _log.debug("MSI token manager closed")


class SessionMaker:
    DEFAULT_ENGINE_CONFIG = dict(
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
    )
    READONLY_ENGINE_CONFIG = dict(
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
        connect_args=dict(
            server_settings=dict(
                default_transaction_read_only="on"  # Enforce read-only mode at the session level
            )
        ),
    )

    def __init__(self, engine_config: dict):
        self._engine_config = engine_config
        self._postgres_settings = PostgresSettings()

    @staticmethod
    async def _test_connection(engine: AsyncEngine) -> bool:
        """Test if database connection is working"""
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            _log.debug(f"Connection test failed: {e}")
            return False

    async def _create_engine_with_retry(self, engine_factory, description: str) -> AsyncEngine:
        """Create engine with retry logic"""
        max_retries = self._postgres_settings.connection_max_retries
        retry_interval = self._postgres_settings.connection_retry_interval

        for attempt in range(max_retries):
            try:
                _log.info(
                    f"Attempting to create {description} engine (attempt {attempt + 1}/{max_retries})"
                )
                engine = await engine_factory()

                # Test the connection
                _log.debug("Testing database connection...")
                if await self._test_connection(engine):
                    _log.info(f"{description} engine created and connection verified")
                    return engine
                else:
                    _log.warning("Connection test failed, closing engine")
                    await engine.dispose()

            except Exception as e:
                _log.warning(
                    f"Failed to create {description} engine (attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                # Exponential backoff
                sleep_time = retry_interval * (2**attempt)
                _log.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)

        raise DatabaseConnectionError(f"Failed to connect to database after {max_retries} attempts")

    async def _create_default_engine(self) -> AsyncEngine:
        _log.debug(f"Creating default engine with config: {self._engine_config}")

        async def factory():
            return create_async_engine(
                self._postgres_settings.create_default_uri(), **self._engine_config
            )

        return await self._create_engine_with_retry(factory, "default")

    async def _create_msi_engine(self) -> AsyncEngine:
        _log.debug(f"Creating MSI engine with config: {self._engine_config}")

        msi_token_manager = _get_msi_token_manager()
        if msi_token_manager is None or not msi_token_manager.is_initialized:
            raise RuntimeError("Cannot create engine before MSI token manager is initialized")

        async def factory():
            engine = create_async_engine(
                self._postgres_settings.create_msi_uri(), **self._engine_config
            )

            # event not supported for async engine - provide token must be synchronous
            @event.listens_for(engine.sync_engine, "do_connect")
            def provide_token(dialect, conn_rec, cargs, cparams):
                _log.debug("Providing MSI token for database connection")
                cparams["password"] = msi_token_manager.value.access_token

            return engine

        return await self._create_engine_with_retry(factory, "MSI")

    async def create_engine(self) -> AsyncEngine:
        _log.debug(f"Creating engine (USE_MSI={self._postgres_settings.use_msi})")
        if self._postgres_settings.use_msi:
            engine = await self._create_msi_engine()
        else:
            engine = await self._create_default_engine()
        _log.debug(f"Engine created successfully: {engine}")
        return engine

    async def create(self) -> async_sessionmaker[AsyncSession]:
        _log.debug("Creating session maker")
        engine = await self.create_engine()
        session_maker = async_sessionmaker(
            engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        _log.debug(f"Session maker created: {session_maker}")
        return session_maker


class SessionMakerSingleton:
    instance: async_sessionmaker[AsyncSession] | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_or_create(cls) -> async_sessionmaker[AsyncSession]:
        if cls.instance is not None:
            _log.debug("Returning existing SessionMakerSingleton instance")
            return cls.instance

        async with cls._lock:
            # Double-check pattern: check again after acquiring lock
            if cls.instance is not None:
                _log.debug("Returning existing SessionMakerSingleton instance (after lock)")
                return cls.instance

            _log.debug("Creating new SessionMakerSingleton instance")
            cls.instance = await SessionMaker(SessionMaker.DEFAULT_ENGINE_CONFIG).create()
            return cls.instance


class ReadOnlySessionMakerSingleton:
    instance: async_sessionmaker[AsyncSession] | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_or_create(cls) -> async_sessionmaker[AsyncSession]:
        if cls.instance is not None:
            _log.debug("Returning existing ReadOnlySessionMakerSingleton instance")
            return cls.instance

        async with cls._lock:
            # Double-check pattern: check again after acquiring lock
            if cls.instance is not None:
                _log.debug("Returning existing ReadOnlySessionMakerSingleton instance (after lock)")
                return cls.instance

            _log.debug("Creating new ReadOnlySessionMakerSingleton instance")
            cls.instance = await SessionMaker(SessionMaker.READONLY_ENGINE_CONFIG).create()
            return cls.instance


# Dependency
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    _log.debug("get_session: Acquiring non-expiring session")
    session_maker = await SessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        session_id = id(session)
        _log.debug(f"get_session: Session opened (id={session_id}, expire_on_commit=False)")
        _track_session(session)
        try:
            yield session
        finally:
            _log.debug(f"get_session: Session closed (id={session_id})")


async def get_readonly_session() -> AsyncGenerator[AsyncSession, None]:
    _log.debug("get_readonly_session: Acquiring non-expiring read-only session")
    session_maker = await ReadOnlySessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        session_id = id(session)
        _track_session(session)
        _log.debug(
            f"get_readonly_session: Session opened (id={session_id}, expire_on_commit=False)"
        )
        try:
            yield session
        finally:
            _log.debug(f"get_readonly_session: Session closed (id={session_id})")


@asynccontextmanager
async def get_session_contex_manager() -> AsyncGenerator[AsyncSession, None]:
    _log.debug("get_session_contex_manager: Acquiring non-expiring session")
    session_maker = await SessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        session_id = id(session)
        _log.debug(
            f"get_session_contex_manager: Session opened (id={session_id}, expire_on_commit=False)"
        )
        _track_session(session)
        try:
            yield session
        finally:
            _log.debug(f"get_session_contex_manager: Session closed (id={session_id})")


@asynccontextmanager
async def get_readonly_session_contex_manager() -> AsyncGenerator[AsyncSession, None]:
    _log.debug("get_readonly_session_contex_manager: Acquiring non-expiring read-only session")
    session_maker = await ReadOnlySessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        session_id = id(session)
        _log.debug(
            f"get_readonly_session_contex_manager: Session opened (id={session_id}, expire_on_commit=False)"
        )
        _track_session(session)
        try:
            yield session
        finally:
            _log.debug(f"get_readonly_session_contex_manager: Session closed (id={session_id})")
