import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncConnection,
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
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine | None:
        """The engine built by `create()`, or None if `create()` has not been called yet."""
        return self._engine

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
        self._engine = engine
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
    _engine: AsyncEngine | None = None
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
            session_maker = SessionMaker(SessionMaker.DEFAULT_ENGINE_CONFIG)
            cls.instance = await session_maker.create()
            cls._engine = session_maker.engine
            return cls.instance

    @classmethod
    async def get_or_create_engine(cls) -> AsyncEngine:
        """Returns the engine backing the default sessions, creating it if needed."""
        await cls.get_or_create()
        if cls._engine is None:
            raise DatabaseConnectionError("Default engine is not available")
        return cls._engine


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
    """Yield a database session for use as a FastAPI dependency.

    WARNING: When used with FastAPI's Depends(), this session stays open until
    ALL BackgroundTasks complete — not just until the response is sent.
    For endpoints that schedule background tasks, use get_session_context_manager()
    instead to avoid holding connections for the duration of long-running tasks.
    """
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
async def get_session_context_manager(
    connection: AsyncConnection | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session.

    When `connection` is given, the session runs on that connection instead of checking one out
    of the pool. A connection-bound session keeps its connection across commits, which an
    engine-bound one does not -- see `advisory_lock_context_manager` for why that matters.
    The session configuration (`expire_on_commit=False`, `autoflush=False`) is the same either
    way, because both go through the shared session maker.
    """
    _log.debug("get_session_context_manager: Acquiring non-expiring session")
    session_maker = await SessionMakerSingleton.get_or_create()
    # Passing bind=None explicitly would override the engine the session maker is bound to.
    session_kwargs = {"bind": connection} if connection is not None else {}
    async with session_maker(**session_kwargs) as session:
        session_id = id(session)
        _log.debug(
            f"get_session_context_manager: Session opened (id={session_id}, expire_on_commit=False)"
        )
        _track_session(session)
        try:
            yield session
        finally:
            _log.debug(f"get_session_context_manager: Session closed (id={session_id})")


@asynccontextmanager
async def get_readonly_session_context_manager() -> AsyncGenerator[AsyncSession, None]:
    _log.debug("get_readonly_session_context_manager: Acquiring non-expiring read-only session")
    session_maker = await ReadOnlySessionMakerSingleton.get_or_create()
    async with session_maker() as session:
        session_id = id(session)
        _log.debug(
            f"get_readonly_session_context_manager: Session opened (id={session_id}, expire_on_commit=False)"
        )
        _track_session(session)
        try:
            yield session
        finally:
            _log.debug(f"get_readonly_session_context_manager: Session closed (id={session_id})")


@asynccontextmanager
async def get_connection_context_manager() -> AsyncGenerator[AsyncConnection, None]:
    """Yield a raw connection from the default engine, closing it on exit."""
    engine = await SessionMakerSingleton.get_or_create_engine()
    async with engine.connect() as connection:
        connection_id = id(connection)
        _log.debug(f"get_connection_context_manager: Connection opened (id={connection_id})")
        try:
            yield connection
        finally:
            _log.debug(f"get_connection_context_manager: Connection closed (id={connection_id})")


async def _try_acquire_advisory_lock(connection: AsyncConnection, lock_key: int) -> bool:
    """Attempts to take the advisory lock on `connection`. Opens and closes nothing."""
    acquired = await connection.scalar(
        text("SELECT pg_try_advisory_lock(:lock_key)"), {"lock_key": lock_key}
    )
    if not acquired:
        return False

    # End the transaction the statement above started: the connection then holds the lock while
    # idle rather than sitting 'idle in transaction', and a session bound to it afterwards
    # begins and owns its own transactions.
    await connection.commit()
    return True


async def _release_advisory_lock(connection: AsyncConnection, lock_key: int) -> None:
    """Releases an advisory lock held by `connection`.

    pg_advisory_unlock returns false rather than raising when the connection does not own the
    lock, so the result is checked explicitly -- an unnoticed failure would leave the lock held
    by a pooled connection and block every later operation on the same key. In that case, and on
    any error, the connection is invalidated: closing the socket for real makes PostgreSQL drop
    every advisory lock the backend held.

    Never raises, other than to propagate cancellation. Failures are only logged, because this
    runs on the cleanup path, where an exception would mask whatever failure is already
    propagating. Closing the connection is left to whoever opened it.
    """
    try:
        released = await connection.scalar(
            text("SELECT pg_advisory_unlock(:lock_key)"), {"lock_key": lock_key}
        )
        await connection.commit()
    except Exception:
        _log.exception(f"Failed to release advisory lock (key={lock_key}). Discarding connection.")
        await _invalidate_connection(connection)
        return
    except BaseException:
        # CancelledError (a background task timing out raises it) is not an `Exception`: without
        # this clause the connection would return to the pool with the unlock possibly never sent.
        _log.error(
            f"Release of advisory lock (key={lock_key}) was interrupted. Discarding connection."
        )
        await _invalidate_connection(connection)
        raise

    if not released:
        _log.error(
            f"Advisory lock (key={lock_key}) was not held by this connection. "
            f"Discarding the connection to make sure the lock is not leaked."
        )
        await _invalidate_connection(connection)
        return

    _log.debug(f"Released advisory lock (key={lock_key})")


async def _invalidate_connection(connection: AsyncConnection) -> None:
    """Discards a connection so PostgreSQL drops the locks it holds.

    Never raises, other than to propagate cancellation -- and even then the socket has already
    been force-closed by the driver, so the locks are gone regardless.
    """
    try:
        await connection.invalidate()
    except Exception:
        # invalidate() raises if the connection is already closed; there is nothing left to do.
        _log.exception("Failed to invalidate connection")


@asynccontextmanager
async def advisory_lock_context_manager(
    lock_key: int, timeout: float, description: str
) -> AsyncGenerator[AsyncConnection, None]:
    """Acquires a PostgreSQL session-level advisory lock, yielding the connection that owns it.

    A session-level advisory lock belongs to the PostgreSQL connection that took it, not to the
    AsyncSession that issued the statement. An AsyncSession bound to the engine returns its
    connection to the pool on every commit and checks out a possibly different one next, so a
    lock taken before a commit cannot be released after it: the unlock lands on another
    connection, silently returns false, and the lock stays held by an idle pooled connection
    until it is recycled. Owning one connection for the whole critical section is what keeps
    acquire and release on the same backend. Bind work to it with
    `get_session_context_manager(connection=...)`.

    Uses pg_try_advisory_lock in a polling loop rather than the blocking pg_advisory_lock, so a
    contended lock fails with TimeoutError instead of hanging indefinitely. The connection is
    opened per attempt and closed again before backing off, so no pooled connection is held
    while waiting.

    Raises TimeoutError if the lock cannot be acquired within `timeout` seconds.
    """
    current_interval = 0.1
    deadline = time.monotonic() + timeout

    while True:
        async with get_connection_context_manager() as connection:
            if await _try_acquire_advisory_lock(connection, lock_key):
                _log.debug(f"Acquired advisory lock for {description} (key={lock_key})")
                try:
                    yield connection
                finally:
                    await _release_advisory_lock(connection, lock_key)
                return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Could not acquire advisory lock for {description} "
                f"(key={lock_key}) within {timeout}s"
            )
        await asyncio.sleep(min(current_interval, remaining))
        current_interval = min(current_interval * 1.5, 5.0)
