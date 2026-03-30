import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.common.models import get_session_context_manager
from statgpt.common.models.database import get_readonly_session_context_manager


class DbServiceBase:
    def __init__(
        self,
        session: AsyncSession | None = None,
        session_lock: asyncio.Lock | None = None,
    ) -> None:
        self.__session = session
        self._scoped_session_active = False
        if session_lock is None:
            self._session_lock = asyncio.Lock()
        else:
            self._session_lock = session_lock

    @property
    def _session(self) -> AsyncSession:
        """Return the current session, raising if none is set.

        Always valid inside a ``_scoped_session()`` block or when the
        service was constructed with a real session.
        """
        if self.__session is None:
            raise RuntimeError(
                "Session is not available. "
                "Use _scoped_session() or provide a session at construction time."
            )
        return self.__session

    @asynccontextmanager
    async def _lock_session(self) -> AsyncIterator[AsyncSession]:
        """Acquire lock and yield session for thread-safe operations."""
        async with self._session_lock:
            yield self._session

    @asynccontextmanager
    async def _scoped_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a short-lived session.

        When no session was provided at construction time, creates a new
        short-lived session via ``get_session_context_manager()`` and
        temporarily sets self._session so that downstream code
        (e.g. VectorStoreFactory) works transparently.
        When a session was provided, yields it directly.
        """
        if self._scoped_session_active:
            raise RuntimeError(
                "Concurrent _scoped_session() calls on the same instance "
                "are not supported. Create a separate service instance per "
                "concurrent coroutine, or provide a session at construction "
                "time."
            )
        if self.__session is not None:
            yield self._session
        else:
            self._scoped_session_active = True
            async with get_session_context_manager() as session:
                self.__session = session
                try:
                    yield session
                finally:
                    self.__session = None
                    self._scoped_session_active = False

    @asynccontextmanager
    async def _scoped_readonly_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a short-lived readonly session.

        Same as ``_scoped_session()`` but creates a readonly session via
        ``get_readonly_session_context_manager()`` when none is provided.
        """
        if self._scoped_session_active:
            raise RuntimeError(
                "Concurrent _scoped_readonly_session() calls on the same "
                "instance are not supported. Create a separate service "
                "instance per concurrent coroutine, or provide a session "
                "at construction time."
            )
        if self.__session is not None:
            yield self._session
        else:
            self._scoped_session_active = True
            async with get_readonly_session_context_manager() as session:
                self.__session = session
                try:
                    yield session
                finally:
                    self.__session = None
                    self._scoped_session_active = False
