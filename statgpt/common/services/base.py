import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession


class DbServiceBase:
    def __init__(self, session: AsyncSession, session_lock: asyncio.Lock | None = None) -> None:
        self._session = session
        if session_lock is None:
            self._session_lock = asyncio.Lock()
        else:
            self._session_lock = session_lock

    @asynccontextmanager
    async def _lock_session(self):
        """Acquire lock and yield session for thread-safe operations."""
        async with self._session_lock:
            yield self._session
