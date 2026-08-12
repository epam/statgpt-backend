"""Integration tests for the advisory lock helpers, against a real PostgreSQL.

This is the test that actually guards issue #583. The bug was real connection-pool behaviour --
an engine-bound `AsyncSession` returns its connection to the pool on every commit, so an unlock
issued after a commit lands on a different backend, silently returns false, and leaves the lock
held by an idle pooled connection. No test double can reproduce that; only a real pool and a
real PostgreSQL backend can.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text

from statgpt.common.models.database import (
    SessionMaker,
    SessionMakerSingleton,
    advisory_lock_context_manager,
    get_session_context_manager,
)


def _reset_session_maker_singleton() -> None:
    SessionMakerSingleton.instance = None
    SessionMakerSingleton._engine = None
    # The class-level lock binds itself to the event loop that first awaits it.
    SessionMakerSingleton._lock = asyncio.Lock()


@pytest_asyncio.fixture(autouse=True)
async def isolated_default_engine() -> AsyncGenerator[None, None]:
    """Gives each test its own default engine, bound to that test's event loop.

    `SessionMakerSingleton` caches one connection pool per process, while pytest-asyncio runs
    every test on a fresh event loop. A pooled connection created on an earlier loop fails on
    checkout with "attached to a different loop", so the singleton is rebuilt for each test and
    disposed while the loop that created it is still running.
    """
    _reset_session_maker_singleton()
    try:
        yield
    finally:
        engine = SessionMakerSingleton._engine
        _reset_session_maker_singleton()
        if engine is not None:
            await engine.dispose()


@pytest.fixture
def lock_key() -> int:
    """A key unique to the test run, so parallel runs cannot collide."""
    return int.from_bytes(uuid.uuid4().bytes[:8], byteorder="big", signed=True)


async def _advisory_lock_count(key: int) -> int:
    """Number of backends currently holding the advisory lock for `key`."""
    async with get_session_context_manager() as session:
        result = await session.scalar(
            text(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND ((classid::bigint << 32) | objid::bigint) = :key"
            ),
            {"key": key},
        )
        return result or 0


async def _wait_until_lock_is_gone(key: int, timeout: float = 5.0) -> None:
    """Waits for every backend holding `key` to let it go.

    A backend releases its advisory locks as it exits, which is shortly after its socket is
    closed rather than at the same instant, so this cannot be asserted outright.
    """
    deadline = time.monotonic() + timeout
    while await _advisory_lock_count(key) != 0:
        if time.monotonic() >= deadline:
            pytest.fail(f"advisory lock {key} was still held after {timeout}s")
        await asyncio.sleep(0.05)


async def test_lock_survives_commits_and_is_released(lock_key: int) -> None:
    """The regression: commits inside the locked region must not orphan the lock."""
    assert await _advisory_lock_count(lock_key) == 0

    async with advisory_lock_context_manager(lock_key, timeout=5.0, description="test") as conn:
        async with get_session_context_manager(connection=conn) as session:
            for _ in range(3):
                await session.execute(text("SELECT 1"))
                await session.commit()

        # Still held after three commits, on the same backend that took it.
        assert await _advisory_lock_count(lock_key) == 1

    assert await _advisory_lock_count(lock_key) == 0, "the lock must be gone once released"


async def test_engine_bound_session_loses_the_lock_on_commit(lock_key: int) -> None:
    """Documents why the dedicated connection is necessary -- the old shape, still broken.

    The engine is local to this test and its pool is primed with two idle connections on purpose.
    A FIFO pool puts a released connection at the back of the queue, so the one a commit gives
    back is provably not the one handed out next -- which is what makes the leak reproducible
    rather than a matter of whatever the pool happened to have idle.
    """
    session_maker_factory = SessionMaker(
        engine_config=dict(pool_size=2, max_overflow=0, pool_use_lifo=False)
    )
    session_maker = await session_maker_factory.create()
    engine = session_maker_factory.engine
    assert engine is not None

    try:
        async with engine.connect() as first, engine.connect() as second:
            await first.execute(text("SELECT 1"))
            await second.execute(text("SELECT 1"))

        async with session_maker() as session:
            holder_pid = await session.scalar(text("SELECT pg_backend_pid()"))
            acquired = await session.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_key}
            )
            assert acquired

            await session.commit()  # returns the connection, and the lock, to the pool

            assert (
                await session.scalar(text("SELECT pg_backend_pid()")) != holder_pid
            ), "the commit must have moved the session to another connection"

            released = await session.scalar(
                text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key}
            )
            assert (
                released is False
            ), "unlock on a different backend returns false rather than raising"

        # The lock is now stranded on an idle pooled connection -- exactly the leak from #583.
        # It cannot be released through this pool: it belongs to a backend that no statement can
        # address any more, which is the whole point of the bug.
        assert await _advisory_lock_count(lock_key) == 1
    finally:
        # Closing the pool closes that connection, and PostgreSQL drops the locks its backend
        # held -- the same thing that eventually happens when the pool recycles a connection.
        await engine.dispose()

    await _wait_until_lock_is_gone(lock_key)


async def test_second_holder_times_out_while_the_lock_is_held(lock_key: int) -> None:
    async with advisory_lock_context_manager(lock_key, timeout=5.0, description="holder"):
        with pytest.raises(TimeoutError, match="within"):
            async with advisory_lock_context_manager(
                lock_key, timeout=0.3, description="contender"
            ):
                pytest.fail("the lock is already held; the body must not run")

    # Once released, the same key can be taken again.
    async with advisory_lock_context_manager(lock_key, timeout=5.0, description="after"):
        assert await _advisory_lock_count(lock_key) == 1
