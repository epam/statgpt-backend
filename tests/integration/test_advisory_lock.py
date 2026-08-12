"""Integration tests for the advisory lock helpers, against a real PostgreSQL.

This is the test that actually guards issue #583. The bug was real connection-pool behaviour --
an engine-bound `AsyncSession` returns its connection to the pool on every commit, so an unlock
issued after a commit lands on a different backend, silently returns false, and leaves the lock
held by an idle pooled connection. No test double can reproduce that; only a real pool and a
real PostgreSQL backend can.
"""

import uuid

import pytest
from sqlalchemy import text

from statgpt.common.models.database import (
    SessionMakerSingleton,
    advisory_lock_context_manager,
    get_session_context_manager,
)


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


async def _release_stranded_lock(key: int) -> None:
    """Frees a lock stranded on a pooled connection, whichever backend ended up holding it."""
    async with get_session_context_manager() as session:
        # The pool may well hand us the stranded connection back, in which case the lock is ours
        # to drop directly -- pg_terminate_backend below deliberately skips the current backend.
        await session.execute(text("SELECT pg_advisory_unlock_all()"))
        await session.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_locks "
                "WHERE locktype = 'advisory' "
                "AND ((classid::bigint << 32) | objid::bigint) = :key "
                "AND pid <> pg_backend_pid()"
            ),
            {"key": key},
        )
        await session.commit()


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
    """Documents why the dedicated connection is necessary -- the old shape, still broken."""
    session_maker = await SessionMakerSingleton.get_or_create()

    async with session_maker() as session:
        acquired = await session.scalar(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_key}
        )
        assert acquired

        await session.commit()  # returns the connection, and the lock, to the pool

        released = await session.scalar(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key})
        assert released is False, "unlock on a different backend returns false rather than raising"

    # The lock is now stranded on a pooled connection -- exactly the leak from #583.
    assert await _advisory_lock_count(lock_key) == 1

    # Clean up. The lock cannot be released from here: it belongs to a backend we can no longer
    # address, which is the whole point of the bug. Terminating that backend is the only way,
    # and it is what PostgreSQL would eventually do when the pool recycles the connection.
    await _release_stranded_lock(lock_key)
    assert await _advisory_lock_count(lock_key) == 0


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
