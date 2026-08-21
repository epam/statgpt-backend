"""Tests for the PostgreSQL advisory lock helpers in `statgpt.common.models.database`.

These guard the invariant the locks depend on: a session-level advisory lock belongs to the
PostgreSQL connection that took it, so acquire and release must happen on the *same* connection.
Holding the lock through an engine-bound `AsyncSession` broke that, because such a session
returns its connection to the pool on every commit -- the unlock then landed on a different
connection, silently returned false, and left the lock held by an idle pooled connection.

They also guard the two rules that keep the pool healthy: no connection is held while waiting
for a contended lock, and no code path leaves a connection open.
"""

import asyncio
from typing import Any

import pytest

from statgpt.common.models import database as module

LOCK_KEY = 4815162342
OTHER_KEY = 1123581321


class FakeConnection:
    """Records advisory lock traffic and its own open/close lifecycle."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.commits = 0
        self.closed = False
        self.invalidated = False
        # Keys pg_try_advisory_lock should refuse to grant, and non-default unlock results.
        self.ungrantable_keys: set[int] = set()
        self.unlock_results: dict[int, bool] = {}

    async def scalar(self, statement: Any, params: dict[str, Any]) -> bool:
        key = params["lock_key"]
        sql = str(statement)
        if "pg_try_advisory_lock" in sql:
            self.calls.append(("lock", key))
            return key not in self.ungrantable_keys
        if "pg_advisory_unlock" in sql:
            self.calls.append(("unlock", key))
            return self.unlock_results.get(key, True)
        raise AssertionError(f"unexpected statement: {sql}")

    async def commit(self) -> None:
        self.commits += 1

    async def invalidate(self) -> None:
        self.invalidated = True

    def locked_keys(self) -> list[int]:
        return [key for op, key in self.calls if op == "lock"]

    def unlocked_keys(self) -> list[int]:
        return [key for op, key in self.calls if op == "unlock"]


class FakeEngine:
    """Hands out a fresh connection per `connect()` and tracks every one it produced."""

    def __init__(self, template: FakeConnection | None = None) -> None:
        self.template = template
        self.connections: list[FakeConnection] = []

    def connect(self) -> "FakeEngine":
        return self

    async def __aenter__(self) -> FakeConnection:
        if self.template is not None:
            conn = self.template
        else:
            conn = FakeConnection()
        self.connections.append(conn)
        return conn

    async def __aexit__(self, *_exc: Any) -> None:
        self.connections[-1].closed = True


@pytest.fixture
def conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, conn: FakeConnection) -> FakeEngine:
    return _bind_engine(monkeypatch, FakeEngine(template=conn))


def _bind_engine(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> FakeEngine:
    async def fake_get_or_create_engine() -> FakeEngine:
        return engine

    monkeypatch.setattr(
        module.SessionMakerSingleton, "get_or_create_engine", fake_get_or_create_engine
    )
    return engine


async def test_lock_is_acquired_and_released_on_the_same_connection(
    conn: FakeConnection, engine: FakeEngine
) -> None:
    async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test") as locked:
        assert locked is conn
        assert conn.calls == [("lock", LOCK_KEY)], "the lock must be held for the whole body"

    assert conn.calls == [("lock", LOCK_KEY), ("unlock", LOCK_KEY)]
    assert len(engine.connections) == 1, "the lock must not be spread over several connections"
    assert conn.closed
    assert not conn.invalidated


async def test_transaction_is_committed_so_the_connection_is_not_idle_in_transaction(
    conn: FakeConnection, engine: FakeEngine
) -> None:
    """The probe autobegins a transaction; leaving it open would block the bound session."""
    async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test"):
        assert conn.commits == 1, "the acquiring statement's transaction must be closed"


async def test_no_connection_is_held_while_waiting_for_a_contended_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each failed probe must close its connection before the backoff sleep."""
    engine = _bind_engine(monkeypatch, FakeEngine())
    attempts = 0

    async def fake_sleep(_delay: float) -> None:
        nonlocal attempts
        attempts += 1
        # Grant the lock only from the third probe onwards.
        if attempts >= 2:
            engine.connections[-1].ungrantable_keys = set()

    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)

    original_aenter = FakeEngine.__aenter__

    async def refusing_aenter(self: FakeEngine) -> FakeConnection:
        conn = await original_aenter(self)
        if attempts < 2:
            conn.ungrantable_keys = {LOCK_KEY}
        return conn

    monkeypatch.setattr(FakeEngine, "__aenter__", refusing_aenter)

    async with module.advisory_lock_context_manager(LOCK_KEY, 10.0, "test"):
        pass

    assert len(engine.connections) == 3, "each retry must open a fresh connection"
    assert all(c.closed for c in engine.connections), "no connection may survive its probe"
    assert engine.connections[-1].unlocked_keys() == [LOCK_KEY]


async def test_timeout_raises_and_leaves_no_connection_open(
    monkeypatch: pytest.MonkeyPatch, conn: FakeConnection
) -> None:
    engine = _bind_engine(monkeypatch, FakeEngine(template=conn))
    conn.ungrantable_keys = {LOCK_KEY}

    with pytest.raises(TimeoutError, match="within"):
        async with module.advisory_lock_context_manager(LOCK_KEY, 0.05, "test"):
            pytest.fail("the body must not run when the lock cannot be acquired")

    assert conn.closed
    assert conn.unlocked_keys() == [], "nothing was locked, so nothing may be unlocked"
    assert engine.connections, "at least one probe must have been attempted"


async def test_connection_is_discarded_when_unlock_reports_the_lock_was_not_held(
    conn: FakeConnection, engine: FakeEngine
) -> None:
    """pg_advisory_unlock returns false rather than raising, which is how the leak went unseen."""
    conn.unlock_results = {LOCK_KEY: False}

    async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test"):
        pass

    assert conn.invalidated, "a connection that may still hold the lock must not be reused"
    assert conn.closed


async def test_release_failure_does_not_mask_the_error_from_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingConnection(FakeConnection):
        async def scalar(self, statement: Any, params: dict[str, Any]) -> bool:
            if "pg_advisory_unlock" in str(statement):
                raise RuntimeError("connection is gone")
            return await super().scalar(statement, params)

    conn = ExplodingConnection()
    _bind_engine(monkeypatch, FakeEngine(template=conn))

    with pytest.raises(ValueError, match="body failed"):
        async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test"):
            raise ValueError("body failed")

    assert conn.invalidated
    assert conn.closed


async def test_cancelled_release_discards_the_connection_and_stays_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled unlock may never have reached PostgreSQL, so the connection cannot be reused."""

    class CancellingConnection(FakeConnection):
        async def scalar(self, statement: Any, params: dict[str, Any]) -> bool:
            if "pg_advisory_unlock" in str(statement):
                raise asyncio.CancelledError
            return await super().scalar(statement, params)

    conn = CancellingConnection()
    _bind_engine(monkeypatch, FakeEngine(template=conn))

    with pytest.raises(asyncio.CancelledError):
        async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test"):
            pass

    assert conn.invalidated, "a connection that may still hold the lock must not be pooled"
    assert conn.closed


async def test_invalidate_failing_does_not_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """invalidate() raises on an already-closed connection; the cleanup path must swallow it."""

    class UninvalidatableConnection(FakeConnection):
        async def invalidate(self) -> None:
            raise RuntimeError("This Connection is closed")

    conn = UninvalidatableConnection()
    conn.unlock_results = {LOCK_KEY: False}
    _bind_engine(monkeypatch, FakeEngine(template=conn))

    async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "test"):
        pass

    assert conn.closed


async def test_distinct_keys_are_locked_independently(
    conn: FakeConnection, engine: FakeEngine
) -> None:
    async with module.advisory_lock_context_manager(LOCK_KEY, 1.0, "a"):
        pass
    async with module.advisory_lock_context_manager(OTHER_KEY, 1.0, "b"):
        pass

    assert conn.locked_keys() == [LOCK_KEY, OTHER_KEY]
    assert conn.unlocked_keys() == [LOCK_KEY, OTHER_KEY]
