"""Tests for the dataset advisory locks of `PgEmbeddinglessVectorStore`.

These guard the invariant the locks depend on: a session-level advisory lock belongs to the
PostgreSQL connection that took it, so acquire and release must happen on the *same*
connection. Holding the lock through an `AsyncSession` instead broke that, because a session
returns its connection to the pool on every commit -- the unlock then landed on a different
connection, silently returned false, and left the lock held by an idle pooled connection.
"""

import uuid
from typing import Any

import pytest

from statgpt.common.settings.database import PostgresSettings
from statgpt.common.vectorstore.pg_vector_store import pg_vector_store as module
from statgpt.common.vectorstore.pg_vector_store.pg_vector_store import PgEmbeddinglessVectorStore

DATASET_A = uuid.UUID("dee4481b-33d9-5268-aeab-decb5f071821")
DATASET_B = uuid.UUID("e5135496-a933-4692-b8bf-b8e9f60d38fa")


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class FakeConnection:
    """Records advisory lock traffic.

    `execute` is only ever used to take locks and `scalar` to release them, which is what lets
    this double tell the two apart.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.execution_options_seen: list[dict[str, Any]] = []
        self.invalidated = False
        self.closed = False
        # Keys pg_try_advisory_lock should refuse to grant, and non-default unlock results.
        self.ungrantable_keys: set[int] = set()
        self.unlock_results: dict[int, bool] = {}

    async def execution_options(self, **opts: Any) -> "FakeConnection":
        self.execution_options_seen.append(opts)
        return self

    async def execute(self, _statement: Any, params: dict[str, Any]) -> _FakeResult:
        key = params["lock_key"]
        self.calls.append(("lock", key))
        return _FakeResult(key not in self.ungrantable_keys)

    async def scalar(self, _statement: Any, params: dict[str, Any]) -> bool:
        key = params["lock_key"]
        self.calls.append(("unlock", key))
        return self.unlock_results.get(key, True)

    async def invalidate(self) -> None:
        self.invalidated = True

    def locked_keys(self) -> list[int]:
        return [key for op, key in self.calls if op == "lock"]

    def unlocked_keys(self) -> list[int]:
        return [key for op, key in self.calls if op == "unlock"]


class FakeEngine:
    """Hands out the same connection every time, and counts how often one was asked for."""

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn
        self.connect_count = 0

    def connect(self) -> "FakeEngine":
        self.connect_count += 1
        return self

    async def __aenter__(self) -> FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc: Any) -> None:
        self._conn.closed = True


@pytest.fixture(autouse=True)
def _postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """PostgresSettings refuses to build without connection env vars."""
    for name, value in {
        "PGVECTOR_HOST": "localhost",
        "PGVECTOR_PORT": "5432",
        "PGVECTOR_DATABASE": "test",
        "PGVECTOR_USER": "test",
        "PGVECTOR_PASSWORD": "test",
    }.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def store() -> PgEmbeddinglessVectorStore:
    store = PgEmbeddinglessVectorStore("SpecialDimensions_12")
    # Keep the acquisition loop short: these tests exercise timeouts.
    store._postgres_settings = PostgresSettings(advisory_lock_timeout=0.05)
    return store


def _bind_engine(monkeypatch: pytest.MonkeyPatch, conn: FakeConnection) -> FakeEngine:
    engine = FakeEngine(conn)

    async def fake_get_engine() -> FakeEngine:
        return engine

    monkeypatch.setattr(module, "get_engine", fake_get_engine)
    return engine


async def test_lock_is_acquired_and_released_on_the_same_connection(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = FakeConnection()
    engine = _bind_engine(monkeypatch, conn)
    key = store._dataset_lock_key(DATASET_A)

    async with store._dataset_lock(DATASET_A):
        assert conn.calls == [("lock", key)], "the lock must be held for the whole body"

    assert conn.calls == [("lock", key), ("unlock", key)]
    assert engine.connect_count == 1, "the lock must not be spread over several connections"
    assert conn.closed
    assert not conn.invalidated


async def test_lock_connection_uses_autocommit(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection that does nothing but hold locks must not sit idle in a transaction."""
    conn = FakeConnection()
    _bind_engine(monkeypatch, conn)

    async with store._dataset_lock(DATASET_A):
        pass

    assert conn.execution_options_seen == [{"isolation_level": "AUTOCOMMIT"}]


async def test_locks_are_acquired_in_sorted_dataset_order_and_released_in_reverse(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = FakeConnection()
    engine = _bind_engine(monkeypatch, conn)
    expected = [store._dataset_lock_key(ds) for ds in sorted([DATASET_B, DATASET_A])]

    async with store._dataset_locks([DATASET_B, DATASET_A]):
        pass

    assert conn.locked_keys() == expected
    assert conn.unlocked_keys() == list(reversed(expected))
    assert engine.connect_count == 1


async def test_already_acquired_locks_are_released_when_a_later_one_times_out(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old code acquired locks outside its `try`, so a timeout leaked the earlier ones."""
    conn = FakeConnection()
    _bind_engine(monkeypatch, conn)
    first_key, second_key = (store._dataset_lock_key(ds) for ds in sorted([DATASET_A, DATASET_B]))
    conn.ungrantable_keys = {second_key}

    with pytest.raises(TimeoutError):
        async with store._dataset_locks([DATASET_A, DATASET_B]):
            pytest.fail("the body must not run when a lock cannot be acquired")

    assert conn.unlocked_keys() == [first_key], "the already acquired lock was leaked"


async def test_duplicate_dataset_ids_are_locked_once(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key locked twice on one connection would need two unlocks to be released."""
    conn = FakeConnection()
    _bind_engine(monkeypatch, conn)
    key = store._dataset_lock_key(DATASET_A)

    async with store._dataset_locks([DATASET_A, DATASET_A]):
        pass

    assert conn.calls == [("lock", key), ("unlock", key)]


async def test_connection_is_discarded_when_unlock_reports_the_lock_was_not_held(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pg_advisory_unlock returns false rather than raising, which is how the leak went unseen."""
    conn = FakeConnection()
    _bind_engine(monkeypatch, conn)
    conn.unlock_results = {store._dataset_lock_key(DATASET_A): False}

    async with store._dataset_lock(DATASET_A):
        pass

    assert conn.invalidated, "a connection that may still hold the lock must not be reused"


async def test_release_failure_does_not_mask_the_error_from_the_body(
    store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingConnection(FakeConnection):
        async def scalar(self, _statement: Any, params: dict[str, Any]) -> bool:
            raise RuntimeError("connection is gone")

    conn = ExplodingConnection()
    _bind_engine(monkeypatch, conn)

    with pytest.raises(ValueError, match="body failed"):
        async with store._dataset_lock(DATASET_A):
            raise ValueError("body failed")

    assert conn.invalidated
