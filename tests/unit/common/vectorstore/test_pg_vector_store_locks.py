"""Tests for how `PgVectorStore` uses the dataset advisory lock.

The lock is taken on a dedicated connection and the work is bound to that same connection, so
that the commits inside the locked region cannot move the session elsewhere and orphan the lock
(see `tests/unit/common/models/test_advisory_lock.py` for the lock helper itself).

Also guards the two invariants around it: `remove_documents_by` scopes its delete to a single
dataset, and `add_documents` never publishes a document without its metadata.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.documents import Document

from statgpt.common.settings.database import PostgresSettings
from statgpt.common.vectorstore.pg_vector_store import pg_vector_store as module
from statgpt.common.vectorstore.pg_vector_store.pg_vector_store import (
    PgEmbeddinglessVectorStore,
    PgVectorStore,
)

DATASET_A = uuid.UUID("dee4481b-33d9-5268-aeab-decb5f071821")
DATASET_B = uuid.UUID("e5135496-a933-4692-b8bf-b8e9f60d38fa")
VERSION_ID = 7


class FakeSession:
    """Records the write traffic a locked region produces."""

    def __init__(self) -> None:
        self.added: list[list[Any]] = []
        self.flushes = 0
        self.commits = 0
        self.bound_connection: Any = None

    def add_all(self, items: list[Any]) -> None:
        self.added.append(list(items))

    async def flush(self) -> None:
        self.flushes += 1
        # A real flush assigns primary keys; the mappings are built from them.
        for batch in self.added:
            for index, item in enumerate(batch):
                if getattr(item, "id", None) is None:
                    item.id = index + 1

    async def commit(self) -> None:
        self.commits += 1


class FakeDocument:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.id: int | None = None


class FakeMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


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
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def locks(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> list[uuid.UUID]:
    """Patches the lock and the session factory; returns the list of locked dataset ids."""
    locked: list[uuid.UUID] = []
    sentinel = object()

    @asynccontextmanager
    async def fake_dataset_lock(_self: Any, dataset_id: uuid.UUID) -> AsyncIterator[Any]:
        locked.append(dataset_id)
        yield sentinel

    @asynccontextmanager
    async def fake_session_cm(connection: Any = None) -> AsyncIterator[FakeSession]:
        session.bound_connection = connection
        yield session

    monkeypatch.setattr(PgEmbeddinglessVectorStore, "_dataset_lock", fake_dataset_lock)
    monkeypatch.setattr(module, "get_session_context_manager", fake_session_cm)
    return locked


def test_lock_key_is_stable_and_scoped_to_collection_and_dataset() -> None:
    store_a = PgEmbeddinglessVectorStore("Indicators_1")
    store_b = PgEmbeddinglessVectorStore("Indicators_2")

    assert store_a._dataset_lock_key(DATASET_A) == store_a._dataset_lock_key(DATASET_A)
    assert store_a._dataset_lock_key(DATASET_A) != store_a._dataset_lock_key(DATASET_B)
    assert store_a._dataset_lock_key(DATASET_A) != store_b._dataset_lock_key(DATASET_A)
    # PostgreSQL advisory keys are signed bigints.
    assert -(2**63) <= store_a._dataset_lock_key(DATASET_A) < 2**63


class TestRemoveDocumentsBy:
    @pytest.fixture
    def store(self, monkeypatch: pytest.MonkeyPatch) -> PgEmbeddinglessVectorStore:
        store = PgEmbeddinglessVectorStore("SpecialDimensions_12")
        self.removed: list[dict[str, Any]] = []
        self.swept = 0

        async def fake_remove_metadata(
            _session: Any, *, dataset_id: uuid.UUID | None, version_ids: list[int] | None
        ) -> list[int]:
            self.removed.append({"dataset_id": dataset_id, "version_ids": version_ids})
            return [1, 2]

        async def fake_sweep(_session: Any) -> None:
            self.swept += 1

        monkeypatch.setattr(store, "_remove_dataset_metadata", fake_remove_metadata)
        monkeypatch.setattr(store, "_clear_documents_without_metadata", fake_sweep)
        return store

    async def test_requires_dataset_id_or_version_ids(
        self, store: PgEmbeddinglessVectorStore
    ) -> None:
        with pytest.raises(ValueError, match="Either dataset_id or version_ids"):
            await store.remove_documents_by()

    async def test_locks_the_dataset_and_sweeps_after_releasing_it(
        self, store: PgEmbeddinglessVectorStore, locks: list[uuid.UUID], session: FakeSession
    ) -> None:
        await store.remove_documents_by(dataset_id=DATASET_A)

        assert locks == [DATASET_A]
        assert self.removed == [{"dataset_id": DATASET_A, "version_ids": None}]
        assert self.swept == 1
        assert session.commits == 2, "one commit for the delete, one for the sweep"

    async def test_delete_is_scoped_to_the_resolved_dataset(
        self,
        store: PgEmbeddinglessVectorStore,
        locks: list[uuid.UUID],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The version_ids branch used to delete every dataset's rows for those versions."""

        async def fake_resolve(_version_ids: list[int]) -> uuid.UUID:
            return DATASET_B

        monkeypatch.setattr(store, "_resolve_dataset_id", fake_resolve)

        await store.remove_documents_by(version_ids=[VERSION_ID])

        assert locks == [DATASET_B]
        assert self.removed == [{"dataset_id": DATASET_B, "version_ids": [VERSION_ID]}]

    async def test_returns_early_when_versions_match_nothing(
        self,
        store: PgEmbeddinglessVectorStore,
        locks: list[uuid.UUID],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_resolve(_version_ids: list[int]) -> None:
            return None

        monkeypatch.setattr(store, "_resolve_dataset_id", fake_resolve)

        await store.remove_documents_by(version_ids=[VERSION_ID])

        assert locks == [], "nothing to remove means nothing to lock"
        assert self.removed == []
        assert self.swept == 0, "an unlocked global sweep must not run for a no-op call"

    async def test_rejects_versions_spanning_several_datasets(
        self, store: PgEmbeddinglessVectorStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_ids(_session: Any, _version_ids: list[int]) -> list[uuid.UUID]:
            return [DATASET_A, DATASET_B]

        @asynccontextmanager
        async def fake_readonly_session() -> AsyncIterator[Any]:
            yield object()

        monkeypatch.setattr(store, "_get_dataset_ids_by_version_ids", fake_ids)
        monkeypatch.setattr(module, "get_readonly_session_context_manager", fake_readonly_session)

        with pytest.raises(ValueError, match="single dataset"):
            await store.remove_documents_by(version_ids=[VERSION_ID])


class TestAddDocuments:
    @pytest.fixture
    def store(self, monkeypatch: pytest.MonkeyPatch) -> PgVectorStore:
        class FakeEmbeddings:
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

        class FakeEmbeddingModel:
            name = "fake"
            model = FakeEmbeddings()

        store = PgVectorStore("Indicators_1", embedding_model=FakeEmbeddingModel())  # type: ignore
        store._postgres_settings = PostgresSettings(batch_size=2)

        async def fake_document_model() -> Any:
            return FakeDocument

        async def fake_metadata_model() -> Any:
            return FakeMetadata

        async def fake_create_table(_session: Any, _model: Any) -> bool:
            return False

        monkeypatch.setattr(store, "_get_document_model", fake_document_model)
        monkeypatch.setattr(store, "_get_metadata_model", fake_metadata_model)
        monkeypatch.setattr(store, "_create_table_if_not_exist", fake_create_table)
        return store

    async def test_work_runs_on_the_connection_holding_the_lock(
        self, store: PgVectorStore, locks: list[uuid.UUID], session: FakeSession
    ) -> None:
        await store.add_documents([Document(page_content="a")], DATASET_A, VERSION_ID)

        assert locks == [DATASET_A]
        assert session.bound_connection is not None, "the session must be bound to the lock conn"

    async def test_documents_and_mappings_land_in_one_commit_per_batch(
        self, store: PgVectorStore, locks: list[uuid.UUID], session: FakeSession
    ) -> None:
        """Committing documents before their mappings left them briefly sweepable."""
        documents = [Document(page_content=str(i)) for i in range(4)]  # batch_size=2 -> 2 batches

        await store.add_documents(documents, DATASET_A, VERSION_ID)

        assert session.flushes == 2, "one flush per batch, to assign the document ids"
        assert session.commits == 2, "one commit per batch, publishing documents with mappings"
        # add_all is called twice per batch: documents, then their mappings.
        assert [type(batch[0]) for batch in session.added] == [
            FakeDocument,
            FakeMetadata,
            FakeDocument,
            FakeMetadata,
        ]

    async def test_mappings_reference_the_flushed_document_ids(
        self, store: PgVectorStore, locks: list[uuid.UUID], session: FakeSession
    ) -> None:
        await store.add_documents([Document(page_content="a")], DATASET_A, VERSION_ID)

        mappings = session.added[1]
        assert [m.document_id for m in mappings] == [1]
        assert all(m.dataset_id == DATASET_A and m.version_id == VERSION_ID for m in mappings)
