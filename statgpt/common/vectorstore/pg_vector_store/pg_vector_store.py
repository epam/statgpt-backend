import asyncio
import hashlib
import io
import json
import logging
import os
import uuid
import zipfile
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiofiles
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.documents import Document
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.expression import func

from statgpt.common.services.base import DbServiceBase
from statgpt.common.settings.database import PostgresSettings
from statgpt.common.settings.document import DimensionValueDocumentMetadataFields
from statgpt.common.utils import batched
from statgpt.common.utils.timer import debug_timer
from statgpt.common.vectorstore.base import VectorStore
from statgpt.common.vectorstore.document import ScoredVectorStoreDocument
from statgpt.common.vectorstore.embeddings import EmbeddingModel

from .document_converter import DocumentConverter
from .models import BaseDocument, BaseDocumentMetadata, BaseModel, ModelsStore

_log = logging.getLogger(__name__)


class PgVectorStore(VectorStore, DbServiceBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        *,
        session: AsyncSession,
        session_lock: asyncio.Lock | None = None,
        **kwargs,
    ):
        VectorStore.__init__(self, collection_name, embedding_model, **kwargs)
        DbServiceBase.__init__(self, session, session_lock)
        self._lock = asyncio.Lock()
        self._postgres_settings = PostgresSettings()

    @property
    def _table_name(self) -> str:
        return self._collection_name

    @property
    def _metadata_table_name(self) -> str:
        return f"{self._table_name}_metadata"

    async def _get_document_model(self) -> type[BaseDocument]:
        with debug_timer("_get_document_model.model"):
            model = ModelsStore.get_document_model(
                self._table_name, self._embedding_model.embedding_length
            )
        return model

    async def _get_metadata_model(self) -> type[BaseDocumentMetadata]:
        with debug_timer("_get_metadata_model.model"):
            return ModelsStore.get_document_metadata_model(self._metadata_table_name)

    @staticmethod
    async def _check_if_table_exists(session: AsyncSession, table_name: str) -> bool:
        """Returns True if the table exists."""
        result = await session.scalar(
            text(
                "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=:table_name)"
            ),
            {"table_name": table_name},
        )
        return result

    @staticmethod
    async def _create_table(session: AsyncSession, model: type[BaseModel]) -> None:
        query = CreateTable(model.__table__, if_not_exists=True)  # type: ignore
        await session.execute(query)
        await session.commit()

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove null bytes from text to prevent PostgreSQL UTF-8 encoding errors."""
        sanitized = text.replace("\x00", "")
        if len(sanitized) != len(text):
            preview = text[:200] if len(text) > 200 else text
            _log.debug(f"Removed null bytes from text. Preview: {preview!r}")
        return sanitized

    @staticmethod
    def _convert_timestamp(value):
        """Converts PyArrow timestamp scalar to Python datetime if needed."""
        return value.as_py() if hasattr(value, 'as_py') else value

    @classmethod
    def _sanitize_metadata(cls, metadata: dict) -> dict:
        """Recursively remove null bytes from all string values in metadata dict."""

        def sanitize_value(value):
            if isinstance(value, str):
                return cls._sanitize_text(value)
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            return value

        return sanitize_value(metadata)

    @staticmethod
    def _set_field(d: dict[str, Any], field_name: str, value: Any) -> dict[str, Any]:
        """Sets a specific field in the dictionary."""
        d[field_name] = value
        return d

    async def _create_table_if_not_exist(self, model: type[BaseModel]) -> bool:
        """Returns True if the table was created"""

        table_name = model.__tablename__

        async with self._lock_session() as session:
            if await self._check_if_table_exists(session, table_name):
                _log.info(f"Table '{table_name}' exist")
                return False
            else:
                await self._create_table(session, model)
                _log.info(f"Created table '{table_name}'")
                return True

    async def clear(self) -> None:
        # Keep outer transaction scopes (e.g. @audit_action) intact:
        # only commit when this method is not called inside an active transaction.
        should_commit = not self._session.in_transaction()
        async with self._lock_session() as session:
            for table in [self._table_name, self._metadata_table_name]:
                if await self._check_if_table_exists(session, table):

                    await session.execute(text(f'DROP TABLE IF EXISTS collections."{table}"'))
                    _log.info(f"Dropped '{table}' table")

            if should_commit:
                await session.commit()

    def _dataset_lock_key(self, dataset_id: uuid.UUID) -> int:
        """Generates a consistent lock key from collection_name and dataset_id.

        The collection_name includes the channel_id (e.g., 'Indicators_{channel_id}'),
        so this creates a composite key that's unique per (channel, dataset) pair.
        This prevents concurrent modifications to the same dataset within a channel.
        """
        composite = f"{self._collection_name}:{dataset_id}"
        hash_bytes = hashlib.sha256(composite.encode()).digest()

        # Use first 8 bytes to generate a bigint for PostgreSQL advisory lock
        return int.from_bytes(hash_bytes[:8], byteorder='big', signed=True)

    async def _acquire_dataset_lock(self, dataset_id: uuid.UUID) -> int:
        """Acquires a session-level advisory lock for the given dataset_id within this channel.

        Returns the lock key which should be passed to _release_dataset_lock.
        This lock persists across multiple transactions until explicitly released.
        This prevents concurrent modifications to the same dataset within the channel.
        """
        lock_key = self._dataset_lock_key(dataset_id)
        async with self._lock_session() as session:
            # Ensure session is in a clean state before acquiring lock
            if session.in_transaction() and not session.is_active:
                await session.rollback()
            await session.execute(
                text("SELECT pg_advisory_lock(:lock_key)"), {"lock_key": lock_key}
            )
        _log.debug(
            f"Acquired advisory lock for collection={self._collection_name} dataset={dataset_id} (key={lock_key})"
        )
        return lock_key

    async def _release_dataset_lock(self, lock_key: int) -> None:
        """Releases a session-level advisory lock.

        Handles PendingRollback state to ensure lock is always released.
        """
        async with self._lock_session() as session:
            try:
                # Rollback if session is in a failed transaction state
                if session.in_transaction() and not session.is_active:
                    await session.rollback()
                await session.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"), {"lock_key": lock_key}
                )
            except Exception as e:
                _log.error(f"Failed to release advisory lock (key={lock_key}): {e}")
                # Still try to rollback to clean up the session
                try:
                    await session.rollback()
                except Exception:
                    pass
                raise
        _log.debug(f"Released advisory lock (key={lock_key})")

    @asynccontextmanager
    async def _dataset_lock(self, dataset_id: uuid.UUID) -> AsyncIterator[None]:
        """Context manager for acquiring and releasing dataset advisory lock."""
        lock_key = await self._acquire_dataset_lock(dataset_id)
        try:
            yield
        finally:
            await self._release_dataset_lock(lock_key)

    @asynccontextmanager
    async def _dataset_locks(self, dataset_ids: list[uuid.UUID]) -> AsyncIterator[None]:
        """Context manager for acquiring and releasing multiple dataset advisory locks.

        Sorts dataset_ids to ensure deterministic lock ordering and prevent deadlocks.
        """
        sorted_dataset_ids = sorted(dataset_ids)
        lock_keys = [await self._acquire_dataset_lock(ds_id) for ds_id in sorted_dataset_ids]
        try:
            yield
        finally:
            for lock_key in lock_keys:
                await self._release_dataset_lock(lock_key)

    async def add_documents(
        self, documents: Iterable[Document], dataset_id: uuid.UUID, version_id: int
    ) -> None:
        async with self._dataset_lock(dataset_id):
            document_model: type[BaseDocument] = await self._get_document_model()
            metadata_model: type[BaseDocumentMetadata] = await self._get_metadata_model()

            await self._create_table_if_not_exist(document_model)
            await self._create_table_if_not_exist(metadata_model)

            for batch in batched(documents, self._postgres_settings.batch_size):
                embeddings = await self._embedding_model.model.aembed_documents(
                    [doc.page_content for doc in batch]
                )

                items = []
                for doc, embedding in zip(batch, embeddings):
                    items.append(
                        document_model(
                            document=self._sanitize_text(doc.page_content),
                            embeddings=embedding,
                        )
                    )

                self._session.add_all(items)
                await self._session.commit()
                _log.info(f"Added {len(items)} documents")

                mappings = [
                    metadata_model(
                        document_id=item.id,
                        dataset_id=dataset_id,
                        version_id=version_id,
                        details=self._sanitize_metadata(doc.metadata),
                    )
                    for item, doc in zip(items, batch)
                ]
                self._session.add_all(mappings)
                await self._session.commit()
                _log.info(f"Added {len(mappings)} document mappings")

    async def _get_dataset_ids_by_version_ids(self, version_ids: list[int]) -> list[uuid.UUID]:
        """Get all unique dataset_ids associated with given version_ids."""
        model = await self._get_metadata_model()
        async with self._lock_session() as session:
            res = await session.scalars(
                select(model.dataset_id).distinct().where(model.version_id.in_(version_ids))
            )
        return [item for item in res.all()]

    async def remove_documents_by(
        self, *, dataset_id: uuid.UUID | None = None, version_ids: list[int] | None = None
    ) -> None:
        """Removes all documents by dataset_id or version_ids.

        Uses advisory locks to prevent concurrent modifications to the same datasets.
        All operations (metadata deletion and orphaned document cleanup) are performed
        in a single transaction to ensure atomicity.
        """
        if not dataset_id and not version_ids:
            raise ValueError("Either dataset_id or version_ids must be provided")

        if dataset_id:
            dataset_ids_to_lock = [dataset_id]
        else:
            dataset_ids_to_lock = await self._get_dataset_ids_by_version_ids(version_ids)  # type: ignore

        async with self._dataset_locks(dataset_ids_to_lock):
            async with self._lock_session() as session:
                document_ids = await self._remove_dataset_metadata(
                    session, dataset_id=dataset_id, version_ids=version_ids
                )
                _log.info(
                    f"Deleted {len(document_ids)} rows from {self._metadata_table_name!r} table"
                )
                await self._clear_documents_without_metadata(session)
                await session.commit()

    async def _remove_dataset_metadata(
        self,
        session: AsyncSession,
        *,
        dataset_id: uuid.UUID | None,
        version_ids: list[int] | None,
    ) -> list[int]:
        """Deletes metadata rows matching the criteria and returns deleted document_ids.

        Should be called within the same transaction as document deletion
        to ensure atomicity.
        """
        model = await self._get_metadata_model()

        if not await self._check_if_table_exists(session, model.__tablename__):
            return []

        query = delete(model)

        if dataset_id is not None:
            query = query.where(model.dataset_id == dataset_id)

        if version_ids:
            query = query.where(model.version_id.in_(version_ids))

        res = await session.execute(query.returning(model.document_id))
        return [i for i in res.scalars().all()]

    async def _clear_documents_without_metadata(self, session: AsyncSession) -> None:
        """Deletes all documents that have no metadata references.

        Should be called within the same transaction as metadata deletion
        to ensure atomicity and prevent race conditions.
        """
        doc_model: type[BaseDocument] = await self._get_document_model()
        metadata_model: type[BaseDocumentMetadata] = await self._get_metadata_model()

        query = (
            delete(doc_model)
            .where(~doc_model.id.in_(select(metadata_model.document_id)))
            .returning(doc_model.id)
        )

        res = await session.execute(query)
        deleted_ids = [i for i in res.scalars().all()]
        _log.info(
            f"Deleted {len(deleted_ids)} documents without mapping from {doc_model.__tablename__!r} table"
        )

    async def search_with_similarity_score(
        self,
        query: str,
        *,
        k: int = 10,
        version_ids: set[int],
        metadata_filters: dict[str, set] | None = None,
    ) -> list[ScoredVectorStoreDocument]:

        with debug_timer("vector_store._get_document_model"):
            model = await self._get_document_model()

        with debug_timer("vector_store._get_mapping_model"):
            mapping_model = await self._get_metadata_model()

        with debug_timer("vector_store.prepare_embeddings"):
            embedding = (await self._embedding_model.model.aembed_documents([query]))[0]

        with debug_timer("vector_store.prepare_sql_query"):

            if self._embedding_model.is_normalized_to_one:
                distance = model.embeddings.max_inner_product(embedding)
            else:
                distance = model.embeddings.cosine_distance(embedding)

            matching_docs_query = select(mapping_model.document_id.distinct()).where(
                mapping_model.version_id.in_(version_ids)
            )
            if metadata_filters:
                for field, values in metadata_filters.items():
                    matching_docs_query = matching_docs_query.where(
                        mapping_model.details[field].astext.in_(values)
                    )

            # Distance is calculated once per document, not per metadata record
            top_k_docs_cte = (
                select(model.id.label("doc_id"), distance.label("distance"))
                .where(model.id.in_(matching_docs_query.scalar_subquery()))
                .order_by(distance)
                .limit(k)
                .cte("top_k_docs")
            )

            sql_query = (
                select(model, mapping_model, top_k_docs_cte.c.distance)
                .select_from(top_k_docs_cte)
                .join(model, model.id == top_k_docs_cte.c.doc_id)
                .join(mapping_model, mapping_model.document_id == top_k_docs_cte.c.doc_id)
                .where(mapping_model.version_id.in_(version_ids))
                .order_by(top_k_docs_cte.c.distance, mapping_model.id)
            )

        with debug_timer("vector_store.search.sql"):
            async with self._lock_session() as session:
                res = await session.execute(sql_query)

        with debug_timer("vector_store.search.process"):
            # `1 - cosine distance` gives cosine similarity score.
            # NOTE: for other distances, formula to get similarity score is different.
            # see notes for similar functions above.
            result = [
                DocumentConverter.to_scored_vector_store_document(
                    pg_document=doc, pg_metadata=meta, score=(1 - dist)
                )
                for doc, meta, dist in res.all()
            ]
        return result

    async def get_dataset_ids_by_documents_ids(self, ids: Iterable[int]) -> list[int]:
        model = await self._get_metadata_model()

        async with self._lock_session() as session:
            res = await session.scalars(
                select(model.dataset_id).distinct().where(model.document_id.in_(ids))
            )
        return [item for item in res.all()]

    async def get_document_ids_by_dataset_id(self, dataset_id: uuid.UUID) -> list[int]:
        model = await self._get_metadata_model()

        async with self._lock_session() as session:
            res = await session.scalars(
                select(model.document_id).where(model.dataset_id == dataset_id)
            )
        return [item for item in res.all()]

    def _build_delete_orphaned_query(self, document_table: str, metadata_table: str) -> TextClause:
        """Build SQL query to delete documents not referenced by any metadata."""
        return text(
            f"""
            DELETE FROM collections."{document_table}"
            WHERE id NOT IN (
                SELECT DISTINCT document_id
                FROM collections."{metadata_table}"
            )
        """
        )

    @staticmethod
    async def _get_duplicates(
        session: AsyncSession, query: TextClause, params=None
    ) -> tuple[bool, int]:
        result = await session.execute(statement=query, params=params)
        row = result.fetchone()

        if row:
            duplicate_count = int(row.duplicate_count)
            has_duplicates = duplicate_count > 0
            return (has_duplicates, duplicate_count)

        return (False, 0)

    async def has_duplicates(self) -> tuple[bool, int]:
        """Checks if there are duplicate documents based on document content."""
        document_model = await self._get_document_model()
        metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            for model in [document_model, metadata_model]:
                if not await self._check_if_table_exists(session, model.__tablename__):
                    _log.info(f"Table {model.__tablename__!r} does not exist.")
                    return False, 0

            duplicate_check_query = text(
                f"""
                WITH duplicate_groups AS (
                    SELECT d.document, COUNT(DISTINCT d.id) as doc_count
                    FROM collections."{document_model.__tablename__}" d
                    INNER JOIN collections."{metadata_model.__tablename__}" m ON d.id = m.document_id
                    GROUP BY d.document
                    HAVING COUNT(DISTINCT d.id) > 1
                )
                SELECT COUNT(*) as group_count,
                       COALESCE(SUM(doc_count - 1), 0) as duplicate_count
                FROM duplicate_groups
                """
            )

            return await self._get_duplicates(session, duplicate_check_query, params=None)

    async def has_duplicates_in_versions(self, version_ids: set[int]) -> tuple[bool, int]:
        """Checks if there are duplicate documents based on document content.

        Returns:
            tuple[bool, int]: (has_duplicates, duplicate_count)
        """
        document_model = await self._get_document_model()
        metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            for model in [document_model, metadata_model]:
                if not await self._check_if_table_exists(session, model.__tablename__):
                    _log.info(f"Table {model.__tablename__!r} does not exist.")
                    return False, 0

            # Build WHERE clause for version_ids filter
            where_clause = ""
            params = {}
            if version_ids:
                where_clause = "WHERE m.version_id = ANY(:version_ids)"
                params = {"version_ids": list(version_ids)}

            duplicate_check_query = text(
                f"""
                WITH duplicate_groups AS (
                    SELECT d.document, COUNT(DISTINCT d.id) as doc_count
                    FROM collections."{document_model.__tablename__}" d
                    INNER JOIN collections."{metadata_model.__tablename__}" m ON d.id = m.document_id
                    {where_clause}
                    GROUP BY d.document
                    HAVING COUNT(DISTINCT d.id) > 1
                )
                SELECT COUNT(*) as group_count,
                       COALESCE(SUM(doc_count - 1), 0) as duplicate_count
                FROM duplicate_groups
                """
            )

            return await self._get_duplicates(session, duplicate_check_query, params=params)

    async def get_total_size(self) -> int:
        """Returns the total number of documents in the vector store."""
        metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            if not await self._check_if_table_exists(session, metadata_model.__tablename__):
                _log.info(f"Table {metadata_model.__tablename__!r} does not exist.")
                return 0

            size_query = select(func.count(func.distinct(metadata_model.document_id))).select_from(
                metadata_model
            )
            result = await session.execute(size_query)
            size = result.scalar_one()
            return size

    async def get_size(self, version_ids: set[int]) -> int:
        """Returns the number of documents in the vector store."""
        metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            if not await self._check_if_table_exists(session, metadata_model.__tablename__):
                _log.info(f"Table {metadata_model.__tablename__!r} does not exist.")
                return 0

            size_query = (
                select(func.count(func.distinct(metadata_model.document_id)))
                .select_from(metadata_model)
                .where(metadata_model.version_id.in_(version_ids))
            )
            result = await session.execute(size_query)
            size = result.scalar_one()
            return size

    async def deduplicate_by_document_content(self) -> None:
        """Removes and remaps duplicate documents based on `document` field content.

        Process:
            1. Acquires exclusive locks on both document and metadata tables
            2. Identifies duplicate documents based on identical document content
            3. Remaps all metadata references from duplicates to the keeper (lowest document_id)
            4. Deletes orphaned duplicate documents
        """
        with debug_timer("deduplicate_content._get_document_model"):
            document_model = await self._get_document_model()

        with debug_timer("deduplicate_content._get_metadata_model"):
            metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            try:
                with debug_timer("deduplicate_content._check_if_table_exists"):
                    for name in [metadata_model.__tablename__, document_model.__tablename__]:
                        if not await self._check_if_table_exists(session, name):
                            _log.info(f"Table {name!r} does not exist. Skipping deduplication.")
                            return

                with debug_timer("deduplicate_content.acquire_locks"):
                    await session.execute(
                        text(
                            f'LOCK TABLE collections."{metadata_model.__tablename__}" IN EXCLUSIVE MODE'
                        )
                    )
                    await session.execute(
                        text(
                            f'LOCK TABLE collections."{document_model.__tablename__}" IN EXCLUSIVE MODE'
                        )
                    )
                    _log.info(
                        f"Acquired exclusive locks on {metadata_model.__tablename__} and {document_model.__tablename__}"
                    )

                with debug_timer("deduplicate_content.remap_references"):
                    # Use a CTE to find duplicate documents and remap metadata in one query
                    remap_query = text(
                        f"""
                        WITH duplicate_mapping AS (
                            SELECT DISTINCT
                                id as document_id,
                                FIRST_VALUE(id) OVER (
                                    PARTITION BY document
                                    ORDER BY id
                                ) as keeper_id
                            FROM collections."{document_model.__tablename__}"
                        )
                        UPDATE collections."{metadata_model.__tablename__}" m
                        SET document_id = dm.keeper_id
                        FROM duplicate_mapping dm
                        WHERE m.document_id = dm.document_id
                          AND dm.document_id != dm.keeper_id
                    """
                    )

                    result = await session.execute(remap_query)
                    remapped_count = result.rowcount
                    _log.info(f"Remapped {remapped_count} metadata rows to keeper documents")

                with debug_timer("deduplicate_content.delete_orphaned"):
                    delete_query = self._build_delete_orphaned_query(
                        document_model.__tablename__, metadata_model.__tablename__
                    )
                    result = await session.execute(delete_query)
                    deleted_count = result.rowcount
                    _log.info(f"Deleted {deleted_count} orphaned documents")

                with debug_timer("deduplicate_content.commit"):
                    await session.commit()
                    _log.info(
                        f"Content deduplication completed: {remapped_count} metadata rows remapped, {deleted_count} documents deleted"
                    )
            except Exception:
                _log.exception("Content deduplication failed. Transaction will be rolled back.")
                raise

    async def _export_documents_to_file(
        self,
        document_model: type[BaseDocument],
        metadata_model: type[BaseDocumentMetadata],
        file_path: str,
        version_ids: set[int],
    ) -> int:
        """Exports documents referenced by specified version_ids to Parquet file using streaming."""
        doc_count = 0

        # Define Parquet schema
        schema_fields: list[tuple[str, pa.DataType]] = [
            ('id', pa.int32()),
            ('document', pa.string()),
            ('embeddings', pa.list_(pa.float32())),
            ('created_at', pa.timestamp('us', tz='UTC')),
            ('updated_at', pa.timestamp('us', tz='UTC')),
        ]
        schema = pa.schema(schema_fields)

        async with self._lock_session() as session:
            # Only export documents that are referenced by metadata with specified version_ids
            matching_doc_ids = select(metadata_model.document_id.distinct()).where(
                metadata_model.version_id.in_(version_ids)
            )
            query = (
                select(document_model)
                .where(document_model.id.in_(matching_doc_ids.scalar_subquery()))
                .execution_options(yield_per=self._postgres_settings.batch_size)
            )
            result = await session.stream_scalars(query)

            writer = None
            try:
                async for batch in result.partitions():
                    # Prepare batch data
                    ids = []
                    documents = []
                    embeddings = []
                    created_ats = []
                    updated_ats = []

                    for doc in batch:
                        ids.append(doc.id)
                        documents.append(doc.document)
                        # Convert embeddings to list if needed and ensure float32
                        emb = (
                            doc.embeddings.tolist()
                            if hasattr(doc.embeddings, 'tolist')
                            else doc.embeddings
                        )
                        embeddings.append(np.array(emb, dtype=np.float32))
                        created_ats.append(doc.created_at)
                        updated_ats.append(doc.updated_at)

                    # Create PyArrow table for this batch
                    batch_table = pa.table(
                        {
                            'id': ids,
                            'document': documents,
                            'embeddings': embeddings,
                            'created_at': created_ats,
                            'updated_at': updated_ats,
                        },
                        schema=schema,
                    )

                    # Write batch
                    if writer is None:
                        writer = pq.ParquetWriter(file_path, schema, compression='zstd')
                    writer.write_table(batch_table)
                    doc_count += len(batch)

                    _log.info(f"Exported {doc_count} documents...")
            finally:
                if writer is not None:
                    writer.close()

        return doc_count

    async def _export_mappings_to_file(
        self, metadata_model: type[BaseDocumentMetadata], file_path: str, version_ids: set[int]
    ) -> int:
        """Exports metadata mappings for specified version_ids to Parquet file using streaming."""
        mapping_count = 0

        # Define Parquet schema (JSON stored as string)
        schema_fields: list[tuple[str, pa.DataType]] = [
            ('id', pa.int32()),
            ('document_id', pa.int32()),
            ('dataset_id', pa.string()),
            ('details', pa.string()),  # JSON as string
            ('created_at', pa.timestamp('us', tz='UTC')),
            ('updated_at', pa.timestamp('us', tz='UTC')),
        ]
        schema = pa.schema(schema_fields)

        async with self._lock_session() as session:
            query = (
                select(metadata_model)
                .where(metadata_model.version_id.in_(version_ids))
                .execution_options(yield_per=self._postgres_settings.batch_size)
            )
            result = await session.stream_scalars(query)

            writer = None
            try:
                async for batch in result.partitions():
                    # Prepare batch data
                    ids = []
                    document_ids = []
                    dataset_ids = []
                    details_list = []
                    created_ats = []
                    updated_ats = []

                    for mapping in batch:
                        ids.append(mapping.id)
                        document_ids.append(mapping.document_id)
                        dataset_ids.append(str(mapping.dataset_id))
                        details_list.append(json.dumps(mapping.details, ensure_ascii=False))
                        created_ats.append(mapping.created_at)
                        updated_ats.append(mapping.updated_at)

                    # Create PyArrow table for this batch
                    batch_table = pa.table(
                        {
                            'id': ids,
                            'document_id': document_ids,
                            'dataset_id': dataset_ids,
                            'details': details_list,
                            'created_at': created_ats,
                            'updated_at': updated_ats,
                        },
                        schema=schema,
                    )

                    # Write batch
                    if writer is None:
                        writer = pq.ParquetWriter(file_path, schema, compression='zstd')
                    writer.write_table(batch_table)
                    mapping_count += len(batch)

                    _log.info(f"Exported {mapping_count} mappings...")
            finally:
                if writer is not None:
                    writer.close()

        return mapping_count

    async def _write_export_metadata(
        self, folder_path: str, doc_count: int, mapping_count: int
    ) -> None:
        """Writes metadata.json with collection and export information."""
        metadata = {
            "collection_name": self._collection_name,
            "embedding_model_name": self._embedding_model.name,
            "embedding_length": self._embedding_model.embedding_length,
            "is_normalized_to_one": self._embedding_model.is_normalized_to_one,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "document_count": doc_count,
            "mapping_count": mapping_count,
        }
        metadata_path = os.path.join(folder_path, "metadata.json")
        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))

    async def export_to_folder(self, folder_path: str, version_ids: set[int]) -> None:
        """Exports the vector store data to the specified folder.

        Creates:
        - metadata.json: Collection and embedding model metadata
        - documents.parquet: All documents with embeddings (Parquet format with zstd compression)
        - mappings.parquet: All document-metadata mappings (Parquet format with zstd compression)

        The export uses streaming to handle large collections efficiently.
        """
        _log.info(f"Exporting collection '{self._collection_name}' to {folder_path}")

        os.makedirs(folder_path, exist_ok=True)

        document_model = await self._get_document_model()
        metadata_model = await self._get_metadata_model()

        async with self._lock_session() as session:
            doc_table_exists = await self._check_if_table_exists(
                session, document_model.__tablename__
            )
            meta_table_exists = await self._check_if_table_exists(
                session, metadata_model.__tablename__
            )

        if not doc_table_exists or not meta_table_exists:
            _log.warning(
                f"Tables don't exist for collection '{self._collection_name}'. Creating empty export."
            )
            doc_count = 0
            mapping_count = 0
        else:
            documents_path = os.path.join(folder_path, "documents.parquet")
            doc_count = await self._export_documents_to_file(
                document_model, metadata_model, documents_path, version_ids
            )
            _log.info(f"Exported {doc_count} documents")

            mappings_path = os.path.join(folder_path, "mappings.parquet")
            mapping_count = await self._export_mappings_to_file(
                metadata_model, mappings_path, version_ids
            )
            _log.info(f"Exported {mapping_count} mappings")

        await self._write_export_metadata(folder_path, doc_count, mapping_count)

        _log.info(
            f"Successfully exported collection '{self._collection_name}': "
            f"{doc_count} documents, {mapping_count} mappings"
        )

    def _validate_import_zipfile(
        self, zip_file: zipfile.ZipFile, folder_prefix: str
    ) -> tuple[str, str | None, str | None]:
        """Validates required files exist in zip archive.

        Returns tuple of (metadata_path, documents_path, mappings_path) relative to zip root.
        documents_path and mappings_path may be None for empty exports.
        """
        metadata_path = f"{folder_prefix}/metadata.json"
        documents_path = f"{folder_prefix}/documents.parquet"
        mappings_path = f"{folder_prefix}/mappings.parquet"

        zip_contents = set(zip_file.namelist())

        # metadata.json is always required
        if metadata_path not in zip_contents:
            raise FileNotFoundError(f"Required file not found in archive: {metadata_path}")

        # Read metadata to check if this is an empty export
        with zip_file.open(metadata_path) as f:
            text_stream = io.TextIOWrapper(f, encoding='utf-8')
            metadata = json.load(text_stream)

        doc_count = metadata.get("document_count", 0)
        mapping_count = metadata.get("mapping_count", 0)

        # For empty exports, parquet files may not exist
        if doc_count == 0 and mapping_count == 0:
            _log.info(
                f"Empty export detected for '{folder_prefix}', skipping parquet file validation"
            )
            return metadata_path, None, None

        # For non-empty exports, parquet files must exist
        if documents_path not in zip_contents:
            raise FileNotFoundError(f"Required file not found in archive: {documents_path}")
        if mappings_path not in zip_contents:
            raise FileNotFoundError(f"Required file not found in archive: {mappings_path}")

        return metadata_path, documents_path, mappings_path

    def _validate_import_metadata_from_zipfile(
        self, zip_file: zipfile.ZipFile, metadata_path: str
    ) -> dict:
        """Reads and validates metadata from zip against current embedding model."""
        with zip_file.open(metadata_path) as f:
            text_stream = io.TextIOWrapper(f, encoding='utf-8')
            metadata = json.load(text_stream)

        if metadata["embedding_model_name"] != self._embedding_model.name:
            raise ValueError(
                f"Embedding model mismatch: current={self._embedding_model.name}, "
                f"import={metadata['embedding_model_name']}"
            )

        if metadata["embedding_length"] != self._embedding_model.embedding_length:
            raise ValueError(
                f"Embedding length mismatch: current={self._embedding_model.embedding_length}, "
                f"import={metadata['embedding_length']}"
            )

        if metadata.get("is_normalized_to_one") != self._embedding_model.is_normalized_to_one:
            _log.warning(
                f"Embedding normalization differs: current={self._embedding_model.is_normalized_to_one}, "
                f"import={metadata.get('is_normalized_to_one')}"
            )

        return metadata

    async def _import_documents_from_zipfile(
        self, zip_file: zipfile.ZipFile, document_model: type[BaseDocument], file_path: str
    ) -> dict[int, int]:
        """Imports documents from Parquet file in zip archive in batches.

        Returns a mapping from old document IDs to new auto-generated IDs.
        """
        doc_count = 0
        id_mapping: dict[int, int] = {}

        # Read Parquet file from zip archive
        with zip_file.open(file_path) as f:
            parquet_file = pq.ParquetFile(f)

            # Process in batches
            for batch in parquet_file.iter_batches(batch_size=self._postgres_settings.batch_size):
                doc_batch: list[BaseDocument] = []
                batch_dict = batch.to_pydict()

                for i in range(len(batch_dict['id'])):
                    try:
                        doc = document_model(
                            # Don't set id - let PostgreSQL auto-generate it
                            document=self._sanitize_text(batch_dict['document'][i]),
                            embeddings=(
                                batch_dict['embeddings'][i].tolist()
                                if hasattr(batch_dict['embeddings'][i], 'tolist')
                                else batch_dict['embeddings'][i]
                            ),
                            created_at=self._convert_timestamp(batch_dict['created_at'][i]),
                            updated_at=self._convert_timestamp(batch_dict['updated_at'][i]),
                        )
                        doc_batch.append(doc)
                    except (KeyError, TypeError, ValueError) as e:
                        _log.error(f"Error parsing document at index {i}: {e}")
                        raise ValueError(f"Corrupt document data: {e}")

                self._session.add_all(doc_batch)
                await self._session.flush()

                # Map old IDs to new auto-generated IDs
                for i, doc in enumerate(doc_batch):
                    old_id = batch_dict['id'][i]
                    id_mapping[old_id] = doc.id

                await self._session.commit()
                doc_count += len(doc_batch)
                _log.info(f"Imported {doc_count} documents...")

        return id_mapping

    async def _import_mappings_from_zipfile(
        self,
        zip_file: zipfile.ZipFile,
        metadata_model: type[BaseDocumentMetadata],
        file_path: str,
        dataset_versions: dict[uuid.UUID, int],
        data_sources: dict[uuid.UUID, int],
        id_mapping: dict[int, int],
    ) -> int:
        """Imports metadata mappings from Parquet file in zip archive in batches.

        Args:
            id_mapping: Maps old document IDs to new auto-generated IDs
        """
        mapping_count = 0

        # Read Parquet file from zip archive
        with zip_file.open(file_path) as f:
            parquet_file = pq.ParquetFile(f)

            # Process in batches
            for batch in parquet_file.iter_batches(batch_size=self._postgres_settings.batch_size):
                mapping_batch: list[BaseDocumentMetadata] = []
                batch_dict = batch.to_pydict()

                for i in range(len(batch_dict['id'])):
                    try:
                        dataset_id = uuid.UUID(batch_dict['dataset_id'][i])
                        old_document_id = batch_dict['document_id'][i]

                        # Look up new document ID
                        new_document_id = id_mapping.get(old_document_id)
                        if new_document_id is None:
                            _log.error(
                                f"Mapping references non-existent document_id={old_document_id}. "
                                f"This indicates corrupted or mismatched export data."
                            )
                            raise ValueError(
                                f"Mapping references document_id={old_document_id} which was not found in documents"
                            )

                        # 'details' field processing
                        details = json.loads(batch_dict['details'][i])
                        details = self._sanitize_metadata(details)
                        details = self._set_field(
                            details,
                            DimensionValueDocumentMetadataFields.DATA_SOURCE_ID,
                            data_sources[dataset_id],
                        )

                        mapping = metadata_model(
                            # Don't set id - let PostgreSQL auto-generate it
                            document_id=new_document_id,  # Use new document ID
                            dataset_id=dataset_id,
                            version_id=dataset_versions[dataset_id],
                            details=details,
                            created_at=self._convert_timestamp(batch_dict['created_at'][i]),
                            updated_at=self._convert_timestamp(batch_dict['updated_at'][i]),
                        )
                        mapping_batch.append(mapping)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                        _log.error(f"Error parsing mapping at index {i}: {e}")
                        raise ValueError(f"Corrupt mapping data: {e}")

                self._session.add_all(mapping_batch)
                await self._session.commit()
                mapping_count += len(mapping_batch)
                _log.info(f"Imported {mapping_count} mappings...")

        return mapping_count

    async def import_from_zipfile(
        self,
        zip_file: zipfile.ZipFile,
        folder_prefix: str,
        dataset_versions: dict[uuid.UUID, int],
        data_sources: dict[uuid.UUID, int],
    ) -> None:
        """Clears the current vector store and imports data from the specified zip file folder.

        Validates:
        - Embedding model name matches
        - Embedding length matches
        - File formats are valid

        Raises:
        - ValueError: If validation fails or files are missing/corrupt
        - FileNotFoundError: If required files don't exist in archive
        """
        _log.info(
            f"Importing collection '{self._collection_name}' from zip archive: {folder_prefix}"
        )

        metadata_path, documents_path, mappings_path = self._validate_import_zipfile(
            zip_file, folder_prefix
        )
        metadata = self._validate_import_metadata_from_zipfile(zip_file, metadata_path)

        _log.info(
            f"Importing {metadata['document_count']} documents and {metadata['mapping_count']} mappings "
            f"from archive"
        )

        await self.clear()
        _log.info(f"Cleared existing tables for collection '{self._collection_name}'")

        document_model = await self._get_document_model()
        metadata_model = await self._get_metadata_model()
        await self._create_table_if_not_exist(document_model)
        await self._create_table_if_not_exist(metadata_model)
        _log.info(f"Created tables for collection '{self._collection_name}'")

        # Skip import if this was an empty export (no parquet files)
        if documents_path is None or mappings_path is None:
            _log.info("Empty export detected, skipping document and mapping import")
            doc_count = 0
            mapping_count = 0
        else:
            id_mapping = await self._import_documents_from_zipfile(
                zip_file, document_model, documents_path
            )
            doc_count = len(id_mapping)
            _log.info(f"Imported {doc_count} documents")

            mapping_count = await self._import_mappings_from_zipfile(
                zip_file, metadata_model, mappings_path, dataset_versions, data_sources, id_mapping
            )
            _log.info(f"Imported {mapping_count} mappings")

        _log.info(
            f"Successfully imported collection '{self._collection_name}': "
            f"{doc_count} documents, {mapping_count} mappings"
        )
