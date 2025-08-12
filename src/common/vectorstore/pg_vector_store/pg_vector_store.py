from collections.abc import Iterable

from langchain_core.documents import Document
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql import text

from common import models
from common.config import multiline_logger as logger
from common.config.database import PG_VECTOR_STORE_BATCH_SIZE
from common.utils import batched
from common.vectorstore.base import VectorStore
from common.vectorstore.document import EmbeddedDocument
from common.vectorstore.embeddings import EmbeddingModel

from .models import BaseDatasetDocumentMapping, BaseDocument, BaseModel, CollectionName, ModelsStore


class PgVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        datasource: str | None = None,
        *,
        session: AsyncSession,
        **kwargs,
    ):
        super().__init__(collection_name, embedding_model, datasource, **kwargs)
        self._session = session

        self._table_name: str | None = None
        self._mapping_table_name: str | None = None

    async def _find_collection(
        self, collection_name: str, datasource: str | None, embedding_name
    ) -> CollectionName | None:
        query = select(CollectionName).where(
            CollectionName.collection_name == collection_name,
            CollectionName.datasource == datasource,
            CollectionName.embedding_model_name == embedding_name,
        )
        async with models.get_session_contex_manager() as session:
            q_res = await session.execute(query)
        return q_res.scalar_one_or_none()

    async def _add_collection(
        self, collection_name: str, datasource: str | None, embedding_name: str
    ) -> CollectionName:

        item = CollectionName(
            collection_name=collection_name,
            datasource=datasource,
            embedding_model_name=embedding_name,
        )
        self._session.add(item)
        await self._session.commit()
        # await self._session.refresh(item)
        return item

    async def _get_collection_name_or_create(
        self, collection_name: str, datasource: str | None, embedding_name: str
    ) -> CollectionName:
        if name := await self._find_collection(
            self._collection_name, datasource, self._embedding_model.name
        ):
            return name

        return await self._add_collection(collection_name, datasource, embedding_name)

    async def _get_table_name(self) -> str:
        if self._table_name is None:
            collection_name = await self._get_collection_name_or_create(
                self._collection_name, self._datasource, self._embedding_model.name
            )
            self._table_name = collection_name.table_name
        return self._table_name

    async def _get_document_model(self) -> type[BaseDocument]:
        return ModelsStore.get_document_model(
            await self._get_table_name(), self._embedding_model.embedding_length
        )

    async def _get_mapping_table_name(self) -> str:
        return f"{await self._get_table_name()}_mapping"

    async def _get_mapping_model(self) -> type[BaseDatasetDocumentMapping]:
        return ModelsStore.get_dataset_document_mapping_model(await self._get_mapping_table_name())

    async def _check_if_table_exists(self, table_name: str) -> bool:
        """Returns True if the table exists."""

        async with models.get_session_contex_manager() as session:
            result = await session.scalar(
                text(
                    "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=:table_name)"
                ),
                {"table_name": table_name},
            )
        return result

    async def _create_table(self, model: type[BaseModel]):
        query = CreateTable(model.__table__)  # type: ignore
        await self._session.execute(query)
        await self._session.commit()

    async def _create_table_if_not_exist(self, model: type[BaseModel]) -> bool:
        """Returns True if the table was created"""

        table_name = model.__tablename__
        if await self._check_if_table_exists(table_name):
            logger.info(f"Table '{table_name}' exist")
            return False
        else:
            await self._create_table(model)
            logger.info(f"Created table '{table_name}'")
            return True

    async def clear(self) -> None:
        collection_name = await self._find_collection(
            self._collection_name, self._datasource, self._embedding_model.name
        )

        if collection_name is None:
            return

        for table in [collection_name.table_name, f"{collection_name.table_name}_mapping"]:
            if await self._check_if_table_exists(table):
                await self._session.execute(text(f'DROP TABLE collections."{table}"'))
                logger.info(f"Dropped '{table}' table")

        await self._session.delete(collection_name)
        await self._session.commit()

    async def import_documents(
        self, documents: Iterable[EmbeddedDocument], dataset_id: int | None = None
    ) -> None:
        document_model: type[BaseDocument] = await self._get_document_model()

        await self._create_table_if_not_exist(document_model)

        batch: list[EmbeddedDocument]
        for batch in batched(documents, PG_VECTOR_STORE_BATCH_SIZE):
            items = [
                document_model(
                    document=doc.page_content,
                    details=doc.metadata,
                    embeddings=doc.embeddings,
                )
                for doc in batch
            ]

            self._session.add_all(items)
            await self._session.commit()
            logger.info(f"Added {len(items)} documents")

            if dataset_id:
                await self._map_documents_with_dataset(
                    dataset_id=dataset_id,
                    document_ids=[d.id for d in items],
                )

    async def add_documents(
        self, documents: Iterable[Document], dataset_id: int | None = None
    ) -> None:
        table_name: str = await self._get_table_name()
        document_model: type[BaseDocument] = await self._get_document_model()

        await self._create_table_if_not_exist(document_model)

        logger.info(f"Adding new documents to table '{table_name}'...")  # {len(documents)}
        new_documents_models = await self._add_documents(documents)

        if dataset_id:
            await self._map_documents_with_dataset(
                dataset_id=dataset_id,
                document_ids=[d.id for d in new_documents_models],
            )

    async def _add_documents(self, documents: Iterable[Document]) -> list[BaseDocument]:
        document_model: type[BaseDocument] = await self._get_document_model()

        res = []

        for batch in batched(documents, PG_VECTOR_STORE_BATCH_SIZE):
            embeddings = await self._embedding_model.model.aembed_documents(
                [doc.page_content for doc in batch]
            )

            items = []
            for doc, embedding in zip(batch, embeddings):
                items.append(
                    document_model(
                        document=doc.page_content,
                        details=doc.metadata,
                        embeddings=embedding,
                    )
                )

            self._session.add_all(items)
            await self._session.commit()
            logger.info(f"Added {len(items)} documents")

            res.extend(items)
        return res

    async def _map_documents_with_dataset(
        self, dataset_id: int, document_ids: list[int]
    ) -> list[BaseDatasetDocumentMapping]:
        mapping_model: type[BaseDatasetDocumentMapping] = await self._get_mapping_model()

        await self._create_table_if_not_exist(mapping_model)

        items = [
            mapping_model(dataset_id=dataset_id, document_id=doc_id) for doc_id in document_ids
        ]

        for batch in batched(items, PG_VECTOR_STORE_BATCH_SIZE):
            self._session.add_all(batch)
            await self._session.commit()
            logger.info(f"Added {len(batch)} document mappings for dataset(id={dataset_id})")

        return items

    async def remove_documents_by_dataset_id(self, dataset_id: int) -> None:

        document_ids = await self._remove_dataset_mapping(dataset_id)
        logger.info(f"Deleted {len(document_ids)} documents mapping")
        if document_ids:
            await self._clear_documents_without_mapping(document_ids)

    async def _remove_dataset_mapping(self, dataset_id: int) -> list[int]:
        model = await self._get_mapping_model()

        if not await self._check_if_table_exists(model.__tablename__):
            return []

        query = delete(model).where(model.dataset_id == dataset_id).returning(model.document_id)
        res = await self._session.execute(query)
        await self._session.commit()

        return [i for i in res.scalars().all()]

    async def _clear_documents_without_mapping(self, document_ids: list[int]) -> None:
        doc_model: type[BaseDocument] = await self._get_document_model()
        mapping_model: type[BaseDatasetDocumentMapping] = await self._get_mapping_model()

        res = await self._session.scalars(select(mapping_model.document_id))
        mapped_docs = {i for i in res.all()}

        document_ids = [i for i in document_ids if i not in mapped_docs]
        query = delete(doc_model).where(doc_model.id.in_(document_ids))
        await self._session.execute(query)
        await self._session.commit()
        logger.info(f"Deleted {len(document_ids)} documents")

    async def search(self, query: str, k: int = 10) -> list[Document]:
        """For a given query, get its nearest neighbors.

        Docs: https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
        """
        model = await self._get_document_model()
        embedding = (await self._embedding_model.model.aembed_documents([query]))[0]

        # inner product is the same as cosine distance for normalized vectors.
        # OpenAI docs say that computing inner product is faster than cosine distance.
        # However, I didn't notice much difference
        # (however, I didn't measure it thoroughly,
        # only with couple of runs using time.time()).
        # So, let's use explicit cosine distance.
        distance = model.embeddings.cosine_distance(embedding)
        sql_query = select(model).order_by(distance).limit(k)

        async with models.get_session_contex_manager() as session:
            res = await session.scalars(sql_query)
        return [item.to_document() for item in res.all()]

    async def search_with_similarity_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """For a given query, get its nearest neighbors with similarity scores.

        Docs: https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
        """
        model = await self._get_document_model()
        embedding = (await self._embedding_model.model.aembed_documents([query]))[0]

        # inner product is the same as cosine distance for normalized vectors.
        # OpenAI docs say that computing inner product is faster than cosine distance.
        # However, I didn't notice much difference
        # (however, I didn't measure it thoroughly,
        # only with couple of runs using time.time()).
        # So, let's use explicit cosine distance.
        distance = model.embeddings.cosine_distance(embedding)
        sql_query = select(model, distance).order_by(distance).limit(k)

        async with models.get_session_contex_manager() as session:
            res = await session.execute(sql_query)
        # `1 - cosine distance` gives cosine similarity score.
        # NOTE: for other distances, formula to get similarity score is different.
        # For example, for negative inner product we must take negative value: similarity = -distance
        return [(item.to_document(), 1 - distance) for item, distance in res.all()]

    async def search_with_similarity_score_and_dataset_id(
        self, query: str, k: int = 10, dataset_ids: set[int] | None = None
    ) -> list[tuple[Document, float, int]]:
        """
        For a given query, get its nearest neighbors with similarity scores along with the dataset ids it belongs to.
        """
        model = await self._get_document_model()
        mapping_model = await self._get_mapping_model()
        embedding = (await self._embedding_model.model.aembed_documents([query]))[0]
        distance = model.embeddings.cosine_distance(embedding)

        sql_query = select(model, distance, mapping_model.dataset_id).join(
            mapping_model, mapping_model.document_id == model.id
        )
        if dataset_ids:
            sql_query = sql_query.where(mapping_model.dataset_id.in_(dataset_ids))
        sql_query = sql_query.order_by(distance).limit(k)
        async with models.get_session_contex_manager() as session:
            res = await session.execute(sql_query)

        doc_models = [(item, dist, ds_id) for item, dist, ds_id in res.all()]
        # `1 - cosine distance` gives cosine similarity score.
        # NOTE: for other distances, formula to get similarity score is different.
        # see notes for similar functions above.
        return [(item.to_document(), 1 - dist, ds_id) for item, dist, ds_id in doc_models]

    async def get_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
        ids: Iterable[int] | None = None,
        dataset_id: int | None = None,
        include_embeddings: bool = False,
    ) -> list[EmbeddedDocument]:
        model = await self._get_document_model()
        if not await self._check_if_table_exists(model.__tablename__):
            return []

        query = select(model)

        if dataset_id is not None:
            mapping_model = await self._get_mapping_model()
            query = query.join(mapping_model, mapping_model.document_id == model.id)
            query = query.where(mapping_model.dataset_id == dataset_id)

        if ids is not None:
            query = query.where(model.id.in_(ids))

        query = query.limit(limit).offset(offset)

        async with models.get_session_contex_manager() as session:
            res = await session.scalars(query)
        return [item.to_document(include_embeddings) for item in res.all()]

    async def get_dataset_ids_by_documents_ids(self, ids: Iterable[int]) -> list[int]:
        model = await self._get_mapping_model()

        async with models.get_session_contex_manager() as session:
            res = await session.scalars(
                select(model.dataset_id).distinct().where(model.document_id.in_(ids))
            )
        return [item for item in res.all()]

    async def get_document_ids_by_dataset_id(self, dataset_id: int) -> list[int]:
        model = await self._get_mapping_model()

        async with models.get_session_contex_manager() as session:
            res = await session.scalars(
                select(model.document_id).where(model.dataset_id == dataset_id)
            )
        return [item for item in res.all()]
