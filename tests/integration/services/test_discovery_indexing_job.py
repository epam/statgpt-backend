from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from statgpt.admin.services import discovery_indexing_job as job_module
from statgpt.admin.services import discovery_publisher as publisher_module
from statgpt.admin.services.discovery_indexing_job import AdminPortalDiscoveryIndexingJobService
from statgpt.admin.services.discovery_publisher import build_metadata, document_filename
from statgpt.admin.services.discovery_summarization import SummarizeOutcome
from statgpt.common import models, schemas
from statgpt.common.schemas.generic_rag import GenericRagDocumentStatus
from statgpt.common.services import GenericRagIngestionClient

_APPLICATION = "discovery-indexing-test-records"
_AREA_APPLICATION = "discovery-indexing-test-areas"
_PREVIOUS_INDEXED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_UPLOAD_ERROR = "record upload rejected"
_VOCABULARY_ERROR = "vocabulary application unreachable"


@dataclass(frozen=True)
class _Catalog:
    channel_id: int
    deployment_id: str
    job_id: int
    published_id: int
    withdrawn_id: int
    failed_id: int


def _record(channel_id: int, dataset_id: str) -> models.DiscoveryDataset:
    return models.DiscoveryDataset(
        channel_id=channel_id,
        reference_area="Germany",
        regional_coverage="",
        agency="Test statistics agency",
        dataset_id=dataset_id,
        name=dataset_id,
        description="Annual statistical observations.",
        url="https://stats.example.org/datasets",
        time_coverage="2020-2026",
        frequency_coverage="Annual",
        indicators_coverage=[schemas.DiscoveryIndicator(name="Value", units="units").model_dump()],
        indicators_summary="Value (units)",
        missing_indicators="",
        indexing_status=schemas.DiscoveryIndexingStatus.NEW,
    )


@pytest_asyncio.fixture
async def database(
    async_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = async_session_factory.kw["bind"]
    try:
        async with async_session_factory() as session:
            assert await session.scalar(text("SELECT current_database()")) == "test_db"
        yield async_session_factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def catalog(
    database: async_sessionmaker[AsyncSession],
    initial_status: schemas.DiscoveryIndexingStatus,
) -> AsyncIterator[_Catalog]:
    config = schemas.ChannelConfig(
        supreme_agent=schemas.SupremeAgentConfig(
            name="Test agent",
            domain="Statistics",
            terminology_domain="Statistics",
            language_instructions=["Respond in English."],
        ),
        discovery_datasets=schemas.DiscoveryDatasetsTool(
            name="discovery_datasets",
            description="Discovery indexing integration test.",
            details=schemas.DiscoveryDatasetsDetails(
                applicationId=_APPLICATION,
                referenceAreaApplicationId=_AREA_APPLICATION,
                templates=schemas.DiscoveryDatasetsTemplates(wrapper="{items}", item="{name}"),
            ),
        ),
    )
    async with database() as session:
        channel = models.Channel(
            title="Discovery indexing integration test",
            description="Integration test",
            deployment_id=f"discovery-indexing-test-{uuid4()}",
            details=config.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        session.add(channel)
        await session.flush()

        published = _record(channel.id, "published")
        published.indexing_status = initial_status
        published.index_error = "Previous publishing error"
        if initial_status is schemas.DiscoveryIndexingStatus.OUTDATED:
            published.indexed_at = _PREVIOUS_INDEXED_AT
        withdrawn = _record(channel.id, "withdrawn")
        withdrawn.description = ""
        withdrawn.indexing_status = schemas.DiscoveryIndexingStatus.INDEXED
        withdrawn.indexed_at = _PREVIOUS_INDEXED_AT
        withdrawn.index_error = "Previous withdrawal error"
        failed = _record(channel.id, "upload-failure")
        job = models.DiscoveryIndexingJob(
            channel_id=channel.id, status=schemas.PreprocessingStatusEnum.QUEUED
        )
        session.add_all([published, withdrawn, failed, job])
        await session.commit()
        seeded = _Catalog(
            channel.id, channel.deployment_id, job.id, published.id, withdrawn.id, failed.id
        )

    try:
        yield seeded
    finally:
        async with database() as session:
            await session.execute(
                delete(models.Channel).where(models.Channel.id == seeded.channel_id)
            )
            await session.commit()


def _document(
    document_id: int, record: models.DiscoveryDataset, deployment_id: str
) -> schemas.GenericRagDocument:
    return schemas.GenericRagDocument(
        id=document_id,
        display_name=document_filename(record, deployment_id, schemas.DiscoveryGrade.C),
        metadata=build_metadata(record, deployment_id, schemas.DiscoveryGrade.C).model_dump(),
        status=GenericRagDocumentStatus.READY,
    )


@pytest.mark.parametrize(
    "initial_status",
    [schemas.DiscoveryIndexingStatus.NEW, schemas.DiscoveryIndexingStatus.OUTDATED],
)
@pytest.mark.parametrize("vocabulary_failure", ["indexing", "transport"])
async def test_record_results_survive_vocabulary_failure(
    database: async_sessionmaker[AsyncSession],
    catalog: _Catalog,
    initial_status: schemas.DiscoveryIndexingStatus,
    vocabulary_failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = {20: _document(20, _record(catalog.channel_id, "withdrawn"), catalog.deployment_id)}
    if initial_status is schemas.DiscoveryIndexingStatus.OUTDATED:
        documents[10] = _document(
            10, _record(catalog.channel_id, "published"), catalog.deployment_id
        )

    records_client = AsyncMock(spec=GenericRagIngestionClient)
    records_client.__aenter__.return_value = records_client
    records_client.get_metadata_schema.return_value = schemas.GenericRagMetadataSchema(
        schema=schemas.DiscoveryDocumentMetadata.channel_json_schema()
    )
    records_client.list_documents.side_effect = lambda: list(documents.values())
    records_client.delete_document.side_effect = documents.pop

    def store(
        document_id: int,
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        metadata: schemas.DiscoveryDocumentMetadata,
    ) -> schemas.GenericRagDocument:
        document = schemas.GenericRagDocument(
            id=document_id,
            display_name=filename,
            mime_type=mime_type,
            metadata=metadata.model_dump(),
            status=GenericRagDocumentStatus.READY,
        )
        documents[document_id] = document
        return document

    def upload(
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        metadata: schemas.DiscoveryDocumentMetadata,
        overwrite: bool,
    ) -> schemas.GenericRagDocument:
        if metadata.dataset_id == "upload-failure":
            raise RuntimeError(_UPLOAD_ERROR)
        return store(
            max(documents, default=0) + 1,
            filename=filename,
            content=content,
            mime_type=mime_type,
            metadata=metadata,
        )

    records_client.upload_document.side_effect = upload
    records_client.update_document.side_effect = store

    vocabulary_client = AsyncMock(spec=GenericRagIngestionClient)
    vocabulary_client.__aenter__.return_value = vocabulary_client
    vocabulary_client.get_metadata_schema.return_value = schemas.GenericRagMetadataSchema(
        schema=schemas.ReferenceAreaDocumentMetadata.channel_json_schema()
    )
    vocabulary_client.list_documents.return_value = []
    if vocabulary_failure == "indexing":
        vocabulary_client.upload_document.return_value = schemas.GenericRagDocument(
            id=99, status=GenericRagDocumentStatus.ERROR
        )
    else:
        vocabulary_client.list_documents.side_effect = RuntimeError(_VOCABULARY_ERROR)

    clients = {_APPLICATION: records_client, _AREA_APPLICATION: vocabulary_client}
    monkeypatch.setattr(GenericRagIngestionClient, "for_application", clients.__getitem__)
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=SummarizeOutcome(unchanged=2))
    monkeypatch.setattr(
        job_module, "DiscoveryIndicatorSummarizer", MagicMock(return_value=summarizer)
    )
    published_at = datetime.now(UTC)
    monkeypatch.setattr(publisher_module, "get_ts_utcnow", lambda: published_at)

    async with database() as session:
        rollback = AsyncMock(wraps=session.rollback)
        monkeypatch.setattr(session, "rollback", rollback)
        await AdminPortalDiscoveryIndexingJobService(session).process_job(catalog.job_id)
        rollback.assert_awaited_once()

    async with database() as session:
        job = await session.get(models.DiscoveryIndexingJob, catalog.job_id)
        published = await session.get(models.DiscoveryDataset, catalog.published_id)
        withdrawn = await session.get(models.DiscoveryDataset, catalog.withdrawn_id)
        failed = await session.get(models.DiscoveryDataset, catalog.failed_id)
        assert (
            job is not None
            and published is not None
            and withdrawn is not None
            and failed is not None
        )
        assert job.status is schemas.PreprocessingStatusEnum.FAILED
        assert job.reason_for_failure is not None
        if vocabulary_failure == "indexing":
            assert "'Germany'" in job.reason_for_failure
            vocabulary_client.upload_document.assert_awaited_once()
        else:
            assert _VOCABULARY_ERROR in job.reason_for_failure
            vocabulary_client.upload_document.assert_not_awaited()
        assert published.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED
        assert published.indexed_at == published_at
        assert published.index_error is None
        assert withdrawn.validation_status is schemas.DiscoveryValidationStatus.INVALID
        assert withdrawn.indexing_status is schemas.DiscoveryIndexingStatus.NEW
        assert withdrawn.indexed_at is None
        assert withdrawn.index_error is None
        assert failed.indexing_status is schemas.DiscoveryIndexingStatus.FAILED
        assert failed.indexed_at is None
        assert failed.index_error == f"RuntimeError: {_UPLOAD_ERROR}"

    records_client.delete_document.assert_awaited_once_with(20)
    assert {document.metadata["dataset_id"] for document in documents.values()} == {"published"}
