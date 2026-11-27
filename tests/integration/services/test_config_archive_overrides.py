import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import yaml
from sqlalchemy import select

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.admin.services.import_export_jobs import ExportMetadata, JobsService
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import models, schemas
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig, QuanthubSdmxDataSourceConfig

from .test_sdmx_datasets import DIMENSIONS, get_channel, get_data_source


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize(
    "overrides", [{}, {"isOfficial": False, "citation": None, "pinnedColumns": []}]
)
async def test_config_archive_preserves_explicitness_and_import_idempotence(
    session,
    async_session_factory,
    monkeypatch,
    clear_all,
    sdmx_clint_mock,
    tmp_path,
    version,
    overrides,
):
    @asynccontextmanager
    async def session_context():
        async with async_session_factory() as isolated_session:
            yield isolated_session

    monkeypatch.setattr(models, "get_session_context_manager", session_context)
    monkeypatch.setattr(models, "get_readonly_session_context_manager", session_context)
    auth = SystemUserAuthContext()
    source = await get_data_source(session)
    channel = await get_channel(session)
    service = DataSetService(session)
    details = {
        "urn": {"agencyId": "IMF.STA", "resourceId": "CPI"},
        "dimensions": DIMENSIONS,
        "useTitleFromSrc": False,
        **overrides,
    }
    dataset = await service.create_dataset(
        schemas.DataSetBase(title="CPI", data_source_id=source.id, details=details), auth
    )
    await service.add_dataset_to_channel(channel.id, dataset.id)
    await session.commit()

    await JobsService._export_data_to_folder(
        channel.id, str(tmp_path), schemas.ExportScope.CONFIGS, auth
    )
    exported_channel = yaml.safe_load((tmp_path / JobsConfig.CHANNEL_FILE).read_text())
    exported_source = yaml.safe_load((tmp_path / JobsConfig.DATA_SOURCES_FILE).read_text())[
        "dataSources"
    ][0]
    exported_dataset = yaml.safe_load((tmp_path / JobsConfig.DATASETS_FILE).read_text())[
        "dataSets"
    ][0]
    assert exported_dataset["details"] == details
    assert exported_source["details"] == source.details
    assert "locale" not in exported_channel["details"]
    assert "max_agent_iterations" not in exported_channel["details"]["supreme_agent"]

    if version < 3:
        exported_channel["details"] = schemas.ChannelConfig.model_validate(
            exported_channel["details"]
        ).model_dump(mode="json", by_alias=True)
        exported_source["details"] = QuanthubSdmxDataSourceConfig.model_validate(
            exported_source["details"]
        ).model_dump(mode="json", by_alias=True)
        exported_dataset["details"] = QuanthubDataSetConfig.model_validate(
            exported_dataset["details"]
        ).model_dump(mode="json", by_alias=True)
    expected_channel = schemas.ChannelConfig.model_validate(exported_channel["details"]).model_dump(
        mode="json", by_alias=True, exclude_unset=True
    )
    expected_source = exported_source["details"]
    expected_dataset = exported_dataset["details"]
    imported_uuid = uuid.uuid4()
    exported_channel["deployment_id"] = "archive-channel"
    exported_source["title"] = "archive-source"
    exported_dataset["dataSource"] = "archive-source"
    exported_dataset["id_"] = str(imported_uuid)
    metadata = ExportMetadata(
        export_version=version,
        export_start_datetime=datetime.now().isoformat(),
        export_finish_datetime=datetime.now().isoformat(),
        scope=schemas.ExportScope.CONFIGS,
        deployment_id="archive-channel",
    )
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata.json", metadata.model_dump_json())
        archive.writestr(JobsConfig.CHANNEL_FILE, yaml.safe_dump(exported_channel))
        archive.writestr(
            JobsConfig.DATA_SOURCES_FILE, yaml.safe_dump({"dataSources": [exported_source]})
        )
        archive.writestr(JobsConfig.DATASETS_FILE, yaml.safe_dump({"dataSets": [exported_dataset]}))

    jobs = JobsService(session)
    jobs._update_job_status = AsyncMock()
    with zipfile.ZipFile(archive_path) as archive:
        imported_id, deduplicate = await jobs._import_data_from_zip(
            models.Job(),
            archive,
            False,
            True,
            True,
            schemas.RecordUploadMode.UPSERT,
            schemas.RecordUploadMode.UPSERT,
            auth,
        )
    assert not deduplicate
    await session.commit()
    imported_channel = await session.get(models.Channel, imported_id)
    imported_source = await session.scalar(
        select(models.DataSource).where(models.DataSource.title == "archive-source")
    )
    imported_dataset = await session.scalar(
        select(models.DataSet).where(models.DataSet.id_ == imported_uuid)
    )
    assert imported_channel.details == expected_channel
    assert imported_source.details == expected_source
    assert imported_dataset.details == expected_dataset
    source_updated_at, dataset_updated_at = imported_source.updated_at, imported_dataset.updated_at
    with zipfile.ZipFile(archive_path) as archive:
        repeated_id, _ = await jobs._import_data_from_zip(
            models.Job(),
            archive,
            False,
            True,
            True,
            schemas.RecordUploadMode.UPSERT,
            schemas.RecordUploadMode.UPSERT,
            auth,
        )
    assert repeated_id == imported_id
    await session.refresh(imported_source)
    await session.refresh(imported_dataset)
    assert imported_source.updated_at == source_updated_at
    assert imported_dataset.updated_at == dataset_updated_at
    assert imported_dataset.id_ == imported_uuid
    assert imported_dataset.details == expected_dataset


@pytest.mark.asyncio
async def test_legacy_refresh_persists_complete_new_snapshot_without_rewriting_history(
    session, clear_all, sdmx_clint_mock, mocker, tmp_path
):
    auth = SystemUserAuthContext()
    source = await get_data_source(session)
    channel = await get_channel(session)
    service = DataSetService(session)
    sparse = {
        "urn": {"agencyId": "IMF.STA", "resourceId": "CPI"},
        "dimensions": DIMENSIONS,
        "useTitleFromSrc": False,
    }
    effective = QuanthubDataSetConfig.model_validate(sparse)
    legacy_live = effective.model_dump(mode="json", by_alias=True)
    dataset = await service.create_dataset(
        schemas.DataSetBase(title="CPI", data_source_id=source.id, details=legacy_live), auth
    )
    channel_dataset = await service.add_dataset_to_channel(channel.id, dataset.id)
    old = models.ChannelDatasetVersion(
        channel_dataset_id=channel_dataset.id,
        preprocessing_status=schemas.PreprocessingStatusEnum.COMPLETED,
        creation_reason="Legacy index",
        indexing_config_hash=effective.indexing_hash,
        structure_hash="1234567890",
        resolved_config=None,
    )
    session.add(old)
    await session.commit()
    await session.refresh(old)
    old_id = old.id
    handler = await service._get_handler(source.id)
    service._get_handler = AsyncMock(return_value=handler)
    mocker.patch.object(
        handler, "get_structure_hash_and_metadata", AsyncMock(return_value=("1234567890", {}))
    )
    resolve = mocker.spy(handler, "resolve_config")

    response = await service.update(dataset.id, schemas.DataSetUpdateRequest(details=sparse), auth)
    await session.commit()
    assert response.channel_results[0].status is schemas.ChannelDatasetUpdateStatus.AUTO_UPDATED
    snapshot = response.channel_results[0].new_version
    assert snapshot.resolved_config["isOfficial"] is False
    assert snapshot.resolved_config["urn"]["version"] == "4.0.0"
    assert snapshot.resolved_config["dimensions"]["COICOP_1999"]["isRequired"] is False
    assert snapshot.pointer_to == old_id
    assert snapshot.indexing_config_hash == effective.indexing_hash
    resolve.assert_awaited_once()
    await session.refresh(old)
    assert old.resolved_config is None
    assert old.creation_reason == "Legacy index"
    assert (
        await session.scalar(select(models.DataSet.details).where(models.DataSet.id == dataset.id))
        == sparse
    )
    persisted = await session.get(models.ChannelDatasetVersion, snapshot.id)
    assert persisted.resolved_config == snapshot.resolved_config
    service._export_versions({dataset.id_: snapshot}, str(tmp_path))
    exported = yaml.safe_load((tmp_path / JobsConfig.VERSIONS_FILE).read_text())
    assert exported["data"][str(dataset.id_)]["resolved_config"] == snapshot.resolved_config
