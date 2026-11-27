import datetime
import io
import zipfile
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.import_export_jobs import ExportMetadata, JobsService
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import schemas
from statgpt.common.auth.auth_context import AuthContext


def _metadata(version: int | None) -> ExportMetadata:
    return ExportMetadata(
        export_version=version,
        export_start_datetime="2026-11-27T10:00:00",
        export_finish_datetime="2026-11-27T10:01:00",
        scope=schemas.ExportScope.CONFIGS,
        deployment_id="test",
    )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_supported_archive_versions(version):
    JobsService._validate_export_version(_metadata(version))
    assert JobsConfig.CURRENT_EXPORT_VERSION == 3
    assert JobsConfig.DISCOVERY_INDICATORS_JSON_VERSION == 2


async def test_export_job_writes_version_three_metadata(mocker):
    now = datetime.datetime(2026, 1, 1)
    job = SimpleNamespace(
        id=1,
        created_at=now,
        updated_at=now,
        channel_id=1,
        file=None,
        type=schemas.JobType.EXPORT,
        status=schemas.PreprocessingStatusEnum.QUEUED,
        reason_for_failure=None,
    )
    service = JobsService(MagicMock())
    service.get_job_model_by_id = AsyncMock(return_value=job)
    service._export_data_to_folder = AsyncMock(return_value="test")

    async def update_status(item, status):
        item.status = status

    service._update_job_status = AsyncMock(side_effect=update_status)
    metadata = []

    async def upload(path, local_path, **kwargs):
        with zipfile.ZipFile(local_path) as archive:
            metadata.append(ExportMetadata.model_validate_json(archive.read("metadata.json")))
        return SimpleNamespace(url="files/export.zip")

    storage = MagicMock()
    storage.put_local_file = AsyncMock(side_effect=upload)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=storage)
    context.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(
        "statgpt.admin.services.import_export_jobs.attachments_storage_factory",
        return_value=context,
    )

    result = await service.export_channel_in_background(
        1, schemas.ExportScope.CONFIGS, MagicMock(spec=AuthContext)
    )

    assert result.status is schemas.PreprocessingStatusEnum.COMPLETED
    assert len(metadata) == 1
    assert metadata[0].export_version == 3
    assert metadata[0].scope is schemas.ExportScope.CONFIGS


@pytest.mark.parametrize(
    "version,supported,discovery",
    [
        (3, {1, 2}, False),
        (4, {1, 2, 3}, False),
        (None, {1, 2, 3}, False),
        (1, {1, 2, 3}, True),
    ],
)
async def test_incompatible_archive_is_rejected_before_any_import(
    mocker, version, supported, discovery
):
    mocker.patch.object(JobsConfig, "SUPPORTED_EXPORT_VERSIONS", supported)

    @asynccontextmanager
    async def session_context():
        yield MagicMock()

    mocker.patch(
        "statgpt.admin.services.import_export_jobs.models.get_session_context_manager",
        session_context,
    )
    channel_service = mocker.patch("statgpt.admin.services.import_export_jobs.ChannelService")
    dataset_service = mocker.patch("statgpt.admin.services.import_export_jobs.DataSetService")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", _metadata(version).model_dump_json())
        if discovery:
            archive.writestr(JobsConfig.DISCOVERY_DATASETS_FILE, "")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        await JobsService(MagicMock())._import_data_from_zip(
            MagicMock(),
            archive,
            False,
            True,
            True,
            schemas.RecordUploadMode.UPSERT,
            schemas.RecordUploadMode.UPSERT,
            MagicMock(spec=AuthContext),
        )

    channel_service.assert_not_called()
    dataset_service.assert_not_called()
