"""Tests for the upsert/replace mode of a channel import's discovery datasets."""

import csv
import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.discovery_dataset import AdminPortalDiscoveryDatasetService
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common.schemas import DiscoveryUploadSummary


def _make_zip(rows: list[dict[str, str]] | None) -> zipfile.ZipFile:
    """An archive with the records file, or - for `None` - one that does not carry it."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", "{}")
        if rows is not None:
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=list(COLUMN_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(JobsConfig.DISCOVERY_DATASETS_FILE, csv_buffer.getvalue())
    buffer.seek(0)
    return zipfile.ZipFile(buffer, "r")


def _row(agency: str = "IMF", dataset_id: str = "TABEL1_1") -> dict[str, str]:
    row = {name: "" for name in COLUMN_FIELDS}
    row.update(agency=agency, dataset_id=dataset_id)
    return row


def _service() -> AdminPortalDiscoveryDatasetService:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service._upsert = AsyncMock(return_value=DiscoveryUploadSummary())  # type: ignore[method-assign]
    return service


@pytest.mark.asyncio
async def test_upsert_keeps_records_the_archive_does_not_mention() -> None:
    """The default, so today's imports behave as they did."""
    service = _service()

    await service.import_discovery_datasets_from_zip(_make_zip([_row()]), channel_id=1)

    assert service._upsert.await_args.kwargs["delete_absent"] is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_replace_deletes_records_the_archive_does_not_mention() -> None:
    service = _service()

    await service.import_discovery_datasets_from_zip(
        _make_zip([_row()]), channel_id=1, delete_absent=True
    )

    call = service._upsert.await_args  # type: ignore[attr-defined]
    assert call.kwargs["delete_absent"] is True
    assert [candidate.record.dataset_id for candidate in call.args[1]] == ["TABEL1_1"]


@pytest.mark.asyncio
async def test_replace_clears_the_channel_when_the_archive_carries_no_records() -> None:
    """An archive without the file describes a channel with none, so replace empties it."""
    service = _service()

    await service.import_discovery_datasets_from_zip(
        _make_zip(None), channel_id=1, delete_absent=True
    )

    service._upsert.assert_awaited_once_with(1, [], delete_absent=True)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_upsert_leaves_the_channel_alone_when_the_archive_carries_no_records() -> None:
    service = _service()

    await service.import_discovery_datasets_from_zip(_make_zip(None), channel_id=1)

    service._upsert.assert_not_awaited()  # type: ignore[attr-defined]
