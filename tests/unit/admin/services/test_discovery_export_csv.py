"""Tests for the discovery dataset CSV export.

The CSV an admin downloads has to be the one the channel export writes, because it is fed
straight back into the upload endpoint of another channel.
"""

import csv
import io
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.discovery_dataset import AdminPortalDiscoveryDatasetService
from statgpt.admin.services.discovery_upload import RECORD_FIELDS, parse_discovery_file
from statgpt.common import models, utils
from statgpt.common.schemas import (
    DiscoveryDatasetBase,
    DiscoveryIndexingStatus,
    DiscoveryValidationStatus,
    read_indicators,
)


def _stored(agency: str = "IMF", dataset_id: str = "TABEL1_1", **overrides: object):
    """A stand-in for a stored row, carrying the deployment-local state too.

    The point of several of these tests is that the state does not reach the file.
    """
    values: dict[str, object] = {name: "" for name in RECORD_FIELDS}
    values["indicators_coverage"] = []
    values.update(agency=agency, dataset_id=dataset_id)
    values.update(overrides)
    values.setdefault("id", 1)
    values.setdefault("validation_status", DiscoveryValidationStatus.VALID)
    values.setdefault("indexing_status", DiscoveryIndexingStatus.INDEXED)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _service(records: list[models.DiscoveryDataset]) -> AdminPortalDiscoveryDatasetService:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service.get_record_models_by_channel = AsyncMock(return_value=records)  # type: ignore[method-assign]
    return service


@pytest.mark.asyncio
async def test_columns_are_the_descriptive_fields_in_order() -> None:
    """The upload path reads this file by header, so the order is part of the contract."""
    content = await _service([_stored()]).export_discovery_datasets_to_csv(channel_id=1)

    header = content.splitlines()[0]
    assert header == ",".join(DiscoveryDatasetBase.model_fields)


@pytest.mark.asyncio
async def test_deployment_local_state_is_not_exported() -> None:
    """Validation and indexing state describes the channel that holds the record, not the
    dataset, and is re-derived by the next indexing job."""
    content = await _service([_stored()]).export_discovery_datasets_to_csv(channel_id=1)

    assert "validation_status" not in content
    assert "indexing_status" not in content
    assert "INDEXED" not in content


@pytest.mark.asyncio
async def test_a_slice_that_matches_nothing_is_a_header_on_its_own() -> None:
    """A filtered export has to stay a valid CSV, so an empty result is not an error."""
    content = await _service([]).export_discovery_datasets_to_csv(channel_id=1)

    assert content == ",".join(DiscoveryDatasetBase.model_fields) + "\r\n"


@pytest.mark.asyncio
async def test_the_filters_reach_the_query() -> None:
    service = _service([])

    await service.export_discovery_datasets_to_csv(
        channel_id=7,
        validation_status=DiscoveryValidationStatus.INVALID,
        indexing_status=DiscoveryIndexingStatus.FAILED,
        agency="IMF",
    )

    kwargs = service.get_record_models_by_channel.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["validation_status"] is DiscoveryValidationStatus.INVALID
    assert kwargs["indexing_status"] is DiscoveryIndexingStatus.FAILED
    assert kwargs["agency"] == "IMF"
    assert kwargs["limit"] is None


@pytest.mark.asyncio
async def test_the_download_matches_what_the_channel_export_writes(tmp_path) -> None:
    """The two have to agree byte for byte: an export of one channel is an upload to another."""
    records = [_stored(dataset_id="TABEL1_1"), _stored(agency="OECD", dataset_id="TABEL2_2")]
    service = _service(records)

    downloaded = await service.export_discovery_datasets_to_csv(channel_id=1)

    written = tmp_path / "discovery_datasets.csv"
    utils.write_csv_from_dict_list(service._export_rows(records), str(written))
    assert downloaded.encode("utf-8") == written.read_bytes()


@pytest.mark.asyncio
async def test_an_export_longer_than_pythons_field_limit_uploads_unchanged(
    default_csv_field_limit: int,
) -> None:
    indicators = [{"name": f"indicator {i}", "units": "units"} for i in range(10_000)]
    service = _service([_stored(indicators_coverage=indicators)])

    content = await service.export_discovery_datasets_to_csv(channel_id=1)
    parsed = parse_discovery_file(
        content.encode("utf-8"), "export.csv", max_rows=10, max_indicators_per_dataset=50_000
    )

    assert parsed.rows[0].indicators == read_indicators(indicators)
    assert len(content) > default_csv_field_limit


@pytest.mark.asyncio
async def test_indicators_are_exported_in_the_flat_form() -> None:
    indicators = [
        {"name": "broad money (M2)", "units": "billions of Rp"},
        {"name": "exports; goods", "units": ""},
    ]
    service = _service([_stored(indicators_coverage=indicators)])

    [row] = csv.DictReader(
        io.StringIO(await service.export_discovery_datasets_to_csv(channel_id=1))
    )

    # A ';' would split the indicator in two on the way back in, so it is written as ','.
    assert row["indicators_coverage"] == "broad money (M2) (billions of Rp); exports, goods"
