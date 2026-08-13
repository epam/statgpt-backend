"""Tests that one failing dataset does not truncate a `content init` batch.

The customer's reproduction, as a test: several valid dataset configs plus one that fails.
Every valid dataset must be processed and the bad one reported, whatever order the config
files happen to be read in.
"""

import datetime
import os
import uuid
from typing import Any

import pytest
import yaml

from statgpt.cli.commands.content import _process_datasets, _verify_datasets
from statgpt.cli.shared.batch_report import BatchReport
from statgpt.common.schemas import Channel, DataSet, DataSource, DataSourceType
from statgpt.common.schemas.dataset import Status

_NOW = datetime.datetime(2026, 1, 1)
_SOURCE_TITLE = "sdmx-source"


def _urn(resource_id: str) -> dict[str, Any]:
    return {"agency_id": "AG", "resource_id": resource_id, "version": "1.0"}


def _dataset_cfg(resource_id: str, *, source: str | None = _SOURCE_TITLE) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "title": resource_id,
        "details": {"urn": _urn(resource_id)},
        "channels": ["channel-a"],
    }
    if source is not None:
        cfg["dataSource"] = source
    return cfg


def _data_source() -> DataSource:
    return DataSource(
        id=1,
        created_at=_NOW,
        updated_at=_NOW,
        title=_SOURCE_TITLE,
        description="",
        type_id=3,
        details={},
        type=DataSourceType(id=3, created_at=_NOW, updated_at=_NOW, name="SDMX21", description=""),
    )


def _dataset(title: str, *, status: str = "online", details: str = "") -> DataSet:
    return DataSet(
        id=abs(hash(title)) % 10_000,
        id_=uuid.uuid5(uuid.NAMESPACE_URL, title),
        created_at=_NOW,
        updated_at=_NOW,
        title=title,
        description="",
        data_source_id=1,
        details={"urn": _urn(title)},
        data_source=None,
        status=Status(status=status, details=details),
    )


class _StubAdminClient:
    """Minimal stand-in for AdminClient, recording what the batch actually attempted."""

    def __init__(self, failing_titles: set[str] | None = None):
        self._failing_titles = failing_titles or set()
        self.created: list[str] = []
        self.linked: list[tuple[int, int]] = []
        self.all_datasets: list[DataSet] = []

    async def create_dataset(self, ds_cfg: dict[str, Any]) -> DataSet:
        title = ds_cfg["title"]
        if title in self._failing_titles:
            raise RuntimeError(f"unsupported DSD for {title}")
        self.created.append(title)
        return _dataset(title)

    async def get_channel_datasets(self, channel_id: int) -> list[Any]:
        return []

    async def add_dataset_to_channel(self, channel_id: int, dataset_id: int) -> None:
        self.linked.append((channel_id, dataset_id))

    async def get_datasets(self, channel_id: int | None = None, limit: int = 5000):
        return self.all_datasets


def _write_configs(datasets_dir: str, configs: dict[str, list[dict[str, Any]]]) -> None:
    """Write one YAML file per entry, so filename order is under the test's control."""
    os.makedirs(datasets_dir, exist_ok=True)
    for filename, datasets in configs.items():
        with open(os.path.join(datasets_dir, filename), "w", encoding="utf-8") as f:
            yaml.safe_dump({"dataSets": datasets}, f)


async def _run(
    tmp_path,
    configs: dict[str, list[dict[str, Any]]],
    *,
    client: _StubAdminClient,
    channels: dict[str, Any] | None = None,
    data_sources: dict[str, DataSource] | None = None,
    link_channels: bool = True,
) -> tuple[BatchReport, list[tuple[str, DataSet]]]:
    _write_configs(os.path.join(str(tmp_path), "datasets"), configs)
    report = BatchReport(title="Summary")

    touched = await _process_datasets(
        client,  # type: ignore[arg-type]
        report,
        "client-a",
        str(tmp_path),
        {_SOURCE_TITLE: _data_source()} if data_sources is None else data_sources,
        {"channel-a": _channel()} if channels is None else channels,
        {},
        None,
        link_channels=link_channels,
    )
    return report, touched


def _channel() -> Channel:
    return Channel(
        id=11,
        created_at=_NOW,
        updated_at=_NOW,
        deployment_id="channel-a",
        title="Channel A",
        description="",
        llm_model="gpt-4o",
        details={  # type: ignore[arg-type]
            "supremeAgent": {"name": "bot", "domain": "stats", "terminologyDomain": "stats"}
        },
    )


@pytest.mark.asyncio
async def test_a_failing_dataset_does_not_stop_the_ones_after_it(tmp_path) -> None:
    client = _StubAdminClient(failing_titles={"DF_B"})

    report, _ = await _run(
        tmp_path,
        {
            "a.yaml": [_dataset_cfg("DF_A")],
            "b.yaml": [_dataset_cfg("DF_B")],
            "c.yaml": [_dataset_cfg("DF_C")],
            "d.yaml": [_dataset_cfg("DF_D")],
        },
        client=client,
    )

    assert client.created == ["DF_A", "DF_C", "DF_D"]
    assert [item.item_id for item in report.failed] == ["AG:DF_B(1.0)"]
    assert "unsupported DSD for DF_B" in (report.failed[0].message or "")
    assert report.has_failures


@pytest.mark.asyncio
async def test_the_report_does_not_depend_on_config_file_order(tmp_path) -> None:
    """Renaming the failing config used to change which datasets got onboarded."""
    forward = _StubAdminClient(failing_titles={"DF_B"})
    report_forward, _ = await _run(
        tmp_path / "forward",
        {
            "01_a.yaml": [_dataset_cfg("DF_A")],
            "02_b.yaml": [_dataset_cfg("DF_B")],
            "03_c.yaml": [_dataset_cfg("DF_C")],
        },
        client=forward,
    )

    reverse = _StubAdminClient(failing_titles={"DF_B"})
    report_reverse, _ = await _run(
        tmp_path / "reverse",
        {
            "01_c.yaml": [_dataset_cfg("DF_C")],
            "02_b.yaml": [_dataset_cfg("DF_B")],
            "03_a.yaml": [_dataset_cfg("DF_A")],
        },
        client=reverse,
    )

    assert sorted(forward.created) == sorted(reverse.created) == ["DF_A", "DF_C"]
    assert {(i.item_id, i.status) for i in report_forward.items} == {
        (i.item_id, i.status) for i in report_reverse.items
    }


@pytest.mark.asyncio
async def test_a_dataset_whose_data_source_is_unavailable_is_skipped_not_created(tmp_path) -> None:
    """Creating it without `data_source_id` would leave it unusable, so do not try."""
    client = _StubAdminClient()

    report, _ = await _run(
        tmp_path,
        {"a.yaml": [_dataset_cfg("DF_A"), _dataset_cfg("DF_B", source="missing-source")]},
        client=client,
    )

    assert client.created == ["DF_A"]
    assert not report.has_failures, "a knock-on skip is not a failure of the dataset itself"
    assert [item.item_id for item in report.skipped] == ["AG:DF_B(1.0)"]
    assert "missing-source" in (report.skipped[0].message or "")


@pytest.mark.asyncio
async def test_a_dataset_with_no_data_source_key_is_still_processed(tmp_path) -> None:
    """Configs may carry `data_source_id` directly; that path must keep working."""
    client = _StubAdminClient()

    report, _ = await _run(tmp_path, {"a.yaml": [_dataset_cfg("DF_A", source=None)]}, client=client)

    assert client.created == ["DF_A"]
    assert not report.has_failures
    assert not report.skipped


@pytest.mark.asyncio
async def test_an_unavailable_channel_is_recorded_rather_than_passed_over(tmp_path) -> None:
    """An unlinked dataset is invisible to the chat backend; that must not be silent."""
    client = _StubAdminClient()

    report, _ = await _run(tmp_path, {"a.yaml": [_dataset_cfg("DF_A")]}, client=client, channels={})

    assert client.created == ["DF_A"]
    assert client.linked == []
    assert [item.item_id for item in report.skipped] == ["AG:DF_A(1.0) -> channel-a"]


@pytest.mark.asyncio
async def test_no_channel_noise_when_channels_were_not_requested(tmp_path) -> None:
    """`--only datasets` does not process channels, so absent links are expected."""
    client = _StubAdminClient()

    report, _ = await _run(
        tmp_path, {"a.yaml": [_dataset_cfg("DF_A")]}, client=client, link_channels=False
    )

    assert client.created == ["DF_A"]
    assert not report.skipped


@pytest.mark.asyncio
async def test_an_invalid_urn_is_reported_against_the_dataset(tmp_path) -> None:
    client = _StubAdminClient()

    report, _ = await _run(
        tmp_path,
        {"a.yaml": [{"title": "DF_BAD", "details": {"urn": {"nope": 1}}}, _dataset_cfg("DF_A")]},
        client=client,
    )

    assert client.created == ["DF_A"]
    assert [item.item_id for item in report.failed] == ["DF_BAD"]
    assert "cannot read config" in (report.failed[0].message or "")


@pytest.mark.asyncio
async def test_a_null_details_block_does_not_abort_the_batch(tmp_path) -> None:
    """`details:` with no value reads back as None, and `None.get(...)` would raise.

    A dataset with no URN is tolerated (it is simply labelled by title), so the point here
    is that reading the config cannot throw its way out of the loop.
    """
    client = _StubAdminClient()

    report, _ = await _run(
        tmp_path,
        {"a.yaml": [{"title": "DF_NO_URN", "details": None}, _dataset_cfg("DF_A")]},
        client=client,
    )

    assert client.created == ["DF_NO_URN", "DF_A"]
    assert not report.has_failures
    assert [item.item_id for item in report.items] == ["DF_NO_URN", "AG:DF_A(1.0)"]


@pytest.mark.asyncio
async def test_a_failed_link_does_not_lose_the_dataset(tmp_path) -> None:
    client = _StubAdminClient()

    async def _boom(channel_id: int, dataset_id: int) -> None:
        raise RuntimeError("link rejected")

    client.add_dataset_to_channel = _boom  # type: ignore[method-assign]

    report, touched = await _run(tmp_path, {"a.yaml": [_dataset_cfg("DF_A")]}, client=client)

    assert client.created == ["DF_A"]
    assert [label for label, _ in touched] == ["AG:DF_A(1.0)"]
    assert [item.item_id for item in report.failed] == ["AG:DF_A(1.0) -> channel-a"]


@pytest.mark.asyncio
async def test_verification_reports_a_dataset_that_cannot_be_loaded() -> None:
    """Registered and linked, but unloadable - it would never index and never answer."""
    client = _StubAdminClient()
    unloadable = _dataset("DF_B", status="offline", details="failed to load the dataflow")
    client.all_datasets = [_dataset("DF_A"), unloadable]

    report = BatchReport(title="Summary")
    await _verify_datasets(
        client,  # type: ignore[arg-type]
        report,
        [("AG:DF_A(1.0)", _dataset("DF_A")), ("AG:DF_B(1.0)", unloadable)],
    )

    assert [item.item_id for item in report.failed] == ["AG:DF_B(1.0)"]
    assert "failed to load the dataflow" in (report.failed[0].message or "")


@pytest.mark.asyncio
async def test_verification_is_quiet_when_every_dataset_is_online() -> None:
    client = _StubAdminClient()
    client.all_datasets = [_dataset("DF_A")]

    report = BatchReport(title="Summary")
    await _verify_datasets(
        client, report, [("AG:DF_A(1.0)", _dataset("DF_A"))]  # type: ignore[arg-type]
    )

    assert not report.items
