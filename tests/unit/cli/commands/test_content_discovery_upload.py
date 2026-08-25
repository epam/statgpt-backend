"""Tests for the discovery datasets step of `content init`.

Every file in a client's folder belongs to every channel that client declares, and a file the
API refuses must not stop the ones after it.
"""

import datetime
from typing import Any

import pytest

from statgpt.cli.commands.content import _upload_discovery_datasets
from statgpt.cli.shared.admin_client import DiscoveryPayloadError
from statgpt.cli.shared.batch_report import BatchItemStatus, BatchReport
from statgpt.common.schemas import (
    Channel,
    DiscoveryPayloadErrorDetail,
    DiscoveryPayloadProblem,
    DiscoveryUploadMode,
    DiscoveryUploadSummary,
)

_NOW = datetime.datetime(2026, 1, 1)


def _channel(deployment_id: str, channel_id: int) -> Channel:
    return Channel(
        id=channel_id,
        created_at=_NOW,
        updated_at=_NOW,
        title=deployment_id,
        description="",
        deployment_id=deployment_id,
        llm_model="gpt-4o",
        details={  # type: ignore[arg-type]
            "supremeAgent": {"name": "bot", "domain": "stats", "terminologyDomain": "stats"}
        },
    )


def _channel_cfg(*deployment_ids: str) -> dict[str, Any]:
    return {"channels": [{"deployment_id": deployment_id} for deployment_id in deployment_ids]}


class _StubAdminClient:
    """Records every upload, and refuses the files named in `refuse`."""

    def __init__(self, refuse: set[str] | None = None, channels: list[Channel] | None = None):
        self.uploads: list[tuple[int, str, DiscoveryUploadMode]] = []
        self._refuse = refuse or set()
        self._channels = channels or []

    async def get_channels(self) -> list[Channel]:
        return self._channels

    async def upload_discovery_datasets(
        self, channel_id: int, file_path: str, mode: DiscoveryUploadMode
    ) -> DiscoveryUploadSummary:
        import os

        name = os.path.basename(file_path)
        self.uploads.append((channel_id, name, mode))
        if name in self._refuse:
            raise DiscoveryPayloadError(
                DiscoveryPayloadErrorDetail(
                    message="1 problem(s) found",
                    problems=[
                        DiscoveryPayloadProblem(message="must not be empty", field="agency", row=2)
                    ],
                )
            )
        return DiscoveryUploadSummary(created=1, rows_read=1)


def _write(tmp_path, *names: str):
    directory = tmp_path / "discovery_datasets"
    directory.mkdir()
    for name in names:
        (directory / name).write_text("agency,dataset_id\nBPS,T1\n", encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_every_file_is_uploaded_to_every_channel_of_the_client(tmp_path) -> None:
    directory = _write(tmp_path, "b.csv", "a.xlsx", "notes.txt", "~$a.xlsx")
    channels = {
        "channel-a": _channel("channel-a", 1),
        "channel-b": _channel("channel-b", 2),
    }
    client = _StubAdminClient()
    report = BatchReport(title="t")

    await _upload_discovery_datasets(
        client, report, _channel_cfg("channel-a", "channel-b"), channels, str(directory)
    )

    assert client.uploads == [
        (1, "a.xlsx", DiscoveryUploadMode.UPSERT),
        (1, "b.csv", DiscoveryUploadMode.UPSERT),
        (2, "a.xlsx", DiscoveryUploadMode.UPSERT),
        (2, "b.csv", DiscoveryUploadMode.UPSERT),
    ], "files are uploaded in name order, and only the supported ones"
    assert not report.has_failures


@pytest.mark.asyncio
async def test_a_refused_file_does_not_stop_the_rest(tmp_path) -> None:
    directory = _write(tmp_path, "a.csv", "b.csv")
    client = _StubAdminClient(refuse={"a.csv"})
    report = BatchReport(title="t")

    await _upload_discovery_datasets(
        client,
        report,
        _channel_cfg("channel-a"),
        {"channel-a": _channel("channel-a", 1)},
        str(directory),
    )

    assert [name for _, name, _ in client.uploads] == ["a.csv", "b.csv"]
    by_status = {item.item_id: item.status for item in report.items}
    assert by_status == {
        "channel-a: a.csv": BatchItemStatus.FAILED,
        "channel-a: b.csv": BatchItemStatus.OK,
    }


@pytest.mark.asyncio
async def test_channels_not_processed_in_this_run_are_looked_up(tmp_path) -> None:
    """`--only discovery` never populates the channels dict, so the ids come from the API."""
    directory = _write(tmp_path, "a.csv")
    client = _StubAdminClient(channels=[_channel("channel-a", 42)])
    report = BatchReport(title="t")

    await _upload_discovery_datasets(client, report, _channel_cfg("channel-a"), {}, str(directory))

    assert client.uploads == [(42, "a.csv", DiscoveryUploadMode.UPSERT)]


@pytest.mark.asyncio
async def test_a_channel_that_does_not_exist_yet_is_reported(tmp_path) -> None:
    directory = _write(tmp_path, "a.csv")
    client = _StubAdminClient()
    report = BatchReport(title="t")

    await _upload_discovery_datasets(client, report, _channel_cfg("channel-a"), {}, str(directory))

    assert client.uploads == []
    assert [item.status for item in report.items] == [BatchItemStatus.SKIPPED]


@pytest.mark.asyncio
async def test_a_folder_without_usable_files_uploads_nothing(tmp_path) -> None:
    directory = _write(tmp_path, "README.md")
    client = _StubAdminClient()
    report = BatchReport(title="t")

    await _upload_discovery_datasets(
        client,
        report,
        _channel_cfg("channel-a"),
        {"channel-a": _channel("channel-a", 1)},
        str(directory),
    )

    assert client.uploads == []
    assert report.items == []
