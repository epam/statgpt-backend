"""Tests for the discovery datasets step of `content init`.

Every file in a client's folder belongs to every channel that client declares. The folder is
sent as one all-or-nothing request per channel, and a channel that refuses it must not stop
the channels after it.
"""

import datetime
import os
from collections.abc import Sequence
from typing import Any

import pytest

from statgpt.cli.commands.content import _upload_discovery_datasets
from statgpt.cli.shared.admin_client import DiscoveryPayloadError
from statgpt.cli.shared.batch_report import BatchItemStatus, BatchReport
from statgpt.common.schemas import (
    Channel,
    DiscoveryPayloadErrorDetail,
    DiscoveryPayloadProblem,
    DiscoveryUploadSummary,
    RecordUploadMode,
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
    """Records every upload, and refuses a batch holding a file named in `refuse`."""

    def __init__(self, refuse: set[str] | None = None, channels: list[Channel] | None = None):
        self.uploads: list[tuple[int, list[str], RecordUploadMode]] = []
        self._refuse = refuse or set()
        self._channels = channels or []

    async def get_channels(self) -> list[Channel]:
        return self._channels

    async def upload_discovery_datasets(
        self, channel_id: int, file_paths: Sequence[str], mode: RecordUploadMode
    ) -> DiscoveryUploadSummary:
        names = [os.path.basename(path) for path in file_paths]
        self.uploads.append((channel_id, names, mode))
        if self._refuse & set(names):
            raise DiscoveryPayloadError(
                DiscoveryPayloadErrorDetail(
                    message="1 problem(s) found",
                    problems=[
                        DiscoveryPayloadProblem(
                            message="must not be empty", file="b.csv", field="agency", row=2
                        )
                    ],
                )
            )
        return DiscoveryUploadSummary(created=len(names), rows_read=len(names))


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
        (1, ["a.xlsx", "b.csv"], RecordUploadMode.UPSERT),
        (2, ["a.xlsx", "b.csv"], RecordUploadMode.UPSERT),
    ], "one request per channel, files in name order, and only the supported ones"
    assert [item.item_id for item in report.items] == [
        "channel-a: 2 discovery file(s)",
        "channel-b: 2 discovery file(s)",
    ]
    assert not report.has_failures


@pytest.mark.asyncio
async def test_one_bad_file_fails_the_whole_batch_and_the_run_goes_on(tmp_path) -> None:
    directory = _write(tmp_path, "a.xlsx", "b.csv")
    client = _StubAdminClient(refuse={"b.csv"})
    report = BatchReport(title="t")

    await _upload_discovery_datasets(
        client,
        report,
        _channel_cfg("channel-a", "channel-b"),
        {"channel-a": _channel("channel-a", 1), "channel-b": _channel("channel-b", 2)},
        str(directory),
    )

    assert [channel_id for channel_id, _, _ in client.uploads] == [1, 2]
    by_status = {item.item_id: item.status for item in report.items}
    assert by_status == {
        "channel-a: 2 discovery file(s)": BatchItemStatus.FAILED,
        "channel-b: 2 discovery file(s)": BatchItemStatus.FAILED,
    }


@pytest.mark.asyncio
async def test_channels_not_processed_in_this_run_are_looked_up(tmp_path) -> None:
    """`--only discovery` never populates the channels dict, so the ids come from the API."""
    directory = _write(tmp_path, "a.csv")
    client = _StubAdminClient(channels=[_channel("channel-a", 42)])
    report = BatchReport(title="t")

    await _upload_discovery_datasets(client, report, _channel_cfg("channel-a"), {}, str(directory))

    assert client.uploads == [(42, ["a.csv"], RecordUploadMode.UPSERT)]


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
