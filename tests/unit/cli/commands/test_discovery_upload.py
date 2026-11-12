"""Tests for `discovery upload`: which files one invocation sends, and how the result reads.

The point of a batch is that the admin can name a folder or a glob and get every file in it
uploaded together, so what matters is the expansion - and that a mistake in it costs nothing.
"""

import datetime
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from statgpt.cli.commands import discovery as discovery_module
from statgpt.cli.commands.discovery import upload_command, upload_handler
from statgpt.cli.shared.discovery import render_payload_problems, render_upload_summary
from statgpt.common.schemas import (
    Channel,
    DiscoveryPayloadErrorDetail,
    DiscoveryPayloadProblem,
    DiscoveryUploadFileSummary,
    DiscoveryUploadSummary,
    RecordUploadMode,
)

_NOW = datetime.datetime(2026, 1, 1)
_DEPLOYMENT_ID = "my-channel"


def _channel() -> Channel:
    return Channel(
        id=7,
        created_at=_NOW,
        updated_at=_NOW,
        title="My Channel",
        description="",
        deployment_id=_DEPLOYMENT_ID,
        llm_model="gpt-4o",
        details={  # type: ignore[arg-type]
            "supremeAgent": {"name": "bot", "domain": "stats", "terminologyDomain": "stats"}
        },
    )


class _StubAdminClient:
    def __init__(self) -> None:
        self.uploads: list[list[str]] = []
        self.channel_reads = 0

    async def health_check(self) -> bool:
        return True

    async def get_channels(self) -> list[Channel]:
        self.channel_reads += 1
        return [_channel()]

    async def upload_discovery_datasets(
        self, channel_id: int, file_paths: Sequence[str], mode: RecordUploadMode
    ) -> DiscoveryUploadSummary:
        self.uploads.append(list(file_paths))
        return DiscoveryUploadSummary(rows_read=len(file_paths))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _StubAdminClient:
    stub = _StubAdminClient()

    @asynccontextmanager
    async def get_admin_client(*_: Any, **__: Any):
        yield stub

    monkeypatch.setattr(discovery_module, "get_admin_client", get_admin_client)
    return stub


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("a.xlsx", "b.csv", "~$a.xlsx", ".hidden.csv", "notes.txt"):
        (folder / name).write_bytes(b"x")
    (folder / "nested.xlsx").mkdir()
    return folder


async def test_a_directory_sends_its_upload_files_in_one_request(
    client: _StubAdminClient, data_dir: Path, tmp_path: Path
) -> None:
    extra = tmp_path / "extra.csv"
    extra.write_bytes(b"x")

    await upload_handler(channel=_DEPLOYMENT_ID, file=[str(data_dir), str(extra)])

    assert client.uploads == [[str(data_dir / "a.xlsx"), str(data_dir / "b.csv"), str(extra)]]


async def test_a_quoted_glob_is_expanded_by_the_cli(
    client: _StubAdminClient, data_dir: Path
) -> None:
    await upload_handler(channel=_DEPLOYMENT_ID, file=[str(data_dir / "*")])

    assert client.uploads == [[str(data_dir / "a.xlsx"), str(data_dir / "b.csv")]]


async def test_an_existing_file_with_glob_characters_is_taken_literally(
    client: _StubAdminClient, tmp_path: Path
) -> None:
    path = tmp_path / "report [2024].xlsx"
    path.write_bytes(b"x")

    await upload_handler(channel=_DEPLOYMENT_ID, file=[str(path)])

    assert client.uploads == [[str(path)]]


async def test_a_repeated_flag_is_one_deduplicated_request(
    client: _StubAdminClient, data_dir: Path
) -> None:
    args = upload_command.parse_args(
        f"-c {_DEPLOYMENT_ID} --file {data_dir / 'b.csv'} {data_dir / 'a.xlsx'}"
        f" --file {data_dir / 'a.xlsx'}"
    )

    await upload_handler(**args)

    assert client.uploads == [[str(data_dir / "a.xlsx"), str(data_dir / "b.csv")]]


@pytest.mark.parametrize("value", ["*.xlsx", "missing.csv", "empty"])
async def test_a_value_yielding_no_file_sends_nothing(
    client: _StubAdminClient, tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()

    await upload_handler(channel=_DEPLOYMENT_ID, file=[str(tmp_path / value)])

    assert client.uploads == []
    assert client.channel_reads == 0
    assert value in capsys.readouterr().out


async def test_replace_confirmation_speaks_of_the_files(
    client: _StubAdminClient, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def confirm(prompt: str, **_: Any) -> bool:
        prompts.append(prompt)
        return False

    monkeypatch.setattr(discovery_module, "confirm_interactive", confirm)

    await upload_handler(channel=_DEPLOYMENT_ID, file=[str(data_dir)], mode="replace")

    assert "absent from the files" in prompts[0]
    assert client.uploads == []


def test_problems_from_files_get_a_file_column(capsys: pytest.CaptureFixture[str]) -> None:
    detail = DiscoveryPayloadErrorDetail(
        message="The payload has 2 problems; nothing was saved.",
        problems=[
            DiscoveryPayloadProblem(message="Empty agency.", file="a.xlsx", row=4, cell="C4"),
            DiscoveryPayloadProblem(message="Bad value.", file="b.csv", row=9),
        ],
    )

    render_payload_problems(detail)

    out = capsys.readouterr().out
    assert "File" in out
    assert "a.xlsx" in out and "b.csv" in out


def test_problems_of_a_json_payload_have_no_file_column(
    capsys: pytest.CaptureFixture[str],
) -> None:
    detail = DiscoveryPayloadErrorDetail(
        message="The payload has 1 problem; nothing was saved.",
        problems=[DiscoveryPayloadProblem(message="Duplicate.", index=0)],
    )

    render_payload_problems(detail)

    assert "File" not in capsys.readouterr().out


def test_the_per_file_table_appears_only_for_several_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    one = DiscoveryUploadSummary(
        rows_read=3, files=[DiscoveryUploadFileSummary(file="a.xlsx", rows_read=3)]
    )
    render_upload_summary(one)
    assert "a.xlsx" not in capsys.readouterr().out

    two = DiscoveryUploadSummary(
        rows_read=8,
        files=[
            DiscoveryUploadFileSummary(file="a.xlsx", rows_read=3),
            DiscoveryUploadFileSummary(file="b.csv", rows_read=5),
        ],
    )
    render_upload_summary(two)
    out = capsys.readouterr().out
    assert "a.xlsx" in out and "b.csv" in out


def test_long_file_names_wrap_instead_of_being_cut(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two template copies differ only at the end of their names, so cutting them hides which."""
    monkeypatch.setenv("COLUMNS", "120")
    name = "Grade_C_Dataset_Discovery_Index_Template_Separate_Indicators_Sheets_2.xlsx"
    detail = DiscoveryPayloadErrorDetail(
        message="The payload has 1 problem; nothing was saved.",
        problems=[DiscoveryPayloadProblem(message=f"{name} row 2 clashes.", file=name, row=2)],
    )

    render_payload_problems(detail)

    out = capsys.readouterr().out
    assert "…" not in out
    assert "s_2.xlsx" in out
