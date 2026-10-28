"""Tests for the `--force` option of `discovery reindex`.

It names the stages to redo. The API still reads a boolean, but the CLI no longer sends one.
"""

import datetime
from contextlib import asynccontextmanager
from typing import Any

import pytest

from statgpt.cli.commands import discovery as discovery_module
from statgpt.cli.commands.discovery import reindex_handler
from statgpt.common.schemas import (
    Channel,
    DiscoveryIndexingJob,
    DiscoveryIndexingStage,
    PreprocessingStatusEnum,
)

_NOW = datetime.datetime(2026, 1, 1)
_CHANNEL_ID = 7
_DEPLOYMENT_ID = "my-channel"


def _channel() -> Channel:
    return Channel(
        id=_CHANNEL_ID,
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


def _job() -> DiscoveryIndexingJob:
    return DiscoveryIndexingJob(
        id=17,
        channel_id=_CHANNEL_ID,
        status=PreprocessingStatusEnum.COMPLETED,
        created_at=_NOW,
        updated_at=_NOW,
        records_summarized=2,
        records_failed=1,
    )


class _StubAdminClient:
    """Answers the channel lookup and the job calls, and records the trigger."""

    def __init__(self) -> None:
        self.triggered: list[frozenset[DiscoveryIndexingStage]] = []

    async def health_check(self) -> bool:
        return True

    async def get_channels(self) -> list[Channel]:
        return [_channel()]

    async def trigger_discovery_indexing(
        self, channel_id: int, force: frozenset[DiscoveryIndexingStage]
    ) -> DiscoveryIndexingJob:
        self.triggered.append(force)
        return _job()

    async def get_discovery_indexing_job(self, job_id: int) -> DiscoveryIndexingJob:
        return _job()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _StubAdminClient:
    stub = _StubAdminClient()

    @asynccontextmanager
    async def get_admin_client(*_: Any, **__: Any):
        yield stub

    monkeypatch.setattr(discovery_module, "get_admin_client", get_admin_client)
    return stub


@pytest.mark.parametrize(
    ("force", "expected"),
    [
        (None, frozenset()),
        ("summarization", {DiscoveryIndexingStage.SUMMARIZATION}),
        ("Publishing", {DiscoveryIndexingStage.PUBLISHING}),
        (" summarization , publishing ", set(DiscoveryIndexingStage)),
    ],
)
async def test_the_named_stages_are_forced(
    client: _StubAdminClient, force: str | None, expected: set[DiscoveryIndexingStage]
) -> None:
    await reindex_handler(channel=_DEPLOYMENT_ID, force=force)

    assert client.triggered == [frozenset(expected)]


@pytest.mark.parametrize("force", ["true", "all", "summarization,validation"])
async def test_anything_but_a_stage_is_refused_before_a_job_is_started(
    client: _StubAdminClient, force: str
) -> None:
    with pytest.raises(ValueError, match="Invalid stages"):
        await reindex_handler(channel=_DEPLOYMENT_ID, force=force)

    assert client.triggered == []
