"""Tests for `discovery clear`.

Emptying a channel is the one discovery command that cannot be undone, so what matters is
that it does not run without being asked to, and that it does not ask when there is nothing
to delete.
"""

import datetime
from contextlib import asynccontextmanager
from typing import Any

import pytest

from statgpt.cli.commands import discovery as discovery_module
from statgpt.cli.commands.discovery import clear_handler
from statgpt.cli.settings import cli_runtime
from statgpt.cli.shared.prompts import NonInteractiveError
from statgpt.common.schemas import Channel, DiscoveryDatasetStats

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


class _StubAdminClient:
    """Answers the two reads the command makes, and records the delete."""

    def __init__(self, total: int) -> None:
        self._total = total
        self.cleared: list[int] = []

    async def health_check(self) -> bool:
        return True

    async def get_channels(self) -> list[Channel]:
        return [_channel()]

    async def get_discovery_stats(self, channel_id: int) -> DiscoveryDatasetStats:
        return DiscoveryDatasetStats(total=self._total)

    async def clear_discovery_datasets(self, channel_id: int) -> list[Any]:
        self.cleared.append(channel_id)
        return [object()] * self._total


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _StubAdminClient:
    """The command's admin client, defaulting to a channel that holds three records."""
    stub = _StubAdminClient(total=3)

    @asynccontextmanager
    async def get_admin_client(*_: Any, **__: Any):
        yield stub

    monkeypatch.setattr(discovery_module, "get_admin_client", get_admin_client)
    return stub


@pytest.fixture
def confirmations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every prompt shown, answered yes. Replaced per-test where a no is wanted."""
    prompts: list[str] = []

    def confirm(prompt: str, **_: Any) -> bool:
        prompts.append(prompt)
        return True

    monkeypatch.setattr(discovery_module, "confirm_interactive", confirm)
    return prompts


async def test_a_confirmed_clear_deletes_the_channels_records(
    client: _StubAdminClient, confirmations: list[str]
) -> None:
    await clear_handler(channel=_DEPLOYMENT_ID)

    assert client.cleared == [_CHANNEL_ID]
    assert len(confirmations) == 1


async def test_the_prompt_names_the_channel_and_what_would_go(
    client: _StubAdminClient, confirmations: list[str]
) -> None:
    """Confirming a deletion is only meaningful if it says how much is being deleted."""
    await clear_handler(channel=_DEPLOYMENT_ID)

    assert "3" in confirmations[0]
    assert _DEPLOYMENT_ID in confirmations[0]
    assert "documents" in confirmations[0]


async def test_a_declined_confirmation_deletes_nothing(
    client: _StubAdminClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discovery_module, "confirm_interactive", lambda *_, **__: False)

    await clear_handler(channel=_DEPLOYMENT_ID)

    assert client.cleared == []


async def test_yes_skips_the_prompt(client: _StubAdminClient, confirmations: list[str]) -> None:
    await clear_handler(channel=_DEPLOYMENT_ID, yes=True)

    assert client.cleared == [_CHANNEL_ID]
    assert confirmations == []


async def test_an_empty_channel_is_not_worth_a_prompt(
    client: _StubAdminClient, confirmations: list[str]
) -> None:
    """Nothing to delete, so there is nothing to confirm and nothing to call."""
    client._total = 0

    await clear_handler(channel=_DEPLOYMENT_ID)

    assert client.cleared == []
    assert confirmations == []


async def test_an_unknown_channel_deletes_nothing(client: _StubAdminClient) -> None:
    await clear_handler(channel="no-such-channel")

    assert client.cleared == []


async def test_non_interactive_without_yes_says_which_flag_to_pass(
    client: _StubAdminClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt cannot be shown, so the escape hatch has to be named."""
    monkeypatch.setattr(cli_runtime, "non_interactive", True)

    with pytest.raises(NonInteractiveError) as exc_info:
        await clear_handler(channel=_DEPLOYMENT_ID)

    assert "-y/--yes" in str(exc_info.value)
    assert client.cleared == []
