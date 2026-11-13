"""Tests that a partially-failed batch is reported through the process exit code.

A summary on stdout is not enough: a pipeline that reads exit 0 as "everything was
onboarded" would keep mistaking a truncated run for a complete one.
"""

import pytest

from statgpt.cli import _execute_direct
from statgpt.cli.commands.base import Command, CommandArg, CommandGroup, CommandRegistry
from statgpt.cli.shared.batch_report import BatchPartialFailureError


def _registry(handler) -> CommandRegistry:
    registry = CommandRegistry(version="test")
    registry.register_command(Command(name="run", description="test command", handler=handler))
    return registry


@pytest.mark.asyncio
async def test_a_partial_failure_exits_non_zero_without_a_duplicate_error(capsys) -> None:
    async def handler() -> None:
        print("summary already rendered")
        raise BatchPartialFailureError("some items failed")

    exit_code = await _execute_direct(_registry(handler), ["run"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "summary already rendered" in out
    assert "Command failed" not in out, "the summary already explained it"
    assert "some items failed" not in out


@pytest.mark.asyncio
async def test_a_clean_run_exits_zero() -> None:
    async def handler() -> None:
        return None

    assert await _execute_direct(_registry(handler), ["run"]) == 0


@pytest.mark.asyncio
async def test_an_unexpected_error_still_reports_itself(capsys) -> None:
    async def handler() -> None:
        raise RuntimeError("something unrelated broke")

    exit_code = await _execute_direct(_registry(handler), ["run"])

    assert exit_code == 1
    assert "something unrelated broke" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_path_with_a_space_reaches_the_handler_whole() -> None:
    """The system shell already split the arguments; re-joining must keep its quoting."""
    received: dict[str, object] = {}

    async def handler(file: list[str], channel: str | None = None) -> None:
        received.update(file=file, channel=channel)

    registry = CommandRegistry(version="test")
    group = CommandGroup(name="discovery", description="test group")
    group.add_command(
        Command(
            name="upload",
            description="test command",
            handler=handler,
            args=[
                CommandArg(name="channel", description="c", short_name="c"),
                CommandArg(name="file", description="f", multiple=True),
            ],
        )
    )
    registry.register_group(group)

    exit_code = await _execute_direct(
        registry, ["discovery", "upload", "-c", "ch", "--file", "my data/a b.xlsx", "c.csv"]
    )

    assert exit_code == 0
    assert received == {"file": ["my data/a b.xlsx", "c.csv"], "channel": "ch"}
