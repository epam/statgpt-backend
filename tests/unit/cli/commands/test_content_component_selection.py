"""Tests for how `content init` decides which components a run covers."""

import pytest

from statgpt.cli.commands import content
from statgpt.cli.commands.content import VALID_COMPONENTS, _resolve_components


def _answer(monkeypatch, selection: set[str]) -> None:
    """Stand in for the checkbox prompt, which needs a terminal."""

    async def _select(components, all_components):
        return set(selection)

    monkeypatch.setattr(content, "select_components_interactive", _select)


@pytest.mark.asyncio
async def test_selecting_nothing_cancels_the_run(monkeypatch) -> None:
    """Nothing starts checked, so an empty selection is the way to back out."""
    _answer(monkeypatch, set())

    assert await _resolve_components(only=None, yes=False) is None


@pytest.mark.asyncio
async def test_a_selection_narrows_the_run(monkeypatch) -> None:
    _answer(monkeypatch, {"channels", "discovery"})

    assert await _resolve_components(only=None, yes=False) == {"channels", "discovery"}


@pytest.mark.asyncio
async def test_selecting_datasets_pulls_in_data_sources(monkeypatch) -> None:
    """A dataset cannot be registered without its data source, as `--only` already assumes."""
    _answer(monkeypatch, {"datasets"})

    assert await _resolve_components(only=None, yes=False) == {"datasets", "datasources"}


@pytest.mark.asyncio
async def test_yes_and_only_skip_the_prompt() -> None:
    assert await _resolve_components(only=None, yes=True) == VALID_COMPONENTS
    assert await _resolve_components(only="discovery", yes=False) == {"discovery"}


@pytest.mark.asyncio
async def test_an_unknown_component_is_refused() -> None:
    with pytest.raises(ValueError, match="Invalid components"):
        await _resolve_components(only="nope", yes=False)
