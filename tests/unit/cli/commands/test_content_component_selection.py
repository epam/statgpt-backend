"""Tests for how `content init` decides which components a run covers."""

from unittest.mock import AsyncMock, Mock

import pytest

from statgpt.cli.commands import content
from statgpt.cli.commands.content import VALID_COMPONENTS, _resolve_components
from statgpt.cli.shared.batch_report import BatchReport


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "only", ["channels", "datasets", "channels,glossaries,files,discovery", "files,discovery"]
)
async def test_refresh_respects_component_selection_and_does_not_change_other_components(
    tmp_path, monkeypatch, only: str
) -> None:
    components = await _resolve_components(only=only, yes=False)
    for directory in ["dial_files", "discovery_datasets"]:
        (tmp_path / directory).mkdir()
    monkeypatch.setattr(content.utils, "read_yaml", lambda path: {})
    monkeypatch.setattr(content.utils, "optional_read_yaml", lambda path: None)
    channels = AsyncMock(return_value={})
    sources = AsyncMock(return_value={})
    datasets = AsyncMock(return_value=[])
    files = AsyncMock()
    discovery = AsyncMock()
    monkeypatch.setattr(content, "_process_channels", channels)
    monkeypatch.setattr(content, "_process_data_sources", sources)
    monkeypatch.setattr(content, "_process_datasets", datasets)
    monkeypatch.setattr(content, "_upload_dial_files", files)
    monkeypatch.setattr(content, "_upload_discovery_datasets", discovery)
    selected_datasets = {"AG:DF_A(1.0)"}

    await content._process_client(
        AsyncMock(),
        BatchReport(title="Summary"),
        "sample",
        str(tmp_path),
        {},
        selected_datasets,
        components,
        update_unchanged=True,
    )

    for name, processor in [
        ("channels", channels),
        ("datasources", sources),
        ("datasets", datasets),
    ]:
        if name in components:
            processor.assert_awaited_once()
            assert processor.call_args.kwargs["update_unchanged"] is True
        else:
            processor.assert_not_awaited()
    if "datasets" in components:
        assert "datasources" in components
        assert datasets.call_args.args[7] is selected_datasets
        assert datasets.call_args.kwargs["link_channels"] is ("channels" in components)
    if "channels" in components:
        assert channels.call_args.args[5] is ("glossaries" in components)
    for name, processor in [("files", files), ("discovery", discovery)]:
        if name in components:
            processor.assert_awaited_once()
            assert "update_unchanged" not in processor.call_args.kwargs
        else:
            processor.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_does_not_skip_interactive_confirmation(tmp_path, monkeypatch) -> None:
    (tmp_path / "clients" / "sample").mkdir(parents=True)
    monkeypatch.setattr(content.cli_settings, "config_dir", str(tmp_path))
    _answer(monkeypatch, {"channels"})
    confirmation = Mock(return_value=False)
    admin_client = Mock()
    monkeypatch.setattr(content, "_confirm_full_init", confirmation)
    monkeypatch.setattr(content, "get_admin_client", admin_client)

    await content.init_handler(client_id="sample", update_unchanged=True)

    confirmation.assert_called_once_with(["sample"], {"channels"})
    admin_client.assert_not_called()
