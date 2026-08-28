"""The shipped sample client configuration must survive the merge `content init` performs.

`_add_tools_to_channel` keys each `tools.yaml` entry by its `type`, which is the *channel config
field name* - and `ChannelConfig` ignores unknown keys. So a tool written at the wrong nesting
level, or under the `ToolTypes` value instead of the field name, is dropped in silence: the
config loads, the tool is simply absent. These tests are what turns that into a failure.
"""

from pathlib import Path
from typing import Any

import pytest

from statgpt.cli.commands.content import _add_tools_to_channel
from statgpt.common.schemas import ChannelConfig
from statgpt.common.utils import read_yaml

_SAMPLE = Path(__file__).resolve().parents[4] / "configurations" / "clients" / "sample"


def _merged() -> list[tuple[dict[str, Any], ChannelConfig]]:
    """Every sample channel, as `content init` would build and validate it."""
    tools_cfg = read_yaml(_SAMPLE / "tools.yaml")
    channel_cfg = read_yaml(_SAMPLE / "channels.yaml")

    merged = []
    for ch_cfg in channel_cfg["channels"]:
        _add_tools_to_channel(ch_cfg, tools_cfg)
        merged.append((ch_cfg, ChannelConfig.model_validate(ch_cfg["details"])))
    return merged


def test_the_sample_channels_validate() -> None:
    assert _merged(), "the sample config declares no channels"


def test_every_declared_tool_reaches_the_channel_config() -> None:
    """A `type` that is not a `ChannelConfig` field would be dropped without a word."""
    tools_cfg = read_yaml(_SAMPLE / "tools.yaml")
    declared = {tool["type"] for tool in tools_cfg["tools"]}
    unknown = declared - set(ChannelConfig.model_fields)

    assert not unknown, f"tools.yaml entries keyed by a non-field `type`: {sorted(unknown)}"


def test_no_details_key_is_dropped_by_the_channel_config() -> None:
    """Catches a block nested one level too deep, or written under the wrong name."""
    for ch_cfg, config in _merged():
        dropped = set(ch_cfg["details"]) - set(config.model_dump(by_alias=False))

        assert not dropped, f"{ch_cfg['deployment_id']}: dropped config keys {sorted(dropped)}"


def test_the_sample_channel_configures_discovery_datasets() -> None:
    """The block carries both halves: the publish target and the chat-time lookup."""
    for ch_cfg, config in _merged():
        assert config.discovery_application_id, f"{ch_cfg['deployment_id']}: no publish target"
        assert config.is_discovery_lookup_available is True
        templates = config.discovery_datasets.details.templates  # type: ignore[union-attr]
        assert "{items}" in templates.wrapper, "the wrapper must place the rendered items"


def test_discovery_datasets_is_never_offered_to_the_agent() -> None:
    for _, config in _merged():
        names = {tool.name for tool in config.agent_tools}

        assert config.discovery_datasets is not None
        assert config.discovery_datasets.name not in names


@pytest.mark.parametrize("filename", ["channels.yaml", "tools.yaml", "data_sources.yaml"])
def test_the_sample_config_files_are_present(filename: str) -> None:
    """The paths these tests read are the ones `content init` reads."""
    assert (_SAMPLE / filename).is_file()
