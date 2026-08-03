"""Change-detection tests for the CLI content sync."""

import datetime
from typing import Any

from statgpt.cli.commands.content import _data_source_changed
from statgpt.common.data.sdmx.common.config import SdmxConfig
from statgpt.common.schemas import DataSource, DataSourceType

_SDMX_CONFIG = SdmxConfig(id="proxy", url="https://example.invalid", name="proxy").model_dump(
    mode='json', by_alias=True
)
_PROXY_CONFIG = {"configs": [], "agencies": [{"name": "IMF"}], "structureFanOutEnabled": False}


def _existing(*, type_name: str = "PROXY_SDMX30", details: dict[str, Any]) -> DataSource:
    now = datetime.datetime(2026, 1, 1)
    return DataSource(
        id=1,
        created_at=now,
        updated_at=now,
        title="proxy",
        description="",
        type_id=3,
        details=details,
        type=DataSourceType(id=3, created_at=now, updated_at=now, name=type_name, description=""),
    )


def _incoming(**details: Any) -> dict[str, Any]:
    return {
        "title": "proxy",
        "description": "",
        "type_id": 3,
        "details": {"sdmxConfig": _SDMX_CONFIG, **details},
    }


def test_proxy_source_unchanged_when_the_config_is_left_to_the_config_server() -> None:
    """A config file that says nothing about `proxyConfig` must not trigger an endless update."""
    existing = _existing(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _PROXY_CONFIG})

    assert not _data_source_changed(_incoming(), existing)


def test_proxy_source_unchanged_when_the_config_matches_the_config_server() -> None:
    existing = _existing(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _PROXY_CONFIG})

    assert not _data_source_changed(_incoming(proxyConfig=_PROXY_CONFIG), existing)


def test_proxy_source_changed_when_the_config_differs() -> None:
    existing = _existing(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _PROXY_CONFIG})

    assert _data_source_changed(_incoming(proxyConfig={"agencies": []}), existing)


def test_proxy_source_changed_when_a_persisted_field_differs() -> None:
    existing = _existing(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _PROXY_CONFIG})

    assert _data_source_changed(_incoming(locale="fr"), existing)


def test_plain_sdmx_source_unchanged() -> None:
    existing = _existing(type_name="SDMX21", details={"sdmxConfig": _SDMX_CONFIG})

    assert not _data_source_changed(_incoming(), existing)


def test_source_changed_when_the_title_differs() -> None:
    existing = _existing(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _PROXY_CONFIG})

    assert _data_source_changed({**_incoming(), "title": "renamed"}, existing)
