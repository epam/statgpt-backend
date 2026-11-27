"""Change-detection tests for the CLI content sync."""

import datetime
import uuid
from typing import Any

import pytest

from statgpt.cli.commands.content import _data_source_changed, _dataset_changed
from statgpt.common.data.sdmx.common.config import SdmxConfig, SdmxDataSetConfig
from statgpt.common.schemas import DataSet, DataSource, DataSourceType
from statgpt.common.schemas.dataset import Status

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


def _dataset(details: dict[str, Any]) -> DataSet:
    now = datetime.datetime(2026, 1, 1)
    return DataSet(
        id=7,
        id_=uuid.UUID("12345678-1234-1234-1234-123456789abc"),
        created_at=now,
        updated_at=now,
        title="dataset",
        data_source_id=1,
        data_source=None,
        details=details,
        status=Status(status="online"),
    )


def _dataset_input(existing: DataSet, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "id_": str(existing.id_),
        "title": existing.title,
        "data_source_id": existing.data_source_id,
        "details": details,
    }


def _dataset_details() -> dict[str, Any]:
    return {
        "urn": {"agencyId": "AG", "resourceId": "DF"},
        "dimensions": {
            "INDICATOR": {"dimensionType": "INDICATOR"},
            "TIME_PERIOD": {"dimensionType": "TIME_PERIOD"},
            "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
        },
    }


@pytest.mark.parametrize("full_stored_config", [False, True])
def test_dataset_omitted_defaults_are_effective_equivalent(full_stored_config: bool) -> None:
    details = _dataset_details()
    stored_details = (
        SdmxDataSetConfig.model_validate(details).model_dump(mode="json", by_alias=True)
        if full_stored_config
        else details
    )
    existing = _dataset(stored_details)
    source = _existing(type_name="SDMX21", details={"sdmxConfig": _SDMX_CONFIG})

    for _ in range(2):
        assert not _dataset_changed(_dataset_input(existing, details), existing, source)


def test_dataset_explicit_default_matches_sparse_config() -> None:
    details = _dataset_details()
    existing = _dataset(details)
    source = _existing(type_name="SDMX21", details={"sdmxConfig": _SDMX_CONFIG})
    incoming = _dataset_input(existing, {**details, "isOfficial": False})

    assert not _dataset_changed(incoming, existing, source)


def test_dataset_aliases_and_source_managed_title_are_normalized_on_both_sides() -> None:
    existing = _dataset(
        {
            "urn": {"agency_id": "AG", "resource_id": "DF"},
            "dimensions": {
                "INDICATOR": {"dimension_type": "INDICATOR"},
                "TIME_PERIOD": {"dimension_type": "TIME_PERIOD"},
                "FREQ": {"dimension_type": "NON_INDICATOR", "subtype": "FREQUENCY"},
            },
            "use_title_from_src": True,
        }
    )
    source = _existing(type_name="SDMX21", details={"sdmxConfig": _SDMX_CONFIG})
    incoming = _dataset_input(
        existing,
        {
            "urn": {"agencyId": "AG", "resourceId": "DF"},
            "dimensions": {
                "INDICATOR": {"dimensionType": "INDICATOR"},
                "TIME_PERIOD": {"dimensionType": "TIME_PERIOD"},
                "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
            },
            "useTitleFromSrc": True,
        },
    )
    incoming["title"] = "title from config file"

    assert not _dataset_changed(incoming, existing, source)


@pytest.mark.parametrize("changed_field", ["title", "details"])
def test_dataset_real_changes_are_not_hidden(changed_field: str) -> None:
    details = _dataset_details()
    existing = _dataset(details)
    source = _existing(type_name="SDMX21", details={"sdmxConfig": _SDMX_CONFIG})
    incoming = _dataset_input(existing, details)
    if changed_field == "title":
        incoming["title"] = "renamed"
    else:
        incoming["details"] = {**details, "isOfficial": True}

    assert _dataset_changed(incoming, existing, source)
