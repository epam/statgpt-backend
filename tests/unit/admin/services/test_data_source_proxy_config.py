"""Tests for the SDMX proxy configuration exposed through a data source's `details` field."""

import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import statgpt.common.schemas as schemas
from statgpt.admin.services.data_source import AdminPortalDataSourceService
from statgpt.common.data.sdmx.common.config import SdmxConfig
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.config_client import (
    ProxyConfigServerError,
    ProxyConfigValidationError,
)

_HOST = "http://config-server:8060"
_URL = f"{_HOST}/statgpt/sdmx-proxy-config-server/api/v0/config"
_STORED_CONFIG = {"configs": [], "agencies": [{"name": "IMF"}], "structureFanOutEnabled": False}

_SDMX_CONFIG_MODEL = SdmxConfig(id="proxy", url="https://example.invalid", name="proxy")
_SDMX_CONFIG = _SDMX_CONFIG_MODEL.model_dump(mode='json', by_alias=True)


@pytest.fixture(autouse=True)
def config_server_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDMX_PROXY_CONFIG_SERVER_HOST", _HOST)


@pytest.fixture
def service() -> AdminPortalDataSourceService:
    return AdminPortalDataSourceService(MagicMock())


def _data_source(
    *,
    type_name: str = "PROXY_SDMX30",
    details: dict[str, Any] | None = None,
    title: str = "proxy",
) -> schemas.DataSource:
    now = datetime.datetime(2026, 1, 1)
    return schemas.DataSource(
        id=1,
        created_at=now,
        updated_at=now,
        title=title,
        description="",
        type_id=3,
        details=details if details is not None else {"sdmxConfig": _SDMX_CONFIG},
        type=schemas.DataSourceType(
            id=3, created_at=now, updated_at=now, name=type_name, description=""
        ),
    )


def _proxy_config(**overrides) -> StatGptSdmxProxyDataSourceConfig:
    return StatGptSdmxProxyDataSourceConfig(sdmx_config=_SDMX_CONFIG_MODEL, **overrides)


class TestReadEnrichment:
    async def test_fetches_the_live_configuration(self, service, mocker) -> None:
        fetch = mocker.patch(
            "statgpt.admin.services.data_source.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        item = _data_source()

        await service._enrich_with_proxy_config([item])

        assert item.details["proxyConfig"] == _STORED_CONFIG
        fetch.assert_awaited_once_with(_URL)

    async def test_reports_a_cold_start_as_null(self, service, mocker) -> None:
        mocker.patch(
            "statgpt.admin.services.data_source.fetch_proxy_config",
            AsyncMock(return_value=None),
        )
        item = _data_source()

        await service._enrich_with_proxy_config([item])

        assert item.details["proxyConfig"] is None

    async def test_leaves_the_key_absent_when_the_config_server_fails(
        self, service, mocker, caplog
    ) -> None:
        mocker.patch(
            "statgpt.admin.services.data_source.fetch_proxy_config",
            AsyncMock(side_effect=ProxyConfigServerError("connection refused")),
        )
        item = _data_source()

        with caplog.at_level("WARNING"):
            await service._enrich_with_proxy_config([item])

        assert "proxyConfig" not in item.details
        assert "connection refused" in caplog.text

    async def test_skips_non_proxy_data_sources(self, service, mocker) -> None:
        fetch = mocker.patch("statgpt.admin.services.data_source.fetch_proxy_config", AsyncMock())
        item = _data_source(type_name="SDMX21")

        await service._enrich_with_proxy_config([item])

        assert "proxyConfig" not in item.details
        fetch.assert_not_awaited()

    async def test_fetches_a_shared_config_server_once(self, service, mocker) -> None:
        fetch = mocker.patch(
            "statgpt.admin.services.data_source.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        items = [_data_source(title="a"), _data_source(title="b")]

        await service._enrich_with_proxy_config(items)

        assert [item.details["proxyConfig"] for item in items] == [_STORED_CONFIG] * 2
        fetch.assert_awaited_once_with(_URL)

    async def test_skips_a_data_source_with_invalid_details(self, service, mocker, caplog) -> None:
        fetch = mocker.patch("statgpt.admin.services.data_source.fetch_proxy_config", AsyncMock())
        item = _data_source(details={"sdmxConfig": {"id": "proxy"}})  # missing url and name

        with caplog.at_level("WARNING"):
            await service._enrich_with_proxy_config([item])

        assert "proxyConfig" not in item.details
        assert "Cannot resolve the proxy config server URL" in caplog.text
        fetch.assert_not_awaited()


class TestWritePush:
    async def test_pushes_a_submitted_configuration(self, service, mocker) -> None:
        push = mocker.patch(
            "statgpt.admin.services.data_source.push_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )

        await service._push_proxy_config(_proxy_config(proxy_config=_STORED_CONFIG))

        push.assert_awaited_once_with(_URL, _STORED_CONFIG)

    async def test_leaves_the_config_server_alone_when_nothing_is_submitted(
        self, service, mocker
    ) -> None:
        push = mocker.patch("statgpt.admin.services.data_source.push_proxy_config", AsyncMock())

        await service._push_proxy_config(_proxy_config())

        push.assert_not_awaited()

    async def test_reports_a_rejected_configuration_to_the_caller(self, service, mocker) -> None:
        mocker.patch(
            "statgpt.admin.services.data_source.push_proxy_config",
            AsyncMock(side_effect=ProxyConfigValidationError("agency IMF has no registry")),
        )

        with pytest.raises(HTTPException) as exc:
            await service._push_proxy_config(_proxy_config(proxy_config=_STORED_CONFIG))

        assert exc.value.status_code == 422
        assert "agency IMF has no registry" in exc.value.detail

    async def test_swallows_an_unreachable_config_server(self, service, mocker, caplog) -> None:
        mocker.patch(
            "statgpt.admin.services.data_source.push_proxy_config",
            AsyncMock(side_effect=ProxyConfigServerError("connection refused")),
        )

        with caplog.at_level("WARNING"):
            await service._push_proxy_config(_proxy_config(proxy_config=_STORED_CONFIG))

        assert "connection refused" in caplog.text


class TestImportChangeDetection:
    def test_a_payload_without_a_proxy_config_is_unchanged(self, service) -> None:
        stored = _data_source(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _STORED_CONFIG})

        assert not service._details_changed(stored, {"sdmxConfig": _SDMX_CONFIG})

    def test_a_matching_proxy_config_is_unchanged(self, service) -> None:
        stored = _data_source(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _STORED_CONFIG})
        incoming = {"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _STORED_CONFIG}

        assert not service._details_changed(stored, incoming)

    def test_a_different_proxy_config_is_changed(self, service) -> None:
        stored = _data_source(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _STORED_CONFIG})
        incoming = {"sdmxConfig": _SDMX_CONFIG, "proxyConfig": {"agencies": []}}

        assert service._details_changed(stored, incoming)

    def test_a_different_persisted_field_is_changed(self, service) -> None:
        stored = _data_source(details={"sdmxConfig": _SDMX_CONFIG, "proxyConfig": _STORED_CONFIG})
        incoming = {"sdmxConfig": _SDMX_CONFIG, "locale": "fr"}

        assert service._details_changed(stored, incoming)
