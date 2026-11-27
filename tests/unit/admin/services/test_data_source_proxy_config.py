"""Tests for the SDMX proxy configuration exposed through a data source's `details` field."""

import datetime
import io
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import HTTPException

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.services.data_source import AdminPortalDataSourceService
from statgpt.admin.settings.exim import JobsConfig
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
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        item = _data_source()

        await service._enrich_with_external_details([item])

        assert item.details["proxyConfig"] == _STORED_CONFIG
        fetch.assert_awaited_once_with(_URL)

    async def test_reports_a_cold_start_as_null(self, service, mocker) -> None:
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=None),
        )
        item = _data_source()

        await service._enrich_with_external_details([item])

        assert item.details["proxyConfig"] is None

    async def test_leaves_the_key_absent_when_the_config_server_fails(
        self, service, mocker, caplog
    ) -> None:
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(side_effect=ProxyConfigServerError("connection refused")),
        )
        item = _data_source()

        with caplog.at_level("WARNING"):
            await service._enrich_with_external_details([item])

        assert "proxyConfig" not in item.details
        assert "connection refused" in caplog.text

    async def test_skips_data_sources_without_externally_owned_details(
        self, service, mocker
    ) -> None:
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config", AsyncMock()
        )
        item = _data_source(type_name="SDMX21")

        await service._enrich_with_external_details([item])

        assert "proxyConfig" not in item.details
        fetch.assert_not_awaited()

    async def test_fetches_a_shared_config_server_once(self, service, mocker) -> None:
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        items = [_data_source(title="a"), _data_source(title="b")]

        await service._enrich_with_external_details(items)

        assert [item.details["proxyConfig"] for item in items] == [_STORED_CONFIG] * 2
        fetch.assert_awaited_once_with(_URL)

    async def test_skips_a_data_source_with_invalid_details(self, service, mocker, caplog) -> None:
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config", AsyncMock()
        )
        item = _data_source(details={"sdmxConfig": {"id": "proxy"}})  # missing url and name

        with caplog.at_level("WARNING"):
            await service._enrich_with_external_details([item])

        assert "proxyConfig" not in item.details
        assert "Cannot resolve the external details owner" in caplog.text
        fetch.assert_not_awaited()


class TestExport:
    async def test_writes_the_live_configuration_without_touching_the_caller_s_models(
        self, service, mocker, tmp_path
    ) -> None:
        """The caller keeps using its data sources after the export, so they must come back intact."""
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        item = _data_source()

        await service.export_data_sources([item], str(tmp_path))

        assert "proxyConfig" not in item.details
        exported = yaml.safe_load((tmp_path / JobsConfig.DATA_SOURCES_FILE).read_text())
        assert exported['dataSources'][0]['details']['proxyConfig'] == _STORED_CONFIG


class TestWritePush:
    async def test_pushes_a_submitted_configuration(self, service, mocker) -> None:
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )

        pushed = await service._push_external_details(_proxy_config(proxy_config=_STORED_CONFIG))

        push.assert_awaited_once_with(_URL, _STORED_CONFIG)
        assert pushed == {"proxyConfig": _STORED_CONFIG}

    async def test_leaves_the_config_server_alone_when_nothing_is_submitted(
        self, service, mocker
    ) -> None:
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )

        assert await service._push_external_details(_proxy_config()) is None

        push.assert_not_awaited()

    async def test_reports_a_rejected_configuration_to_the_caller(self, service, mocker) -> None:
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config",
            AsyncMock(side_effect=ProxyConfigValidationError("agency IMF has no registry")),
        )

        with pytest.raises(HTTPException) as exc:
            await service._push_external_details(_proxy_config(proxy_config=_STORED_CONFIG))

        assert exc.value.status_code == 422
        assert "agency IMF has no registry" in exc.value.detail

    async def test_fails_the_write_when_the_config_server_cannot_be_reached(
        self, service, mocker, caplog
    ) -> None:
        """A submitted configuration lives nowhere else, so a dropped push must not read as 200."""
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config",
            AsyncMock(side_effect=ProxyConfigServerError("connection refused")),
        )

        with caplog.at_level("WARNING"), pytest.raises(HTTPException) as exc:
            await service._push_external_details(_proxy_config(proxy_config=_STORED_CONFIG))

        assert exc.value.status_code == 502
        assert "connection refused" in exc.value.detail
        assert "connection refused" in caplog.text

    async def test_an_unconfigured_config_server_url_fails_the_write(self, service, mocker) -> None:
        """`$env:{...}` stays verbatim in an unconfigured deployment; the client rejects it."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(HTTPException) as exc:
            await service._push_external_details(_proxy_config(proxy_config=_STORED_CONFIG))

        assert exc.value.status_code == 502
        assert "unresolved environment variable" in exc.value.detail


class TestWriteResult:
    """What a create/update reports - and audits - as the externally-owned details."""

    async def test_reports_what_the_config_server_stored_without_reading_it_back(
        self, service, mocker
    ) -> None:
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config", AsyncMock()
        )
        item = _data_source()

        await service._apply_external_details(item, {"proxyConfig": _STORED_CONFIG})

        assert item.details["proxyConfig"] == _STORED_CONFIG
        fetch.assert_not_awaited()

    async def test_reads_the_config_server_when_the_write_left_it_untouched(
        self, service, mocker
    ) -> None:
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )
        item = _data_source()

        await service._apply_external_details(item, None)

        assert item.details["proxyConfig"] == _STORED_CONFIG
        fetch.assert_awaited_once_with(_URL)


class TestSparsePersistence:
    async def test_create_keeps_sparse_storage_and_enriches_only_the_response(
        self, service, mocker
    ) -> None:
        details = {"sdmxConfig": {"id": "proxy", "url": "https://example.invalid", "name": "proxy"}}
        now = datetime.datetime(2026, 1, 1)
        source_type = models.DataSourceType(
            id=3, created_at=now, updated_at=now, name="PROXY_SDMX30", description=""
        )
        service._parse_details_field = AsyncMock(return_value=_proxy_config())
        service._session.flush = AsyncMock()

        async def refresh(item, **kwargs):
            item.id = 1
            item.created_at = now
            item.updated_at = now
            item.type = source_type

        service._session.refresh = AsyncMock(side_effect=refresh)
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )

        result = await AdminPortalDataSourceService.create_data_source.__wrapped__(
            service, schemas.DataSourceBase(title="proxy", type_id=3, details=details)
        )

        stored = service._session.add.call_args.args[0]
        assert stored.details == details
        assert stored.description == ""
        assert result.details == {**details, "proxyConfig": _STORED_CONFIG}
        push.assert_not_awaited()

    async def test_replacement_preserves_sparse_config_and_external_ownership(
        self, service, mocker
    ) -> None:
        details = {
            "sdmxConfig": {"id": "proxy", "url": "https://example.invalid", "name": "proxy"},
            "locale": "en",
        }
        stored = _data_source(details={**details, "description": "Legacy default"})
        service._get_item_or_raise = AsyncMock(return_value=stored)
        service._parse_details_field = AsyncMock(return_value=_proxy_config(locale="en"))
        service._session.execute = AsyncMock(return_value=MagicMock())
        service._session.execute.return_value.scalar_one.return_value = stored
        service._session.flush = AsyncMock()
        service._session.refresh = AsyncMock()
        mocker.patch(
            "statgpt.admin.services.data_source.DataSourceSerializer.db_to_schema",
            return_value=_data_source(details=details),
        )
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=_STORED_CONFIG),
        )

        result = await AdminPortalDataSourceService.update.__wrapped__(
            service, stored.id, schemas.DataSourceUpdate(details=details)
        )

        assert service._session.execute.call_args.args[0].compile().params["details"] == details
        assert result.details == {**details, "proxyConfig": _STORED_CONFIG}
        push.assert_not_awaited()


class TestImportChangeDetection:
    async def test_repeated_sparse_import_does_not_update_or_write_external_config(
        self, service, mocker
    ) -> None:
        stored = _data_source(
            details={
                **_proxy_config().model_dump(mode="json", by_alias=True),
                "proxyConfig": _STORED_CONFIG,
            }
        )
        incoming = {
            "sdmxConfig": {"id": "proxy", "url": "https://example.invalid", "name": "proxy"}
        }
        service.get_data_sources_schemas_with_external_details = AsyncMock(return_value=[stored])
        service.update = AsyncMock()
        service.create_data_source = AsyncMock()
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                JobsConfig.DATA_SOURCES_FILE,
                yaml.safe_dump(
                    {"dataSources": [{"title": "proxy", "type_id": 3, "details": incoming}]}
                ),
            )
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            await service.import_data_sources_from_zip(archive, update_data_sources=True)
            await service.import_data_sources_from_zip(archive, update_data_sources=True)
        service.update.assert_not_awaited()
        service.create_data_source.assert_not_awaited()
        push.assert_not_awaited()

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
