"""Config-validation tests for the StatGPT SDMX proxy data source."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from statgpt.common.data.sdmx.common.config import (
    ProviderDiscoveryMode,
    SdmxConfig,
    SdmxDataSourceConfig,
)
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.config_client import ProxyConfigServerError

_URL = "http://config-server:8060/statgpt/sdmx-proxy-config-server/api/v0/config"


def _proxy_config(**overrides) -> StatGptSdmxProxyDataSourceConfig:
    return StatGptSdmxProxyDataSourceConfig(
        sdmx_config=SdmxConfig(id="proxy", url="https://example.invalid", name="proxy"),
        **overrides,
    )


def test_proxy_config_rejects_dataflows_discovery_mode() -> None:
    with pytest.raises(ValidationError) as exc:
        _proxy_config(provider_discovery=ProviderDiscoveryMode.DATAFLOWS)
    assert "provider_discovery='dataflows'" in str(exc.value)


def test_proxy_config_accepts_agencyscheme_discovery_mode() -> None:
    config = _proxy_config(provider_discovery=ProviderDiscoveryMode.AGENCYSCHEME)
    assert config.provider_discovery is ProviderDiscoveryMode.AGENCYSCHEME


def test_proxy_config_default_discovery_is_agencyscheme() -> None:
    assert _proxy_config().provider_discovery is ProviderDiscoveryMode.AGENCYSCHEME


def test_get_config_url_resolves_env_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDMX_PROXY_CONFIG_SERVER_HOST", "http://config-server:8060")
    assert _proxy_config().get_config_url() == (
        "http://config-server:8060/statgpt/sdmx-proxy-config-server/api/v0/config"
    )


def test_get_config_url_keeps_unresolved_env_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SDMX_PROXY_CONFIG_SERVER_HOST", raising=False)
    assert _proxy_config().get_config_url().startswith("$env:{SDMX_PROXY_CONFIG_SERVER_HOST}")


def test_proxy_config_accepts_both_field_spellings() -> None:
    payload = {"configs": [], "agencies": [], "structureFanOutEnabled": True}
    assert _proxy_config(proxy_config=payload).proxy_config == payload
    assert _proxy_config(proxyConfig=payload).proxy_config == payload


def test_proxy_config_is_not_persisted() -> None:
    config = _proxy_config(proxy_config={"agencies": []})

    assert "proxyConfig" not in config.dump_for_storage()
    assert config.model_dump(mode='json', by_alias=True)["proxyConfig"] == {"agencies": []}


def test_proxy_config_is_exposed_in_the_json_schema() -> None:
    """The admin portal renders its editor from this schema."""
    assert "proxyConfig" in StatGptSdmxProxyDataSourceConfig.model_json_schema()["properties"]


def test_matches_stored_ignores_proxy_config_when_incoming_omits_it() -> None:
    incoming = _proxy_config()
    stored = _proxy_config(proxy_config={"agencies": [{"name": "IMF"}]})

    assert incoming.matches_stored(stored)


def test_matches_stored_detects_a_changed_proxy_config() -> None:
    incoming = _proxy_config(proxy_config={"agencies": [{"name": "IMF"}]})
    stored = _proxy_config(proxy_config={"agencies": []})

    assert not incoming.matches_stored(stored)
    assert incoming.matches_stored(_proxy_config(proxy_config={"agencies": [{"name": "IMF"}]}))


def test_matches_stored_detects_a_changed_persisted_field() -> None:
    incoming = _proxy_config(locale="fr")
    stored = _proxy_config(locale="en")

    assert not incoming.matches_stored(stored)


class TestExternalDetails:
    """The config server owns `proxyConfig`, so the config loads and pushes it on request."""

    @pytest.fixture(autouse=True)
    def config_server_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SDMX_PROXY_CONFIG_SERVER_HOST", "http://config-server:8060")

    def test_the_key_is_the_config_server_url(self) -> None:
        assert _proxy_config().external_details_key() == _URL

    async def test_load_returns_the_served_configuration(self, mocker) -> None:
        stored = {"agencies": [{"name": "IMF"}]}
        fetch = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=stored),
        )

        assert await _proxy_config().load_external_details() == {"proxyConfig": stored}
        fetch.assert_awaited_once_with(_URL)

    async def test_load_reports_a_cold_start_as_null(self, mocker) -> None:
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(return_value=None),
        )

        assert await _proxy_config().load_external_details() == {"proxyConfig": None}

    async def test_load_propagates_a_config_server_failure(self, mocker) -> None:
        mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.fetch_proxy_config",
            AsyncMock(side_effect=ProxyConfigServerError("connection refused")),
        )

        with pytest.raises(ProxyConfigServerError):
            await _proxy_config().load_external_details()

    async def test_push_sends_the_submitted_configuration(self, mocker) -> None:
        submitted = {"agencies": [{"name": "IMF"}]}
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )

        await _proxy_config(proxy_config=submitted).push_external_details()

        push.assert_awaited_once_with(_URL, submitted)

    async def test_push_leaves_the_config_server_alone_when_nothing_is_submitted(
        self, mocker
    ) -> None:
        push = mocker.patch(
            "statgpt.common.data.statgpt_sdmx_proxy.config.push_proxy_config", AsyncMock()
        )

        await _proxy_config().push_external_details()

        push.assert_not_awaited()


def test_a_data_source_without_externally_owned_details_has_no_key() -> None:
    """The base hook opts out, so the admin service skips such data sources entirely."""
    config = SdmxDataSourceConfig(
        sdmx_config=SdmxConfig(id="plain", url="https://example.invalid", name="plain")
    )
    assert config.external_details_key() is None
