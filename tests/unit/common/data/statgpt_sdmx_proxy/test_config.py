"""Config-validation tests for the StatGPT SDMX proxy data source."""

import pytest
from pydantic import ValidationError

from statgpt.common.data.sdmx.common.config import ProviderDiscoveryMode, SdmxConfig
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig


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
