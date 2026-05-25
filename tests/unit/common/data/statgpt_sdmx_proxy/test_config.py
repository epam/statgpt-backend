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
