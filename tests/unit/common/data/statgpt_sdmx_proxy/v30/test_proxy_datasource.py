"""Tests for the StatGPT SDMX proxy data source handler."""

from pathlib import Path

import httpx
import pytest
from sdmx.message import StructureMessage
from sdmx.model.common import Agency, AgencyScheme, InternationalString

from statgpt.common.data.base.datasource import ProviderRequiredError
from statgpt.common.data.sdmx.common.config import SdmxConfig
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.v30.datasource import StatGptSdmxProxyDataSourceHandler
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import AsyncStatGptSdmxProxyClient
from statgpt.common.schemas.data_source import Provider

FIXTURE = Path(__file__).parent / "agency_schemes_response.json"
STRUCTURE_CONTENT_TYPE = "application/vnd.sdmx.structure+json;version=2.0.0"


def _proxy_config(**overrides) -> StatGptSdmxProxyDataSourceConfig:
    return StatGptSdmxProxyDataSourceConfig(
        sdmx_config=SdmxConfig(id="proxy", url="https://example.invalid", name="proxy"),
        **overrides,
    )


def _agency_scheme_message(agencies: list[Agency]) -> StructureMessage:
    msg = StructureMessage()
    scheme = AgencyScheme(id="AGENCIES")
    for agency in agencies:
        scheme.items[agency.id] = agency
    msg.add(scheme)
    return msg


def _mock_transport(body: bytes) -> httpx.MockTransport:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": STRUCTURE_CONTENT_TYPE}, content=body)

    return httpx.MockTransport(respond)


class _StubClient:
    def __init__(self, message: StructureMessage):
        self._message = message

    async def agencyscheme(
        self, *, agency_id, resource_id, version, use_cache=False, extra_headers=None
    ):
        return self._message


class _StubProxyHandler(StatGptSdmxProxyDataSourceHandler):
    """Test double that returns a canned StructureMessage instead of hitting HTTP."""

    def __init__(self, config: StatGptSdmxProxyDataSourceConfig, message: StructureMessage):
        super().__init__(config)
        self._stub_client = _StubClient(message)

    async def create_sdmx_client(self, auth_context):  # type: ignore[override]
        return self._stub_client  # type: ignore[return-value]


async def test_proxy_list_datasets_requires_provider() -> None:
    handler = StatGptSdmxProxyDataSourceHandler(_proxy_config())
    with pytest.raises(ProviderRequiredError):
        await handler.list_datasets(auth_context=None)  # type: ignore[arg-type]


async def test_proxy_list_providers_parses_real_agency_scheme_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _mock_transport(FIXTURE.read_bytes())
    monkeypatch.setattr(
        AsyncStatGptSdmxProxyClient,
        "_create_httpx_client",
        staticmethod(lambda *, headers=None: httpx.AsyncClient(transport=transport)),
    )

    handler = StatGptSdmxProxyDataSourceHandler(_proxy_config())
    providers = await handler.list_providers(auth_context=None)  # type: ignore[arg-type]

    assert providers == [
        Provider(id="BIS", name="Bank for International Settlements"),
        Provider(id="IMF", name="International Monetary Fund"),
    ]


async def test_proxy_list_providers_falls_back_to_id_when_localized_name_missing() -> None:
    agency = Agency(id="OECD")
    agency.name = InternationalString()
    message = _agency_scheme_message([agency])

    handler = _StubProxyHandler(_proxy_config(), message)
    providers = await handler.list_providers(auth_context=None)  # type: ignore[arg-type]

    assert providers == [Provider(id="OECD", name="OECD")]


async def test_proxy_list_providers_falls_back_to_any_localization() -> None:
    agency = Agency(id="OECD")
    agency.name = InternationalString()
    agency.name.localizations = {"fr": "Organisation de coopération et de développement"}
    message = _agency_scheme_message([agency])

    handler = _StubProxyHandler(_proxy_config(locale="en"), message)
    providers = await handler.list_providers(auth_context=None)  # type: ignore[arg-type]

    assert providers == [
        Provider(id="OECD", name="Organisation de coopération et de développement"),
    ]
