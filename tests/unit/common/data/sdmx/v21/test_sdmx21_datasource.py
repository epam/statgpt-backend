"""Tests for the SDMX 2.1 data source handler."""

from sdmx.message import StructureMessage
from sdmx.model.common import Agency
from sdmx.model.v21 import DataflowDefinition

from statgpt.common.data.sdmx.common.config import (
    ProviderDiscoveryMode,
    SdmxConfig,
    SdmxDataSourceConfig,
)
from statgpt.common.data.sdmx.v21.datasource import Sdmx21DataSourceHandler
from statgpt.common.schemas.data_source import Provider


def _sdmx_config(**overrides) -> SdmxDataSourceConfig:
    return SdmxDataSourceConfig(
        sdmx_config=SdmxConfig(id="sdmx", url="https://example.invalid", name="sdmx"),
        **overrides,
    )


class _StubClient:
    def __init__(self, message: StructureMessage):
        self._message = message

    async def dataflow(self, *, agency_id, resource_id, version, params=None, use_cache=False):
        return self._message


class _StubSdmx21Handler(Sdmx21DataSourceHandler):
    """Test double for the dataflows discovery mode."""

    def __init__(self, config: SdmxDataSourceConfig, message: StructureMessage):
        super().__init__(config)
        self._stub_client = _StubClient(message)

    async def create_sdmx_client(self, auth_context):  # type: ignore[override]
        return self._stub_client  # type: ignore[return-value]


async def test_sdmx21_list_providers_via_dataflows_dedupes_agency_ids() -> None:
    message = StructureMessage()
    for agency_id, resource_id in [("IMF.RES", "WEO"), ("IMF.RES", "IFS"), ("BIS", "LBS")]:
        dataflow = DataflowDefinition(
            id=resource_id, maintainer=Agency(id=agency_id), version="1.0"
        )
        message.add(dataflow)

    handler = _StubSdmx21Handler(
        _sdmx_config(provider_discovery=ProviderDiscoveryMode.DATAFLOWS), message
    )
    providers = await handler.list_providers(auth_context=None)  # type: ignore[arg-type]

    assert providers == [
        Provider(id="BIS", name="BIS"),
        Provider(id="IMF.RES", name="IMF.RES"),
    ]
