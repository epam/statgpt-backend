from types import SimpleNamespace
from unittest.mock import AsyncMock

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.app.schemas.service import ChannelDatasetsMetadataResponse
from statgpt.common.schemas.tools import DatasetsMetadataAppTool


async def test_exposes_the_metadata_payload_as_structured_content(monkeypatch):
    response = ChannelDatasetsMetadataResponse(
        deployment_id="dep", title="Channel", n_datasets=0, datasets=[]
    )
    build = AsyncMock(return_value=response)
    monkeypatch.setattr(
        "statgpt.app.mcp.tools.datasets_metadata_app.build_channel_datasets_metadata", build
    )
    channel = SimpleNamespace()
    auth_context = SimpleNamespace()
    tool = StatGptMcpTool.from_config(
        DatasetsMetadataAppTool(name="datasets_metadata", description="Datasets metadata."),
        SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=""), out_of_scope=None),  # type: ignore[arg-type]
        inputs={ChainParametersConfig.DATA_SERVICE: SimpleNamespace(channel=channel)},
        auth_context=auth_context,  # type: ignore[arg-type]
    )

    tool_result = await tool.run({})

    # The widget consumes the payload directly; there is no text rendering.
    assert tool_result.content == []
    assert tool_result.structured_content == {
        "deployment_id": "dep",
        "title": "Channel",
        "n_datasets": 0,
        "datasets": [],
    }
    build.assert_awaited_once_with(channel, auth_context)
    assert tool.meta == {"ui": {"visibility": ["app"]}}
