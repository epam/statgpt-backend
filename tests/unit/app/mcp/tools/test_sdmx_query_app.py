from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

from statgpt.app.chains.sdmx_query_app import SdmxProxyResponse
from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.app.mcp.tools.sdmx_query_app import SdmxQueryAppMcpTool
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.schemas.tools import SdmxQueryAppTool


def _build(response: SdmxProxyResponse | None = None, error: Exception | None = None):
    tool_config = SdmxQueryAppTool(
        name="sdmx",
        description="SDMX passthrough.",
        details=SdmxQueryAppDetails.model_validate({"base_url": "https://example.test"}),
    )
    tool = StatGptMcpTool.from_config(
        tool_config,
        SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=""), out_of_scope=None),  # type: ignore[arg-type]
        inputs={},
        auth_context=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert isinstance(tool, SdmxQueryAppMcpTool)
    forward = AsyncMock(return_value=response, side_effect=error)
    tool._proxy.forward = forward  # type: ignore[method-assign]
    return tool, forward


async def test_passthrough_exposes_body_and_http_metadata():
    tool, forward = _build(SdmxProxyResponse("<xml/>", 200, "application/json"))

    tool_result = await tool.run({"path": "/structure/dataflow/IMF/CPI/1.0.0"})

    assert tool_result.content == [TextContent(type="text", text="<xml/>")]
    assert tool_result.structured_content == {
        "statusCode": 200,
        "contentType": "application/json",
    }
    forward.assert_awaited_once_with(
        path="/structure/dataflow/IMF/CPI/1.0.0", method="GET", body=None, accept=None
    )


async def test_upstream_error_responses_are_passed_through():
    tool, _ = _build(SdmxProxyResponse("not found", 404, "text/plain"))

    tool_result = await tool.run({"path": "/structure/nope"})

    assert tool_result.structured_content == {"statusCode": 404, "contentType": "text/plain"}


async def test_absolute_urls_are_rejected_before_forwarding():
    tool, forward = _build()

    with pytest.raises(ToolError, match="Invalid arguments"):
        await tool.run({"path": "https://evil.test/structure"})

    forward.assert_not_called()


async def test_upstream_failure_surfaces_its_message():
    tool, _ = _build(error=ToolUpstreamError("Could not reach the SDMX backend: boom"))

    with pytest.raises(ToolError, match="Could not reach the SDMX backend"):
        await tool.run({"path": "/structure/dataflow"})


def test_defaults_to_app_only_visibility():
    tool, _ = _build()

    assert tool.meta == {"ui": {"visibility": ["app"]}}
    assert tool.output_schema is None
