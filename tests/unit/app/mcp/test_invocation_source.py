from types import SimpleNamespace

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.provider import _build_mcp_inputs
from statgpt.common.schemas.enums import InvocationSource


def test_build_mcp_inputs_marks_the_mcp_flow():
    """The MCP path must be distinguishable from the Supreme Agent path."""
    inputs = _build_mcp_inputs(SimpleNamespace(), SimpleNamespace())  # type: ignore[arg-type]
    assert inputs[ChainParametersConfig.INVOCATION_SOURCE] is InvocationSource.MCP
