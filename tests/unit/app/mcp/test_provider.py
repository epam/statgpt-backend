from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError
from mcp.types import EmbeddedResource, TextContent

from statgpt.app.mcp.provider import _McpToolAdapter
from statgpt.app.schemas.tool_artifact import DataQueryArtifact


def _build_adapter(result) -> _McpToolAdapter:
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(return_value=result))
    return _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        name="fake_tool",
        parameters={},
    )


async def test_data_query_artifact_adds_csv_resources():
    df = pd.DataFrame({"x": [1, 2]})
    response = SimpleNamespace(
        resource_path="IMF:CPI(1.0.0)",
        visual_dataframe=df,
        csv_dataframe=df,
        created_at=datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc),
    )
    artifact = DataQueryArtifact.model_construct(data_responses={"ds1": response})
    result = SimpleNamespace(content="answer", artifact=artifact)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "answer"
    resources = [c for c in tool_result.content if isinstance(c, EmbeddedResource)]
    assert len(resources) == 1
    assert resources[0].resource.mimeType == "text/csv"


async def test_non_data_query_result_returns_text_only():
    result = SimpleNamespace(content="hello", artifact=None)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert len(tool_result.content) == 1
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "hello"


async def test_non_string_content_is_stringified():
    result = SimpleNamespace(content=[{"a": 1}], artifact=None)
    adapter = _build_adapter(result)

    tool_result = await adapter.run({})

    assert len(tool_result.content) == 1
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "[{'a': 1}]"


async def test_tool_failure_raises_tool_error():
    tool = SimpleNamespace(name="fake_tool", ainvoke=AsyncMock(side_effect=RuntimeError("boom")))
    adapter = _McpToolAdapter(
        langchain_tool=tool,  # type: ignore[arg-type]
        inputs={},
        name="fake_tool",
        parameters={},
    )

    with pytest.raises(ToolError):
        await adapter.run({})
