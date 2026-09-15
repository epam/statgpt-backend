"""The payload-budget guard for model-facing MCP tool results (#601): measuring the serialized
size, degrading by dropping embedded resources, and the hard error when a result cannot be made
to fit."""

import json

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.mcp.payload_budget import enforce_payload_budget, measure_tool_result_size


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _resource(text: str, *, ext: str = "csv", mime: str = "text/csv") -> EmbeddedResource:
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl(f"statgpt://data_query/x/y.{ext}"), mimeType=mime, text=text
        ),
    )


# ~~~~~~~~~~~~~ measurement ~~~~~~~~~~~~~


def test_measures_structured_content_and_blocks():
    result = ToolResult(content=[_text("hello")], structured_content={"a": 1})
    expected = len(json.dumps({"a": 1}, separators=(",", ":"))) + len(
        _text("hello").model_dump_json()
    )
    assert measure_tool_result_size(result) == expected


def test_measures_result_without_structured_content():
    result = ToolResult(content=[_text("hi")])
    assert measure_tool_result_size(result) == len(_text("hi").model_dump_json())


def test_non_ascii_structured_content_measured_as_utf8_not_escaped():
    # Non-ASCII text is measured as UTF-8 (as the wire carries it), not as escaped \uXXXX, so a
    # multi-language payload is not counted at several times its real size.
    sc = {"country": "Україна"}
    result = ToolResult(content=[], structured_content=sc)
    compact_utf8 = len(json.dumps(sc, ensure_ascii=False, separators=(",", ":")))
    ascii_escaped = len(json.dumps(sc))  # the naive default would over-count
    assert ascii_escaped > compact_utf8
    assert measure_tool_result_size(result) == compact_utf8


# ~~~~~~~~~~~~~ enforcement ~~~~~~~~~~~~~


def test_within_budget_returns_the_same_result():
    result = ToolResult(content=[_text("hi")], structured_content={"a": 1})
    assert enforce_payload_budget(result, tool_name="t", budget=10_000) is result


def test_over_budget_drops_resources_and_keeps_structured_and_text():
    result = ToolResult(
        content=[_text("summary"), _resource("x" * 5000)],
        structured_content={"status": "ok"},
    )

    degraded = enforce_payload_budget(result, tool_name="data_query", budget=1000)

    assert not any(isinstance(b, EmbeddedResource) for b in degraded.content)
    assert degraded.structured_content == {"status": "ok"}
    texts = [b.text for b in degraded.content if isinstance(b, TextContent)]
    assert "summary" in texts  # the original summary block survives
    assert any("omitted" in t for t in texts)  # a note explains the drop
    assert measure_tool_result_size(degraded) <= 1000


def test_hard_error_when_no_resources_to_drop():
    # A structured-only result that is itself over budget cannot be trimmed: surface an error.
    result = ToolResult(content=[], structured_content={"items": ["y" * 100 for _ in range(100)]})
    with pytest.raises(ToolError, match="too large"):
        enforce_payload_budget(result, tool_name="available_datasets", budget=500)


def test_hard_error_when_dropping_resources_is_not_enough():
    result = ToolResult(content=[_resource("x" * 100)], structured_content={"big": "z" * 2000})
    with pytest.raises(ToolError, match="too large"):
        enforce_payload_budget(result, tool_name="data_query", budget=500)
