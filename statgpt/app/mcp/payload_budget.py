"""Keep model-facing MCP tool results within the host's payload limits (#601).

Anthropic hosts cap tool results (~150,000 characters on Claude.ai/Desktop). A broad SDMX query can
return far more than that, and the host then truncates the result mid-payload, which the user
experiences as a broken tool rather than a query that needs narrowing.

The guard measures the serialized result and, when it is over budget, first drops the heavy embedded
resources (the CSV/Markdown tables, which are the bulk of a data-query payload) and leaves the
compact `structuredContent` and a note behind. If the result still does not fit, it returns an
actionable error asking the caller to narrow the query, rather than a blob the host will corrupt.
"""

import json
import logging

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ContentBlock, EmbeddedResource, TextContent

_log = logging.getLogger(__name__)

_NARROW_HINT = (
    "Narrow the query (for example a shorter time range, fewer dimensions, or fewer series) "
    "and try again."
)


def _block_size(block: ContentBlock) -> int:
    return len(block.model_dump_json())


def measure_tool_result_size(result: ToolResult) -> int:
    """Approximate the serialized size, in characters, of what the host receives for this result:
    the JSON of `structuredContent` plus the JSON of every content block.

    Both parts are measured as compact UTF-8 JSON. `structured_content` is serialized with
    `ensure_ascii=False` and no separator padding so it matches `model_dump_json` (used for the
    content blocks) and the MCP wire format. Without `ensure_ascii=False`, non-ASCII text - common
    in a multi-language deployment - would be counted as six-character escapes and inflate the
    estimate several-fold over what the host actually receives.
    """
    total = 0
    if result.structured_content is not None:
        total += len(
            json.dumps(
                result.structured_content, default=str, ensure_ascii=False, separators=(",", ":")
            )
        )
    total += sum(_block_size(block) for block in result.content)
    return total


def enforce_payload_budget(result: ToolResult, *, tool_name: str, budget: int) -> ToolResult:
    """Return `result` unchanged when it fits `budget`, a degraded result when dropping its embedded
    resources makes it fit, or raise `ToolError` when it cannot be made to fit.

    Called only for model-facing tools (app-only tools are exempt) and only when the budget guard is
    enabled; the caller owns both checks.
    """
    size = measure_tool_result_size(result)
    if size <= budget:
        return result

    resources = [block for block in result.content if isinstance(block, EmbeddedResource)]
    if resources:
        kept = [block for block in result.content if not isinstance(block, EmbeddedResource)]
        dropped_chars = sum(_block_size(block) for block in resources)
        note = TextContent(
            type="text",
            text=(
                f"[{len(resources)} data resource(s), ~{dropped_chars} characters, were omitted "
                f"because the response exceeded the {budget}-character size budget.] {_NARROW_HINT}"
            ),
        )
        degraded = ToolResult(content=[*kept, note], structured_content=result.structured_content)
        degraded_size = measure_tool_result_size(degraded)
        if degraded_size <= budget:
            _log.info(
                "MCP payload over budget for tool=%s (%d > %d chars); dropped %d embedded "
                "resource(s) to fit (%d chars).",
                tool_name,
                size,
                budget,
                len(resources),
                degraded_size,
            )
            return degraded
        size = degraded_size

    _log.warning(
        "MCP payload over budget for tool=%s (%d > %d chars) and cannot be reduced to fit.",
        tool_name,
        size,
        budget,
    )
    raise ToolError(
        f"The result of '{tool_name}' is too large to return ({size} characters, over the "
        f"{budget}-character limit). {_NARROW_HINT}"
    )
