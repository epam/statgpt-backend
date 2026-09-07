"""Marketplace-policy lint for MCP tool metadata.

Both the OpenAI Apps SDK submission guidelines and the Anthropic directory policy reject
tool descriptions/annotations that try to steer the host model's tool selection: priority
or exclusivity claims, coercive or shouting language, superlatives and comparisons with
other providers, and instructions to call or avoid other tools. Such language is easy to
reintroduce in a routine edit, so this module scans the model-visible metadata a channel
exposes over MCP against a ban list and reports any hit.

The check is a heuristic gate, not a replacement for human review: it catches the common,
drift-prone patterns cheaply. It runs only on channels that opt in via
``McpConfig.enforce_marketplace_policy`` - internal channels intentionally keep steering
language for the Supreme Agent and are left alone.

The description linted is ``mcp_description`` directly, not ``effective_mcp_description``:
a channel serving the same tools to both the Supreme Agent and MCP keeps steering language
in the agent-facing ``description`` and a clean override in ``mcp_description``. A
model-visible tool that omits ``mcp_description`` would ship its agent description to MCP,
so a missing override is itself a violation - the agent text can never fall through
unchecked.
"""

import re

from pydantic import BaseModel

from statgpt.common.schemas import BaseToolConfig, ChannelConfig

# Each rule is (name, compiled pattern). A hit on any pattern is a violation. Patterns are
# case-insensitive unless noted; keep them narrow enough that accurate, self-scoped
# "use when / do not use when" descriptions pass.
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Shouting: three or more consecutive all-caps words (e.g. "DRAMATICALLY REDUCES
    # QUALITY"). Case-sensitive so ordinary prose and lone acronyms (GDP, CPI) pass.
    ("shouting_caps", re.compile(r"\b[A-Z]{2,}(?:[\s,]+[A-Z]{2,}){2,}\b")),
    # Exclamation marks read as emphasis aimed at the model, not a human reader.
    ("exclamation", re.compile(r"!")),
    # Priority / exclusivity claims.
    (
        "priority_claim",
        re.compile(
            r"\b(?:always\s+(?:call|use|prefer|run|invoke)"
            r"|call\s+(?:this|it|me)(?:\s+tool)?\s+first"
            r"|use\s+(?:this|me)(?:\s+tool)?\s+(?:first|instead)"
            r"|before\s+(?:calling|using|invoking|any\s+other)"
            r"|(?:highest|top)\s+priority"
            r"|prefer(?:red)?\b"
            r"|instead\s+of\s+(?:any\s+)?other)\b",
            re.IGNORECASE,
        ),
    ),
    # Coercion / urgency directed at the model.
    (
        "coercion",
        re.compile(
            r"\b(?:you\s+must"
            r"|it\s+is\s+(?:very\s+|really\s+)?(?:important|essential|critical|crucial)"
            r"|failure\s+to"
            r"|loss\s+of\s+(?:user\s+)?trust"
            r"|do\s+not\s+forget"
            r"|dramatically"
            r"|never"
            r"|always)\b",
            re.IGNORECASE,
        ),
    ),
    # Superlatives / comparisons with other services.
    (
        "superlative",
        re.compile(
            r"\b(?:best|better|worse|superior|inferior"
            r"|works\s+best"
            r"|most\s+(?:accurate|reliable|comprehensive|powerful|advanced)"
            r"|more\s+(?:accurate|reliable|powerful)\s+than"
            r"|unlike\s+other)\b",
            re.IGNORECASE,
        ),
    ),
    # Cross-tool steering: telling the model to reach for other tools/resources.
    (
        "cross_tool_steering",
        re.compile(
            r"\b(?:refer\s+to"
            r"|consult(?:\s+the)?"
            r"|use\s+the\s+.{1,40}?\s+tool"
            r"|other\s+tools?)\b",
            re.IGNORECASE,
        ),
    ),
]


# Rule name for a model-visible tool that declares no explicit MCP description: without
# one the agent-facing `description` is what ships to MCP, which the policy forbids.
MISSING_MCP_DESCRIPTION = "missing_mcp_description"


class MetadataViolation(BaseModel):
    """A single ban-list hit on a tool's model-visible metadata."""

    deployment_id: str | None
    tool_name: str
    field: str  # "name" or "description"
    rule: str
    match: str


def lint_text(text: str) -> list[tuple[str, str]]:
    """Return ``(rule, matched_substring)`` for every ban-list rule the text hits."""
    hits: list[tuple[str, str]] = []
    for rule, pattern in _BANNED_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append((rule, match.group(0)))
    return hits


def _is_model_visible(tool: BaseToolConfig) -> bool:
    """Whether the host model sees this tool (per the MCP Apps ``visibility`` default)."""
    visibility = tool.mcp_visibility if tool.mcp_visibility is not None else ["model", "app"]
    return "model" in visibility


def lint_tool(tool: BaseToolConfig, deployment_id: str | None = None) -> list[MetadataViolation]:
    """Lint the name and description a model-visible tool exposes over MCP.

    The name is the published identifier (``mcp_name``, or the agent ``name`` when unset);
    tool names are constrained identifiers, so publishing the agent name is acceptable.

    The description is ``mcp_description`` directly: it must be set for a model-visible tool
    on an opted-in channel, and a missing one is a violation rather than a silent fall
    through to the agent-facing ``description``.
    """
    violations: list[MetadataViolation] = []

    for rule, match in lint_text(tool.effective_mcp_name):
        violations.append(
            MetadataViolation(
                deployment_id=deployment_id,
                tool_name=tool.effective_mcp_name,
                field="name",
                rule=rule,
                match=match,
            )
        )

    if tool.mcp_description is None:
        violations.append(
            MetadataViolation(
                deployment_id=deployment_id,
                tool_name=tool.effective_mcp_name,
                field="description",
                rule=MISSING_MCP_DESCRIPTION,
                match="",
            )
        )
    else:
        for rule, match in lint_text(tool.mcp_description):
            violations.append(
                MetadataViolation(
                    deployment_id=deployment_id,
                    tool_name=tool.effective_mcp_name,
                    field="description",
                    rule=rule,
                    match=match,
                )
            )
    return violations


def lint_channel(
    channel: ChannelConfig, deployment_id: str | None = None
) -> list[MetadataViolation]:
    """Lint a channel's model-visible tool metadata, if it opts into the policy.

    Returns an empty list for channels that do not set
    ``mcp.enforce_marketplace_policy`` - so a caller can lint every channel and only the
    opted-in ones contribute violations.
    """
    if not channel.mcp.enforce_marketplace_policy:
        return []
    violations: list[MetadataViolation] = []
    for tool in channel.tools:
        if not _is_model_visible(tool):
            continue
        violations.extend(lint_tool(tool, deployment_id))
    return violations


def format_violations(violations: list[MetadataViolation]) -> str:
    """Render violations as a readable, one-per-line report for a failing check."""
    lines = [
        f"  [{v.deployment_id or '-'}] {v.tool_name}.{v.field}: {v.rule} -> {v.match!r}"
        for v in violations
    ]
    return "\n".join(lines)
