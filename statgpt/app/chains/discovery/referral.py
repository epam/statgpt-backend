"""Rendering a discovery search result as the referral appended to a no-data answer."""

from statgpt.app.schemas.discovery import DiscoverySearchResult

_HEADING = "### Datasets available at their official source"

_PREAMBLE = (
    "The data was not found in this system, but the following official datasets describe data"
    " that may answer the request. They are not onboarded, so they cannot be queried here - each"
    " links to its publisher's own portal."
)

GROUNDING_RULE = (
    "AGENT INSTRUCTIONS: the datasets listed above are descriptions of datasets, not data."
    " Present them to the user with their links, and say plainly that this system cannot query"
    " them. Do not state, estimate or infer any figure, value, trend or date from these"
    " descriptions, and do not use them to answer the original question. Use them only to name,"
    " describe and link the datasets."
)


def render_referral(result: DiscoverySearchResult) -> str:
    """Render the referral block, or an empty string when there is nothing to refer to.

    The block carries its own instructions to the agent. A discovery record's indicator list
    reads exactly like something the agent could answer from, so the rule that bounds it travels
    with the data rather than living only in the system prompt.
    """
    if not result.has_referral:
        return ""

    lines = [_HEADING, "", _PREAMBLE, ""]
    for item in result.items:
        candidate = item.candidate
        lines.append(f"- **{candidate.label}**")
        details = (
            ("Agency", candidate.agency),
            ("Coverage", candidate.reference_area),
            ("Period", candidate.time_coverage),
            ("Frequency", candidate.frequency_coverage),
        )
        lines.extend(f"  - {label}: {value}" for label, value in details if value)
        if candidate.url:
            lines.append(f"  - Source: {candidate.url}")
        if item.reason:
            lines.append(f"  - Why: {item.reason}")
        if item.missing:
            lines.append(f"  - Not covered: {item.missing}")

    lines.extend(["", GROUNDING_RULE])
    return "\n".join(lines)
