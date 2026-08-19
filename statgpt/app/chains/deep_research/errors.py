from statgpt.app.utils.dial_stages import ChoiceI

# User-facing message shown when a Deep Research turn fails, Surfaced by the Supreme Agent.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


def surface_deep_research_error(choice: ChoiceI) -> str:
    """Stream the standard Deep Research failure message to the user and return it.

    Every failure reaches the Supreme Agent as an ERROR tool message — the tool's own
    request/stream errors propagate and are recorded there by `ToolCaller.call_tool`, and
    framework-level errors surface the same way — so both failure paths funnel through here and the
    wording, and the single append-and-return, live in one place."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE
