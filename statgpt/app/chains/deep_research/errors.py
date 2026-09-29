from statgpt.app.utils.dial_stages import ChoiceI

# User-facing message shown when a Deep Research turn fails, Surfaced by the Supreme Agent.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


def surface_deep_research_error(choice: ChoiceI, display_message: str | None = None) -> str:
    """Stream the Deep Research failure message to the user and return it.

    Every failure reaches the Supreme Agent as an ERROR tool message — the tool's own
    request/stream errors propagate and are recorded there by `ToolCaller.call_tool`, and
    framework-level errors surface the same way — so both failure paths funnel through here and the
    wording, and the single append-and-return, live in one place.

    ``display_message`` is the user-facing reason reported by the deployment (see
    `DeepResearchRunner.run`); the standard message is shown when there is none."""
    message = f"\n\n{display_message}" if display_message else DEEP_RESEARCH_ERROR_MESSAGE
    choice.append_content(message)
    return message
