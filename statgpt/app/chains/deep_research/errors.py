from statgpt.app.utils.dial_stages import ChoiceI

# User-facing message shown when a Deep Research turn fails. Shared by the tool (mid-stream /
# request errors) and the Supreme Agent (framework-level errors that never reach the tool) so
# both failure paths surface identical wording.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


def surface_deep_research_error(choice: ChoiceI) -> str:
    """Stream the standard Deep Research failure message to the user and return it.

    Every failure path — the tool's own request/stream errors and framework-level errors that
    only surface as an ERROR tool message — funnels through here so the wording, and the single
    append-and-return, live in one place."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE
