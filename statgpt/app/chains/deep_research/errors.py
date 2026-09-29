from statgpt.app.utils.dial_stages import ChoiceI

_DEEP_RESEARCH_ERROR_TEXT = "Deep Research could not complete this request. Please try again."

# User-facing message shown when a Deep Research turn fails, Surfaced by the Supreme Agent.
DEEP_RESEARCH_ERROR_MESSAGE = f"\n\n{_DEEP_RESEARCH_ERROR_TEXT}"


class DeepResearchFailedError(Exception):
    """The Deep Research deployment failed; aborts the turn and is delivered as a DIAL error.

    Carries the user-safe ``display_message`` the deployment reported (with its error reference),
    or the standard text when it sent none. Mapped to a DIAL error by `ChannelCompletion`."""

    def __init__(self, display_message: str | None = None):
        self.display_message = display_message or _DEEP_RESEARCH_ERROR_TEXT
        super().__init__(self.display_message)


def surface_deep_research_error(choice: ChoiceI) -> str:
    """Stream the standard Deep Research failure message to the user and return it.

    Failures other than a deployment error (`DeepResearchFailedError`) reach the Supreme Agent as
    an ERROR tool message recorded by `ToolCaller.call_tool`, and the mediation safety cap ends the
    same way — so both paths funnel through here and the wording, and the single
    append-and-return, live in one place."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE
