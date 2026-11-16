from statgpt.app.utils.dial_stages import ChoiceI

_DEEP_RESEARCH_ERROR_TEXT = "Deep Research could not complete this request. Please try again."

DEEP_RESEARCH_ERROR_MESSAGE = f"\n\n{_DEEP_RESEARCH_ERROR_TEXT}"


class DeepResearchFailedError(Exception):
    """The Deep Research deployment failed; mapped to a DIAL error by `ChannelCompletion`."""

    def __init__(self, display_message: str | None = None):
        self.display_message = display_message or _DEEP_RESEARCH_ERROR_TEXT
        super().__init__(self.display_message)


def surface_deep_research_error(choice: ChoiceI) -> str:
    """Stream the standard Deep Research failure message to the user and return it."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE
