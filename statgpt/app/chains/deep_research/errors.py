from statgpt.app.utils.dial_stages import ChoiceI

# Without "try again": the Continue note that follows the DIAL error warns against "Try again".
DEEP_RESEARCH_ERROR_TEXT = "Deep Research could not complete this request."

# Appended to an answer that otherwise succeeds, so it has no Continue button or note.
DEEP_RESEARCH_ERROR_MESSAGE = f"\n\n{DEEP_RESEARCH_ERROR_TEXT} Please try again."


class DeepResearchFailedError(Exception):
    """The Deep Research deployment failed; mapped to a DIAL error by `ChannelCompletion`."""

    def __init__(self, display_message: str | None = None):
        self.display_message = display_message or DEEP_RESEARCH_ERROR_TEXT
        super().__init__(self.display_message)


def surface_deep_research_error(choice: ChoiceI) -> str:
    """Stream the standard Deep Research failure message to the user and return it."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE
