from typing import Any, Protocol, Self

from pydantic import BaseModel, Field

from statgpt.app.config.state import StateVarsConfig

# User-facing message shown when a Deep Research turn fails. Shared by the tool (mid-stream /
# request errors) and the Supreme Agent (framework-level errors that never reach the tool) so
# both failure paths surface identical wording.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


class _SupportsAppendContent(Protocol):
    """Minimal streaming surface (e.g. a DIAL choice/stage) needed to show a message."""

    def append_content(self, content: str) -> None: ...


def surface_deep_research_error(choice: _SupportsAppendContent) -> str:
    """Stream the standard Deep Research failure message to the user and return it.

    Every failure path — the tool's own request/stream errors and framework-level errors that
    only surface as an ERROR tool message — funnels through here so the wording, and the single
    append-and-return, live in one place."""
    choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
    return DEEP_RESEARCH_ERROR_MESSAGE


class DeepResearchTurn(BaseModel):
    """One Supreme Agent <-> Deep Research exchange, persisted in DIAL state.

    Stored with neutral field names on purpose. The sub-conversation is *not* kept in
    DIAL-message shape (``role``/``content``/``custom_content``): the DIAL chat client
    strips ``custom_content`` from any message-shaped object it finds while round-tripping
    state, which would drop Deep Research's own ``dr_state`` and force a re-plan every turn.
    The DIAL messages are rebuilt from these fields only when calling the deployment.
    """

    user_message: str
    assistant_content: str
    # Deep Research's own ``custom_content.state`` from this turn, replayed so it can resume.
    dr_state: dict[str, Any] = Field(default_factory=dict)


class DeepResearchSession(BaseModel):
    """Supreme Agent <-> Deep Research conversation, persisted in DIAL state.

    Carries the multi-turn clarification flow across user turns as a list of
    :class:`DeepResearchTurn`. Kept separate from the user-facing chat history so Deep
    Research's intermediate turns do not pollute the Supreme Agent context.

    The session lives in state only while the clarification flow is in progress; once Deep
    Research delivers the final report the tool drops it, so its mere presence signals an
    in-progress run.
    """

    turns: list[DeepResearchTurn] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict) -> Self | None:
        """Load and validate the session stored in DIAL state, or ``None`` if absent."""
        if raw := state.get(StateVarsConfig.DEEP_RESEARCH_SESSION):
            return cls.model_validate(raw)
        return None

    @staticmethod
    def drop_from_state(state: dict) -> None:
        """Remove the session from DIAL state, if present (idempotent)."""
        state.pop(StateVarsConfig.DEEP_RESEARCH_SESSION, None)
