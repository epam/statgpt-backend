from typing import Any, Self

from pydantic import BaseModel, Field

from statgpt.app.config.state import StateVarsConfig


class DeepResearchTurn(BaseModel):
    """One Supreme Agent <-> Deep Research exchange, persisted in DIAL state.

    Stored with neutral field names on purpose. The sub-conversation is *not* kept in
    DIAL-message shape (``role``/``content``/``custom_content``): the DIAL chat client
    strips ``custom_content`` from any message-shaped object it finds while round-tripping
    state, which would drop Deep Research's own state and force a re-plan every turn.
    The DIAL messages are rebuilt from these fields only when calling the deployment.
    """

    user_message: str
    assistant_content: str
    # Deep Research's own ``custom_content.state`` from this turn, replayed so it can resume.
    deep_research_state: dict[str, Any] = Field(default_factory=dict)


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
    def from_state(cls, state: dict[str, Any]) -> Self | None:
        """Load and validate the session stored in DIAL state, or ``None`` if absent."""
        if raw := state.get(StateVarsConfig.DEEP_RESEARCH_SESSION):
            return cls.model_validate(raw)
        return None

    @staticmethod
    def drop_from_state(state: dict[str, Any]) -> None:
        """Remove the session from DIAL state, if present (idempotent)."""
        state.pop(StateVarsConfig.DEEP_RESEARCH_SESSION, None)
