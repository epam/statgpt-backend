from typing import Any, Self

from pydantic import BaseModel, Field

from statgpt.app.config.state import StateVarsConfig


class DeepResearchTurn(BaseModel):
    """One Supreme Agent <-> Deep Research exchange, persisted in DIAL state.

    Not in DIAL-message shape on purpose (see openspec/specs/deep-research: "Session turns are not
    stored as messages").
    """

    user_message: str
    assistant_content: str
    # Deep Research's own ``custom_content.state`` from this turn, replayed so it can resume.
    deep_research_state: dict[str, Any] = Field(default_factory=dict)


class DeepResearchSession(BaseModel):
    """Supreme Agent <-> Deep Research conversation, kept in DIAL state while a run is ongoing."""

    turns: list[DeepResearchTurn] = Field(default_factory=list)

    @property
    def original_question(self) -> str | None:
        """The question the session was started with, or ``None`` before the first turn is
        recorded (a session is saved only once Deep Research has answered)."""
        return self.turns[0].user_message if self.turns else None

    @property
    def latest_query(self) -> str | None:
        """Deep Research's current research query: the ``preparation.current_query`` of the newest
        turn whose state carries one, which Deep Research rewrites as the session goes on. Falls
        back to the original question when no stored state carries a query."""
        for turn in reversed(self.turns):
            preparation = turn.deep_research_state.get('preparation')
            query = preparation.get('current_query') if isinstance(preparation, dict) else None
            if isinstance(query, str) and (query := query.strip()):
                return query
        return self.original_question

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
