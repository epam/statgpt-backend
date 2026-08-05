from typing import Any, Self

from pydantic import BaseModel, Field

from statgpt.app.config.state import StateVarsConfig

# User-facing message shown when a Deep Research turn fails. Shared by the tool (mid-stream /
# request errors) and the Supreme Agent (framework-level errors that never reach the tool) so
# both failure paths surface identical wording.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


class DeepResearchSession(BaseModel):
    """Supreme Agent <-> Deep Research conversation, persisted in DIAL state.

    Carries the multi-turn clarification flow across user turns. ``messages`` is the
    DIAL-format sub-conversation (user/assistant only) replayed to the Deep Research
    deployment so it can resume its preparation flow; each assistant entry carries Deep
    Research's own ``custom_content.state``. Kept separate from the user-facing chat
    history so Deep Research's intermediate turns do not pollute the Supreme Agent context.

    The session lives in state only while the clarification flow is in progress; once Deep
    Research delivers the final report the tool drops it, so its mere presence signals an
    in-progress run.
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)

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
