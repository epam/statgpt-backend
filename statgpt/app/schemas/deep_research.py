from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field

from statgpt.app.config.state import StateVarsConfig

# User-facing message shown when a Deep Research turn fails. Shared by the tool (mid-stream /
# request errors) and the Supreme Agent (framework-level errors that never reach the tool) so
# both failure paths surface identical wording.
DEEP_RESEARCH_ERROR_MESSAGE = "\n\nDeep Research could not complete this request. Please try again."


class DeepResearchStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class DeepResearchSession(BaseModel):
    """Supreme Agent <-> Deep Research conversation, persisted in DIAL state.

    Mediates the multi-turn clarification flow. `messages` is the DIAL-format
    sub-conversation (user/assistant only) replayed to the Deep Research
    deployment so it can resume its preparation flow; each assistant entry
    carries Deep Research's own `custom_content.state`. `status` drives how the
    Supreme Agent routes the next user message.
    """

    status: DeepResearchStatus = DeepResearchStatus.IN_PROGRESS
    messages: list[dict[str, Any]] = Field(default_factory=list)
    outstanding_questions: list[str] = Field(default_factory=list)
    # Answers the agent derived from conversation context for clarifying questions the user was
    # NOT asked. Held here until the user answers the remaining (verbatim) questions, then merged
    # with the user's reply and forwarded to Deep Research so it receives every answer at once.
    pending_auto_answers: list[str] = Field(default_factory=list)
    # LLM summary of the conversation, generated once when the session starts and forwarded to
    # Deep Research as context. Reused for question triage on later turns to avoid re-summarizing.
    context_summary: str | None = None

    @property
    def is_in_progress(self) -> bool:
        return self.status == DeepResearchStatus.IN_PROGRESS

    @classmethod
    def from_state(cls, state: dict) -> Self | None:
        """Load and validate the session stored in DIAL state, or ``None`` if absent."""
        if raw := state.get(StateVarsConfig.DEEP_RESEARCH_SESSION):
            return cls.model_validate(raw)
        return None
