from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem

_llm_call_duration_context_var: ContextVar[LLMCallDurationManager | None] = ContextVar(
    "llm_call_duration_context", default=None
)


class LLMCallDurationManager:
    def __init__(self) -> None:
        self._durations: dict[str, LLMCallDurationItem] = {}

    def add_duration(self, item: LLMCallDurationItem) -> None:
        if item.id not in self._durations:
            self._durations[item.id] = item
        else:
            self._durations[item.id] += item

    def get_durations(self) -> list[LLMCallDurationItem]:
        return list(self._durations.values())

    @property
    def total_duration_s(self) -> float:
        return sum(item.duration_s for item in self._durations.values())


@contextmanager
def llm_call_duration_context(
    enabled: bool,
) -> Generator[LLMCallDurationManager | None, None, None]:
    manager = LLMCallDurationManager() if enabled else None
    token = _llm_call_duration_context_var.set(manager)
    try:
        yield token.var.get()
    finally:
        _llm_call_duration_context_var.reset(token)


def get_llm_call_duration_manager() -> LLMCallDurationManager | None:
    return _llm_call_duration_context_var.get()
