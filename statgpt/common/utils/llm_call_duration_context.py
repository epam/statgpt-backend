from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem

_llm_call_duration_context_var: ContextVar = ContextVar("llm_call_duration_context")


class LLMCallDurationManager:
    def __init__(self):
        self._durations = {}

    def add_duration(self, item: LLMCallDurationItem):
        if item.id not in self._durations:
            self._durations[item.id] = item
        else:
            self._durations[item.id] += item

    def get_durations(self) -> list[LLMCallDurationItem]:
        return list(self._durations.values())


@contextmanager
def llm_call_duration_context() -> Generator[LLMCallDurationManager, None, None]:
    token = _llm_call_duration_context_var.set(LLMCallDurationManager())
    try:
        yield token.var.get()
    finally:
        _llm_call_duration_context_var.reset(token)


def get_llm_call_duration_manager() -> LLMCallDurationManager:
    return _llm_call_duration_context_var.get()
