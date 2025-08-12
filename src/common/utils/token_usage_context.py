from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from common.schemas.token_usage import TokenUsageItem

_token_usage_context_var: ContextVar = ContextVar("token_usage_context")


class TokenUsageManager:
    def __init__(self):
        self._usage = {}

    def add_usage(self, item: TokenUsageItem):
        if item.id not in self._usage:
            self._usage[item.id] = item
        else:
            self._usage[item.id] += item

    def get_usage(self) -> list[TokenUsageItem]:
        return list(self._usage.values())


@contextmanager
def token_usage_context() -> Generator[TokenUsageManager, None, None]:
    token_usage = _token_usage_context_var.set(TokenUsageManager())
    try:
        yield token_usage.var.get()
    finally:
        _token_usage_context_var.reset(token_usage)


def get_token_usage_manager() -> TokenUsageManager:
    return _token_usage_context_var.get()
