class InvalidLLMStreamResponse(Exception):
    """The exception raised when LLM streams an invalid response."""


def format_exception_reason(exc: BaseException) -> str:
    """Render an exception as a concise human-readable reason string.

    Exception groups are unwrapped into their leaf exceptions.
    See openspec/specs/admin-background-jobs: "The failure reason names the real exception".
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(format_exception_reason(sub) for sub in exc.exceptions)
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
