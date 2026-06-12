class InvalidLLMStreamResponse(Exception):
    """The exception raised when LLM streams an invalid response."""


def format_exception_reason(exc: BaseException) -> str:
    """Render an exception as a concise human-readable reason string.

    ``asyncio.TaskGroup()`` raises a ``BaseExceptionGroup`` whose ``str()`` is the
    generic "unhandled errors in task group (N sub-exceptions)" message, hiding the
    real cause. This unwraps groups recursively and joins the leaf exceptions as
    "<ClassName>: <message>".
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(format_exception_reason(sub) for sub in exc.exceptions)
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
