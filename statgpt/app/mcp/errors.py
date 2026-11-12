"""Actionable error taxonomy for MCP tool calls.

Translates the heterogeneous exceptions raised while executing a StatGPT tool into a
small, fixed set of error classes, each with a single user-visible message template.
The templates state what happened, whether the caller or the end user can fix it, and
the concrete next step. Internal detail (stack traces, upstream bodies, hostnames,
trace/request identifiers) never reaches the caller: it is logged server-side only.
"""

import logging
import math
from enum import StrEnum

import httpx
import openai
from aidial_sdk.exceptions import HTTPException as DIALException
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.app.security.exceptions import AuthenticationError, AuthorizationError
from statgpt.app.utils.dial_exceptions import RateLimitException, format_duration

_log = logging.getLogger(__name__)

# Cap how many invalid fields are named back to the caller so a request with many bad
# arguments still yields a short, readable message.
_MAX_VALIDATION_FIELDS = 5

# OpenAI error codes that mean the request/result exceeded the model's context window.
_CONTEXT_LENGTH_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})


class McpErrorClass(StrEnum):
    """The error classes surfaced to an MCP caller."""

    INVALID_INPUT = "invalid_input"
    MISSING_AUTHORIZATION = "missing_authorization"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    RESULT_TOO_LARGE = "result_too_large"
    INTERNAL_ERROR = "internal_error"


# One message template per class: what happened, then the concrete next step. A caller-safe
# detail, when there is one, goes between the two. ``{when}`` in RATE_LIMITED is filled by
# ``_build_tool_error``.
_MESSAGES: dict[McpErrorClass, tuple[str, str]] = {
    McpErrorClass.INVALID_INPUT: (
        "Invalid input: the request arguments were not accepted.",
        "Correct the arguments and call the tool again.",
    ),
    McpErrorClass.MISSING_AUTHORIZATION: (
        "Authentication required: this tool needs a valid access token.",
        "Sign in (or supply a valid token) and try again.",
    ),
    McpErrorClass.INSUFFICIENT_SCOPE: (
        "Access denied: your account is not permitted to use this tool.",
        "Request the required access from your administrator, then try again.",
    ),
    McpErrorClass.RATE_LIMITED: (
        "Rate limit reached: too many requests were made in a short period.",
        "Wait {when} and try again.",
    ),
    McpErrorClass.UPSTREAM_UNAVAILABLE: (
        "Service temporarily unavailable: a required data service could not be reached.",
        "Try again in a few moments.",
    ),
    McpErrorClass.RESULT_TOO_LARGE: (
        "Result too large: the query returned more data than can be sent back.",
        "Narrow the request (add filters, a shorter time range, or fewer items) and try again.",
    ),
    McpErrorClass.INTERNAL_ERROR: (
        "Internal error: the tool could not complete the request.",
        "The issue has been logged; try again, and contact support if it persists.",
    ),
}

# Server-side log severity per class: caller-fixable problems are low signal, genuine
# server faults get a stack trace.
_LOG_LEVELS: dict[McpErrorClass, int] = {
    McpErrorClass.INVALID_INPUT: logging.DEBUG,
    McpErrorClass.MISSING_AUTHORIZATION: logging.INFO,
    McpErrorClass.INSUFFICIENT_SCOPE: logging.INFO,
    McpErrorClass.RATE_LIMITED: logging.WARNING,
    McpErrorClass.UPSTREAM_UNAVAILABLE: logging.WARNING,
    McpErrorClass.RESULT_TOO_LARGE: logging.INFO,
    McpErrorClass.INTERNAL_ERROR: logging.ERROR,
}

# HTTP statuses that map to a class of their own, for DIAL exceptions and upstream HTTP errors.
_STATUS_CLASSES: dict[int, McpErrorClass] = {
    401: McpErrorClass.MISSING_AUTHORIZATION,
    403: McpErrorClass.INSUFFICIENT_SCOPE,
    429: McpErrorClass.RATE_LIMITED,
}


def _is_context_length_error(exc: openai.BadRequestError) -> bool:
    if getattr(exc, "code", None) in _CONTEXT_LENGTH_CODES:
        return True
    text = str(exc).lower()
    return "context length" in text or "maximum context" in text


def _classify_upstream_status(status_code: int) -> McpErrorClass:
    """Classify an error response from an upstream service (DIAL, an LLM, a data backend)."""
    if status_code >= 500:
        return McpErrorClass.UPSTREAM_UNAVAILABLE
    return _STATUS_CLASSES.get(status_code, McpErrorClass.INTERNAL_ERROR)


def _classify_exception(exc: BaseException) -> McpErrorClass:
    """Map an execution error to its taxonomy class.

    Specific StatGPT/library exceptions take precedence; ``DIALException``s and upstream HTTP
    errors are then classified by status, and connection failures and timeouts mean the
    upstream is unavailable. Anything unrecognised is an internal error. That includes a
    pydantic ``ValidationError``: raised during execution it means an internal model rejected
    some data (e.g. an upstream payload), not that the caller's arguments were wrong. The
    argument-validation site passes ``INVALID_INPUT`` to ``to_tool_error`` explicitly.
    """
    if isinstance(exc, ToolUpstreamError):
        return McpErrorClass.UPSTREAM_UNAVAILABLE
    if isinstance(exc, (openai.RateLimitError, RateLimitException)):
        return McpErrorClass.RATE_LIMITED
    if isinstance(exc, openai.BadRequestError) and _is_context_length_error(exc):
        return McpErrorClass.RESULT_TOO_LARGE
    if isinstance(exc, AuthenticationError):
        return McpErrorClass.MISSING_AUTHORIZATION
    if isinstance(exc, AuthorizationError):
        return McpErrorClass.INSUFFICIENT_SCOPE
    if isinstance(exc, DIALException):
        # Raised by StatGPT itself, so a 5xx is our own fault rather than an upstream outage.
        return _STATUS_CLASSES.get(exc.status_code, McpErrorClass.INTERNAL_ERROR)
    if isinstance(exc, openai.APIStatusError):
        return _classify_upstream_status(exc.status_code)
    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_upstream_status(exc.response.status_code)
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError, TimeoutError)):
        # Includes the openai/httpx timeouts, and a downstream TimeoutError that is not the
        # tool's own deadline.
        return McpErrorClass.UPSTREAM_UNAVAILABLE
    return McpErrorClass.INTERNAL_ERROR


def _retry_after_seconds(exc: BaseException) -> int | None:
    value: str | int | None = None
    if isinstance(exc, (openai.APIStatusError, httpx.HTTPStatusError)):
        # httpx headers are case-insensitive.
        value = exc.response.headers.get("retry-after")
    elif isinstance(exc, DIALException):
        if exc.headers:
            value = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
        if value is None:
            # DIAL's RateLimitException carries the upstream Retry-After in extra_fields
            # rather than as a forwarded response header.
            value = exc.extra_fields.get("retry_after")
    if value is None:
        return None
    try:
        # Delay-seconds may be fractional; round up so the caller never retries too early.
        # An HTTP-date value is not parsed: the message falls back to "a moment".
        seconds = math.ceil(float(value))
    except (ValueError, TypeError, OverflowError):
        return None
    return seconds if seconds > 0 else None


def _summarize_validation_error(exc: ValidationError) -> str:
    """Name the offending fields (location + reason) without echoing their values."""
    parts: list[str] = []
    for err in exc.errors()[:_MAX_VALIDATION_FIELDS]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _caller_safe_detail(error_class: McpErrorClass, exc: BaseException) -> str | None:
    """The detail beyond the class template that is safe to show the caller, if any."""
    if error_class is McpErrorClass.INVALID_INPUT and isinstance(exc, ValidationError):
        return _summarize_validation_error(exc)
    if isinstance(exc, ToolUpstreamError):
        # Its message is written to be shown to the caller.
        return str(exc)
    if isinstance(exc, DIALException) and not isinstance(exc, RateLimitException):
        # DIAL's curated, user-facing text (e.g. which deployment access is missing). A
        # RateLimitException's only restates the retry hint the template already renders.
        return exc.display_message
    return None


def _as_sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _build_tool_error(
    error_class: McpErrorClass,
    *,
    detail: str | None = None,
    retry_after_seconds: int | None = None,
) -> ToolError:
    """Build the ``ToolError`` for a class from its template.

    ``detail`` must already be safe to show the caller (see ``_caller_safe_detail``); it is
    placed between what happened and the next step.
    """
    summary, next_step = _MESSAGES[error_class]
    if error_class is McpErrorClass.RATE_LIMITED:
        when = format_duration(retry_after_seconds) if retry_after_seconds else "a moment"
        next_step = next_step.format(when=when)
    parts = [summary, _as_sentence(detail) if detail else None, next_step]
    # ``to_tool_error`` has already logged the full error at its real severity; keep FastMCP's
    # own re-log of the raised ToolError (see fastmcp server.py) quiet to avoid a duplicate.
    return ToolError(" ".join(p for p in parts if p), log_level=logging.DEBUG)


def to_tool_error(
    exc: BaseException, *, tool_name: str, error_class: McpErrorClass | None = None
) -> ToolError:
    """Translate an execution error into a scrubbed, actionable MCP ``ToolError``.

    Logs the full exception server-side at a severity matching its class, then returns
    a ``ToolError`` whose text is a fixed template — no stack trace, upstream body,
    hostname or identifier leaks into what the caller sees. The only caller-visible
    detail beyond the template is the named invalid field (invalid input), a
    safe-by-design upstream reason (``ToolUpstreamError``) or DIAL's user-facing
    ``display_message``.

    ``error_class`` overrides the classification of ``exc``. The argument-validation site
    passes ``INVALID_INPUT``: only there does a ``ValidationError`` mean the caller's
    arguments were rejected.
    """
    if error_class is None:
        error_class = _classify_exception(exc)
    level = _LOG_LEVELS[error_class]
    _log.log(
        level,
        "MCP tool %s failed [%s]: %s",
        tool_name,
        error_class,
        exc,
        exc_info=(level >= logging.ERROR),
    )
    retry_after_seconds = (
        _retry_after_seconds(exc) if error_class is McpErrorClass.RATE_LIMITED else None
    )
    return _build_tool_error(
        error_class,
        detail=_caller_safe_detail(error_class, exc),
        retry_after_seconds=retry_after_seconds,
    )
