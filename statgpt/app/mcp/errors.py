"""Actionable error taxonomy for MCP tool calls.

Translates the heterogeneous exceptions raised while executing a StatGPT tool into a
small, fixed set of error classes, each with a single user-visible message template.
The templates state what happened, whether the caller or the end user can fix it, and
the concrete next step. Internal detail (stack traces, upstream bodies, hostnames,
trace/request identifiers) never reaches the caller: it is logged server-side only.
"""

import logging
from enum import Enum

import openai
from aidial_sdk.exceptions import HTTPException as DIALException
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.app.security.exceptions import AuthenticationError, AuthorizationError
from statgpt.app.utils.dial_exceptions import RateLimitException

_log = logging.getLogger(__name__)

# Cap how many invalid fields are named back to the caller so a request with many bad
# arguments still yields a short, readable message.
_MAX_VALIDATION_FIELDS = 5

# OpenAI error codes that mean the request/result exceeded the model's context window.
_CONTEXT_LENGTH_CODES = {"context_length_exceeded", "string_above_max_length"}


class McpErrorClass(str, Enum):
    """The error classes surfaced to an MCP caller."""

    INVALID_INPUT = "invalid_input"
    MISSING_AUTHORIZATION = "missing_authorization"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    RESULT_TOO_LARGE = "result_too_large"
    INTERNAL_ERROR = "internal_error"


# One message template per class. ``{when}`` in RATE_LIMITED is filled by ``build_tool_error``.
_MESSAGES: dict[McpErrorClass, str] = {
    McpErrorClass.INVALID_INPUT: (
        "Invalid input: the request arguments were not accepted. "
        "Correct the arguments and call the tool again."
    ),
    McpErrorClass.MISSING_AUTHORIZATION: (
        "Authentication required: this tool needs a valid access token. "
        "Sign in (or supply a valid token) and try again."
    ),
    McpErrorClass.INSUFFICIENT_SCOPE: (
        "Access denied: your account is not permitted to use this tool. "
        "Request the required access from your administrator, then try again."
    ),
    McpErrorClass.RATE_LIMITED: (
        "Rate limit reached: too many requests were made in a short period. "
        "Wait {when} and try again."
    ),
    McpErrorClass.UPSTREAM_UNAVAILABLE: (
        "Service temporarily unavailable: a required data service could not be reached. "
        "Try again in a few moments."
    ),
    McpErrorClass.RESULT_TOO_LARGE: (
        "Result too large: the query returned more data than can be sent back. "
        "Narrow the request (add filters, a shorter time range, or fewer items) and try again."
    ),
    McpErrorClass.INTERNAL_ERROR: (
        "Internal error: the tool could not complete the request. "
        "The issue has been logged; try again, and contact support if it persists."
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


def _is_context_length_error(exc: openai.BadRequestError) -> bool:
    if getattr(exc, "code", None) in _CONTEXT_LENGTH_CODES:
        return True
    text = str(exc).lower()
    return "context length" in text or "maximum context" in text


def _classify_dial_exception(exc: DIALException) -> McpErrorClass:
    return {
        401: McpErrorClass.MISSING_AUTHORIZATION,
        403: McpErrorClass.INSUFFICIENT_SCOPE,
        429: McpErrorClass.RATE_LIMITED,
    }.get(exc.status_code, McpErrorClass.INTERNAL_ERROR)


def classify_exception(exc: BaseException) -> McpErrorClass:
    """Map an execution error to its taxonomy class.

    Specific StatGPT/library exceptions take precedence; the remaining ``DIALException``s
    are classified by HTTP status. Anything unrecognised is an internal error.
    """
    if isinstance(exc, ValidationError):
        return McpErrorClass.INVALID_INPUT
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
        return _classify_dial_exception(exc)
    return McpErrorClass.INTERNAL_ERROR


def _retry_after_seconds(exc: BaseException) -> int | None:
    value: str | int | None = None
    if isinstance(exc, openai.APIStatusError):
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
    elif isinstance(exc, DIALException):
        if exc.headers:
            value = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
        if value is None:
            # DIAL's RateLimitException carries the upstream Retry-After in extra_fields
            # rather than as a forwarded response header.
            value = exc.extra_fields.get("retry_after")
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _summarize_validation_error(exc: ValidationError) -> str:
    """Name the offending fields (location + reason) without echoing their values."""
    parts = []
    for err in exc.errors()[:_MAX_VALIDATION_FIELDS]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def build_tool_error(
    error_class: McpErrorClass,
    *,
    detail: str | None = None,
    retry_after_seconds: int | None = None,
    log_level: int | None = None,
) -> ToolError:
    """Build the ``ToolError`` for a class from its template.

    ``detail`` must already be safe to show the caller (e.g. a ``ToolUpstreamError``
    message crafted for that purpose); it is appended in parentheses.

    ``log_level`` sets the severity FastMCP uses when it re-logs the raised ``ToolError``.
    It defaults to the class severity, so a direct caller gets one log line at the right
    level; ``to_tool_error`` overrides it (having already logged the full detail itself).
    """
    message = _MESSAGES[error_class]
    if error_class is McpErrorClass.RATE_LIMITED:
        when = f"{retry_after_seconds} seconds" if retry_after_seconds else "a moment"
        message = message.format(when=when)
    if detail:
        message = f"{message} ({detail})"
    level = log_level if log_level is not None else _LOG_LEVELS[error_class]
    return ToolError(message, log_level=level)


def to_tool_error(exc: BaseException, *, tool_name: str) -> ToolError:
    """Translate an execution error into a scrubbed, actionable MCP ``ToolError``.

    Logs the full exception server-side at a severity matching its class, then returns
    a ``ToolError`` whose text is a fixed template — no stack trace, upstream body,
    hostname or identifier leaks into what the caller sees. The only caller-visible
    detail beyond the template is the named invalid field (invalid input) or a
    safe-by-design upstream reason (upstream unavailable).
    """
    error_class = classify_exception(exc)
    level = _LOG_LEVELS[error_class]
    _log.log(
        level,
        "MCP tool %s failed [%s]: %s",
        tool_name,
        error_class.value,
        exc,
        exc_info=(level >= logging.ERROR),
    )
    detail: str | None = None
    if error_class is McpErrorClass.INVALID_INPUT and isinstance(exc, ValidationError):
        detail = _summarize_validation_error(exc)
    elif error_class is McpErrorClass.UPSTREAM_UNAVAILABLE:
        detail = str(exc)
    return build_tool_error(
        error_class,
        detail=detail,
        retry_after_seconds=_retry_after_seconds(exc),
        # The full error was just logged above at its real severity; keep FastMCP's own
        # re-log of the raised ToolError (see fastmcp server.py) quiet to avoid a duplicate.
        log_level=logging.DEBUG,
    )
