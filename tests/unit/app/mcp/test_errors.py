import logging
from typing import Any

import httpx
import openai
import pytest
from aidial_sdk.exceptions import HTTPException as DIALException
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ValidationError, field_validator

from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.app.mcp.errors import (
    McpErrorClass,
    _build_tool_error,
    _classify_exception,
    to_tool_error,
)
from statgpt.app.security.exceptions import InsufficientRoleError, MissingApiKeyError
from statgpt.app.utils.dial_exceptions import ForbiddenDeploymentException, RateLimitException

_LLM_URL = "https://llm.internal.example/v1/chat"


def _validation_error() -> ValidationError:
    class _Args(BaseModel):
        limit: int

    with pytest.raises(ValidationError) as exc_info:
        _Args(limit="not-an-int")
    return exc_info.value


def _openai_error(
    cls: type[openai.APIStatusError],
    status: int,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> openai.APIStatusError:
    request = httpx.Request("GET", _LLM_URL)
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("upstream message", response=response, body=body)


def _httpx_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", _LLM_URL)
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("upstream failed", request=request, response=response)


# --- One user-visible message per class (definition-of-done) -----------------------------

_EXPECTED_SNIPPETS: dict[McpErrorClass, str] = {
    McpErrorClass.INVALID_INPUT: "Invalid input",
    McpErrorClass.MISSING_AUTHORIZATION: "Authentication required",
    McpErrorClass.INSUFFICIENT_SCOPE: "Access denied",
    McpErrorClass.RATE_LIMITED: "Rate limit reached",
    McpErrorClass.UPSTREAM_UNAVAILABLE: "Service temporarily unavailable",
    McpErrorClass.RESULT_TOO_LARGE: "Result too large",
    McpErrorClass.INTERNAL_ERROR: "Internal error",
}


@pytest.mark.parametrize("error_class", list(McpErrorClass))
def test_each_class_has_actionable_message(error_class: McpErrorClass):
    error = _build_tool_error(error_class)

    assert isinstance(error, ToolError)
    message = str(error)
    assert message.startswith(_EXPECTED_SNIPPETS[error_class])
    # Every template ends with a concrete next step ("try again" / "call the tool again").
    assert "again" in message


def test_detail_sits_between_what_happened_and_the_next_step():
    error = _build_tool_error(
        McpErrorClass.UPSTREAM_UNAVAILABLE, detail="The SDMX backend did not respond in time."
    )

    assert str(error) == (
        "Service temporarily unavailable: a required data service could not be reached. "
        "The SDMX backend did not respond in time. "
        "Try again in a few moments."
    )


# --- Exception -> class classification ---------------------------------------------------


def test_validation_error_during_execution_is_internal_error():
    # Only the argument-validation site reports INVALID_INPUT (passed explicitly); a pydantic
    # error raised while executing means an internal model rejected some data.
    assert _classify_exception(_validation_error()) is McpErrorClass.INTERNAL_ERROR


def test_upstream_error_is_upstream_unavailable():
    error = ToolUpstreamError("The SDMX backend did not respond in time.")
    assert _classify_exception(error) is McpErrorClass.UPSTREAM_UNAVAILABLE


def test_openai_rate_limit_is_rate_limited():
    error = _openai_error(openai.RateLimitError, 429)
    assert _classify_exception(error) is McpErrorClass.RATE_LIMITED


def test_dial_rate_limit_exception_is_rate_limited():
    # RateLimitException carries HTTP 500 (DIAL Core does not forward 429) yet is a rate limit.
    assert _classify_exception(RateLimitException("slow down")) is McpErrorClass.RATE_LIMITED


def test_missing_api_key_is_missing_authorization():
    assert _classify_exception(MissingApiKeyError()) is McpErrorClass.MISSING_AUTHORIZATION


def test_insufficient_role_is_insufficient_scope():
    assert _classify_exception(InsufficientRoleError()) is McpErrorClass.INSUFFICIENT_SCOPE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, McpErrorClass.MISSING_AUTHORIZATION),
        (403, McpErrorClass.INSUFFICIENT_SCOPE),
        (429, McpErrorClass.RATE_LIMITED),
        # Raised by StatGPT itself: a 5xx is our own fault, not an upstream outage.
        (500, McpErrorClass.INTERNAL_ERROR),
    ],
)
def test_dial_exception_classified_by_status(status: int, expected: McpErrorClass):
    assert _classify_exception(DIALException("boom", status_code=status)) is expected


@pytest.mark.parametrize(
    ("cls", "status", "expected"),
    [
        (openai.AuthenticationError, 401, McpErrorClass.MISSING_AUTHORIZATION),
        (openai.PermissionDeniedError, 403, McpErrorClass.INSUFFICIENT_SCOPE),
        (openai.InternalServerError, 502, McpErrorClass.UPSTREAM_UNAVAILABLE),
        (openai.InternalServerError, 503, McpErrorClass.UPSTREAM_UNAVAILABLE),
        (openai.NotFoundError, 404, McpErrorClass.INTERNAL_ERROR),
    ],
)
def test_openai_status_error_classified_by_status(
    cls: type[openai.APIStatusError], status: int, expected: McpErrorClass
):
    # In the MCP path the user's token is forwarded to DIAL for LLM calls, so an expired token
    # or a missing deployment grant surfaces as an openai error, not a DIALException.
    assert _classify_exception(_openai_error(cls, status)) is expected


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=httpx.Request("GET", _LLM_URL)),
        openai.APITimeoutError(request=httpx.Request("GET", _LLM_URL)),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        TimeoutError("downstream timed out"),
    ],
)
def test_connection_failures_and_timeouts_are_upstream_unavailable(error: BaseException):
    assert _classify_exception(error) is McpErrorClass.UPSTREAM_UNAVAILABLE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, McpErrorClass.MISSING_AUTHORIZATION),
        (429, McpErrorClass.RATE_LIMITED),
        (504, McpErrorClass.UPSTREAM_UNAVAILABLE),
        (400, McpErrorClass.INTERNAL_ERROR),
    ],
)
def test_httpx_status_error_classified_by_status(status: int, expected: McpErrorClass):
    assert _classify_exception(_httpx_status_error(status)) is expected


def test_context_length_error_is_result_too_large():
    error = _openai_error(
        openai.BadRequestError,
        400,
        body={"code": "context_length_exceeded", "message": "maximum context length exceeded"},
    )
    assert _classify_exception(error) is McpErrorClass.RESULT_TOO_LARGE


def test_plain_bad_request_is_internal_error():
    error = _openai_error(openai.BadRequestError, 400, body={"code": "invalid_value"})
    assert _classify_exception(error) is McpErrorClass.INTERNAL_ERROR


def test_unknown_exception_is_internal_error():
    assert _classify_exception(RuntimeError("boom")) is McpErrorClass.INTERNAL_ERROR


# --- to_tool_error scrubs internals and stays actionable ---------------------------------


def test_internal_error_scrubs_internals():
    leaky = RuntimeError(
        "connection to internal-db.prod.svc:5432 failed; trace_id=abc123; SELECT * FROM users"
    )
    error = to_tool_error(leaky, tool_name="data_query")

    message = str(error)
    assert message == (
        "Internal error: the tool could not complete the request. "
        "The issue has been logged; try again, and contact support if it persists."
    )
    for secret in ("internal-db", "5432", "trace_id", "SELECT", "abc123"):
        assert secret not in message


def test_invalid_input_names_the_offending_field():
    error = to_tool_error(
        _validation_error(), tool_name="data_query", error_class=McpErrorClass.INVALID_INPUT
    )

    message = str(error)
    assert message.startswith("Invalid input")
    assert "limit" in message
    assert message.endswith("Correct the arguments and call the tool again.")


def test_validation_error_during_execution_does_not_leak_model_internals():
    class _UpstreamPayload(BaseModel):
        series_key: str

        @field_validator("series_key")
        @classmethod
        def _check(cls, value: str) -> str:
            raise ValueError(f"unexpected key {value!r} from internal-sdmx.svc")

    with pytest.raises(ValidationError) as exc_info:
        _UpstreamPayload(series_key="A.B.C")

    message = str(to_tool_error(exc_info.value, tool_name="data_query"))

    assert message.startswith("Internal error")
    for internal in ("series_key", "internal-sdmx.svc", "A.B.C"):
        assert internal not in message


def test_upstream_unavailable_surfaces_safe_reason():
    reason = "The SDMX backend did not respond in time."
    error = to_tool_error(ToolUpstreamError(reason), tool_name="sdmx_query_app")

    message = str(error)
    assert message.startswith("Service temporarily unavailable")
    assert reason in message


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(
            message="Connection to llm.internal.example failed",
            request=httpx.Request("GET", _LLM_URL),
        ),
        httpx.ConnectError(f"failed to connect to {_LLM_URL}"),
    ],
)
def test_library_upstream_errors_do_not_surface_their_message(error: BaseException):
    # Only a ToolUpstreamError's message is written for the caller; a library error's can name
    # the internal endpoint.
    message = str(to_tool_error(error, tool_name="data_query"))

    assert message.startswith("Service temporarily unavailable")
    assert "llm.internal.example" not in message


def test_expired_token_on_llm_call_asks_to_sign_in():
    error = to_tool_error(_openai_error(openai.AuthenticationError, 401), tool_name="data_query")

    assert str(error).startswith("Authentication required")


def test_rate_limited_includes_retry_after_from_headers():
    error = to_tool_error(
        _openai_error(openai.RateLimitError, 429, headers={"Retry-After": "30"}),
        tool_name="data_query",
    )

    assert "Wait 30 seconds and try again." in str(error)


def test_rate_limited_humanises_a_long_retry_after():
    error = to_tool_error(
        _openai_error(openai.RateLimitError, 429, headers={"retry-after": "7500"}),
        tool_name="data_query",
    )

    assert "Wait 2 hours 5 minutes and try again." in str(error)


def test_rate_limited_rounds_a_fractional_retry_after_up():
    error = to_tool_error(
        _openai_error(openai.RateLimitError, 429, headers={"retry-after": "1.5"}),
        tool_name="data_query",
    )

    assert "Wait 2 seconds and try again." in str(error)


def test_rate_limited_without_retry_after_is_still_actionable():
    error = _build_tool_error(McpErrorClass.RATE_LIMITED)

    assert "Wait a moment and try again" in str(error)


def test_dial_rate_limit_surfaces_retry_after_from_extra_fields():
    # DIAL's RateLimitException stores the upstream Retry-After in extra_fields (it is not
    # forwarded as a response header); it must still reach the caller-visible message.
    error = to_tool_error(RateLimitException("slow down", retry_after="30"), tool_name="data_query")

    assert "30 seconds" in str(error)


def test_dial_display_message_is_surfaced():
    # DIAL's display_message is curated, user-facing text; it tells the user what access to ask for.
    exc = ForbiddenDeploymentException(
        "forbidden",
        display_message="You don't have access to the underlying deployment 'gpt-4o'.",
    )

    message = str(to_tool_error(exc, tool_name="data_query"))

    assert message.startswith("Access denied")
    assert "You don't have access to the underlying deployment 'gpt-4o'." in message


def test_to_tool_error_demotes_framework_relog_to_debug():
    # to_tool_error logs the full error itself, so the ToolError it returns carries a DEBUG
    # log level to keep FastMCP's own re-log of it from duplicating that line at high severity.
    error = to_tool_error(RuntimeError("boom"), tool_name="data_query")

    assert error.log_level == logging.DEBUG
