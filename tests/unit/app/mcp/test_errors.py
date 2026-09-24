import logging

import httpx
import openai
import pytest
from aidial_sdk.exceptions import HTTPException as DIALException
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ValidationError

from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.app.mcp.errors import (
    McpErrorClass,
    build_tool_error,
    classify_exception,
    to_tool_error,
)
from statgpt.app.security.exceptions import InsufficientRoleError, MissingApiKeyError
from statgpt.app.utils.dial_exceptions import RateLimitException


def _validation_error() -> ValidationError:
    class _Args(BaseModel):
        limit: int

    with pytest.raises(ValidationError) as exc_info:
        _Args(limit="not-an-int")
    return exc_info.value


def _openai_error(cls, status: int, *, headers: dict | None = None, body=None) -> openai.APIError:
    request = httpx.Request("GET", "https://llm.internal.example/v1/chat")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("upstream message", response=response, body=body)


# --- One user-visible message per class (definition-of-done) -----------------------------


@pytest.mark.parametrize(
    ("error_class", "expected_snippet"),
    [
        (McpErrorClass.INVALID_INPUT, "Invalid input"),
        (McpErrorClass.MISSING_AUTHORIZATION, "Authentication required"),
        (McpErrorClass.INSUFFICIENT_SCOPE, "Access denied"),
        (McpErrorClass.RATE_LIMITED, "Rate limit reached"),
        (McpErrorClass.UPSTREAM_UNAVAILABLE, "Service temporarily unavailable"),
        (McpErrorClass.RESULT_TOO_LARGE, "Result too large"),
        (McpErrorClass.INTERNAL_ERROR, "Internal error"),
    ],
)
def test_each_class_has_actionable_message(error_class, expected_snippet):
    error = build_tool_error(error_class)

    assert isinstance(error, ToolError)
    message = str(error)
    assert expected_snippet in message
    # Every template ends with a concrete next step ("try again" / "call the tool again").
    assert "again" in message


def test_every_class_is_covered_by_a_template():
    for error_class in McpErrorClass:
        assert str(build_tool_error(error_class))


# --- Exception -> class classification ---------------------------------------------------


def test_validation_error_is_invalid_input():
    assert classify_exception(_validation_error()) is McpErrorClass.INVALID_INPUT


def test_upstream_error_is_upstream_unavailable():
    error = ToolUpstreamError("The SDMX backend did not respond in time (timeout).")
    assert classify_exception(error) is McpErrorClass.UPSTREAM_UNAVAILABLE


def test_openai_rate_limit_is_rate_limited():
    error = _openai_error(openai.RateLimitError, 429)
    assert classify_exception(error) is McpErrorClass.RATE_LIMITED


def test_dial_rate_limit_exception_is_rate_limited():
    # RateLimitException carries HTTP 500 (DIAL Core does not forward 429) yet is a rate limit.
    assert classify_exception(RateLimitException("slow down")) is McpErrorClass.RATE_LIMITED


def test_missing_api_key_is_missing_authorization():
    assert classify_exception(MissingApiKeyError()) is McpErrorClass.MISSING_AUTHORIZATION


def test_insufficient_role_is_insufficient_scope():
    assert classify_exception(InsufficientRoleError()) is McpErrorClass.INSUFFICIENT_SCOPE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, McpErrorClass.MISSING_AUTHORIZATION),
        (403, McpErrorClass.INSUFFICIENT_SCOPE),
        (429, McpErrorClass.RATE_LIMITED),
        (500, McpErrorClass.INTERNAL_ERROR),
    ],
)
def test_dial_exception_classified_by_status(status, expected):
    assert classify_exception(DIALException("boom", status_code=status)) is expected


def test_context_length_error_is_result_too_large():
    error = _openai_error(
        openai.BadRequestError,
        400,
        body={"code": "context_length_exceeded", "message": "maximum context length exceeded"},
    )
    assert classify_exception(error) is McpErrorClass.RESULT_TOO_LARGE


def test_plain_bad_request_is_internal_error():
    error = _openai_error(openai.BadRequestError, 400, body={"code": "invalid_value"})
    assert classify_exception(error) is McpErrorClass.INTERNAL_ERROR


def test_unknown_exception_is_internal_error():
    assert classify_exception(RuntimeError("boom")) is McpErrorClass.INTERNAL_ERROR


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
    error = to_tool_error(_validation_error(), tool_name="data_query")

    message = str(error)
    assert "Invalid input" in message
    assert "limit" in message


def test_upstream_unavailable_surfaces_safe_reason():
    reason = "The SDMX backend did not respond in time (timeout)."
    error = to_tool_error(ToolUpstreamError(reason), tool_name="sdmx_query_app")

    message = str(error)
    assert "Service temporarily unavailable" in message
    assert reason in message


def test_rate_limited_includes_retry_after_from_headers():
    error = to_tool_error(
        _openai_error(openai.RateLimitError, 429, headers={"retry-after": "30"}),
        tool_name="data_query",
    )

    assert "30 seconds" in str(error)


def test_rate_limited_without_retry_after_is_still_actionable():
    error = build_tool_error(McpErrorClass.RATE_LIMITED)

    assert "Wait a moment and try again" in str(error)


def test_dial_rate_limit_surfaces_retry_after_from_extra_fields():
    # DIAL's RateLimitException stores the upstream Retry-After in extra_fields (it is not
    # forwarded as a response header); it must still reach the caller-visible message.
    error = to_tool_error(RateLimitException("slow down", retry_after="30"), tool_name="data_query")

    assert "30 seconds" in str(error)


def test_to_tool_error_demotes_framework_relog_to_debug():
    # to_tool_error logs the full error itself, so the ToolError it returns carries a DEBUG
    # log level to keep FastMCP's own re-log of it from duplicating that line at high severity.
    error = to_tool_error(RuntimeError("boom"), tool_name="data_query")

    assert error.log_level == logging.DEBUG
