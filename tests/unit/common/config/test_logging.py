"""Unit tests for the logging redaction boundary and the trace id stamping."""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn.logging
from opentelemetry import context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from statgpt.common.config.logging import LoggingConfig, RedactingFilter, TraceIdFilter
from statgpt.common.settings.logging import LoggingSettings

# What DIAL Core sends upstream: its trace id travels in `traceparent`, and it is the same id Core
# returns to its caller as `X-DIAL-TRACE-ID`.
_DIAL_TRACE_ID = "588a018d4087ae2597f0597919ba2236"
_DIAL_TRACEPARENT = f"00-{_DIAL_TRACE_ID}-a58a212979b5b5ca-03"


def _make_record(msg: str, args: tuple = (), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


class TestRedactingFilter:
    """Tests for ``RedactingFilter`` — the last-resort scrubbing at the log boundary."""

    def test_bearer_token_is_redacted(self) -> None:
        record = _make_record("auth header: Bearer abc123.def-456_XYZ")
        RedactingFilter().filter(record)
        assert record.getMessage() == "auth header: Bearer <redacted>"

    def test_jwt_is_redacted(self) -> None:
        record = _make_record("token=eyJhbGciOi.eyJzdWIiOiI.SflKxwRJ tail")
        RedactingFilter().filter(record)
        assert record.getMessage() == "token=<redacted-jwt> tail"

    def test_base64_image_is_redacted(self) -> None:
        record = _make_record("img data:image/png;base64,AAAABBBBCCCCDDDD end")
        RedactingFilter().filter(record)
        assert record.getMessage() == "img data:image/png;base64,<base64_image> end"

    def test_email_is_redacted(self) -> None:
        record = _make_record("contact john.doe@example.com now")
        RedactingFilter().filter(record)
        assert record.getMessage() == "contact <redacted-email> now"

    def test_redaction_applies_after_args_interpolation(self) -> None:
        record = _make_record("user said %s", args=("Bearer secrettoken123",))
        RedactingFilter().filter(record)
        assert record.getMessage() == "user said Bearer <redacted>"
        # args are consumed so the redacted message is not re-interpolated downstream.
        assert record.args == ()

    def test_clean_message_is_untouched(self) -> None:
        record = _make_record("processed 5 items in 12ms")
        RedactingFilter().filter(record)
        assert record.getMessage() == "processed 5 items in 12ms"
        assert record.args == ()

    def test_exception_traceback_is_scrubbed(self) -> None:
        try:
            raise ValueError("leaked Bearer abc.def.ghi in error")
        except ValueError:
            exc_info = sys.exc_info()
        record = _make_record("tool failed", exc_info=exc_info)

        RedactingFilter().filter(record)

        assert record.exc_info is None
        assert record.exc_text is not None
        assert "abc.def.ghi" not in record.exc_text
        assert "Bearer <redacted>" in record.exc_text

    def test_filter_never_drops_records(self) -> None:
        record = _make_record("anything")
        assert RedactingFilter().filter(record) is True

    def test_malformed_format_record_does_not_raise(self) -> None:
        # A %-format/args mismatch makes getMessage() raise. The filter must fail open —
        # return True and leave the record untouched — so the log caller is not crashed.
        record = _make_record("count=%d", args=("not-a-number",))
        assert RedactingFilter().filter(record) is True
        assert record.msg == "count=%d"
        assert record.args == ("not-a-number",)


@contextmanager
def _in_request_traced_by_dial_core() -> Iterator[None]:
    """Make DIAL Core's propagated trace current, as the SDK's FastAPI instrumentation does for an
    incoming request."""
    extracted = TraceContextTextMapPropagator().extract({"traceparent": _DIAL_TRACEPARENT})
    token = context.attach(extracted)
    try:
        yield
    finally:
        context.detach(token)


class TestTraceIdFilter:
    """Tests for ``TraceIdFilter`` — the trace id that correlates our logs with DIAL Core."""

    def test_trace_id_is_the_one_dial_core_propagates(self) -> None:
        record = _make_record("tools/call")
        with _in_request_traced_by_dial_core():
            TraceIdFilter().filter(record)
        assert record.trace_id == _DIAL_TRACE_ID

    def test_placeholder_outside_a_request(self) -> None:
        record = _make_record("startup")
        TraceIdFilter().filter(record)
        assert record.trace_id == TraceIdFilter.NO_TRACE_ID

    def test_filter_never_drops_records(self) -> None:
        assert TraceIdFilter().filter(_make_record("anything")) is True

    def test_default_format_renders_the_trace_id(self) -> None:
        # The default format references `trace_id`; the filter must supply it or formatting fails.
        # The field default is read directly so a local LOG_FORMAT override cannot mask it.
        default_format = LoggingSettings.model_fields["format"].default
        formatter = uvicorn.logging.DefaultFormatter(fmt=default_format, use_colors=False)
        record = _make_record("tools/call")
        with _in_request_traced_by_dial_core():
            TraceIdFilter().filter(record)
        assert f"| {_DIAL_TRACE_ID} | test | tools/call" in formatter.format(record)


class TestLoggingConfig:
    """Tests for how ``LoggingConfig`` wires the filters into the logging tree."""

    def test_every_configured_handler_carries_the_filters(self) -> None:
        # A handler whose format references `trace_id` but lacks the filter loses every line to a
        # "Formatting field not found" error instead of crashing, so a dropped filter goes unnoticed.
        # statgpt-ml has its own handler and doesn't propagate, so it must be covered too.
        configured_handlers = [
            handler
            for handler in [
                *logging.getLogger().handlers,
                *logging.getLogger("statgpt-ml").handlers,
            ]
            if isinstance(handler.formatter, uvicorn.logging.DefaultFormatter)
        ]
        assert configured_handlers
        for handler in configured_handlers:
            filter_types = {type(f) for f in handler.filters}
            assert {RedactingFilter, TraceIdFilter} <= filter_types

    def test_uvicorn_access_lines_reach_the_root_handlers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The uvicorn CLI configures `uvicorn.access` before the app is imported: a handler of its
        # own, not propagating, so access lines would bypass the root handlers and their filters.
        access_logger = logging.getLogger("uvicorn.access")
        monkeypatch.setattr(access_logger, "handlers", [logging.StreamHandler()])
        monkeypatch.setattr(access_logger, "propagate", False)
        # Reconfiguring also re-adds the health check filter and replaces the statgpt-ml handler.
        monkeypatch.setattr(access_logger, "filters", list(access_logger.filters))
        statgpt_ml_logger = logging.getLogger("statgpt-ml")
        monkeypatch.setattr(statgpt_ml_logger, "handlers", statgpt_ml_logger.handlers)
        root = logging.getLogger()
        root_handlers = list(root.handlers)
        try:
            LoggingConfig.configure_logging()
        finally:
            for handler in set(root.handlers) - set(root_handlers):
                root.removeHandler(handler)

        assert access_logger.handlers == []
        assert access_logger.propagate is True
