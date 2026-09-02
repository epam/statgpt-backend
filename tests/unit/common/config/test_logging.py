"""Unit tests for the logging redaction boundary."""

import logging
import sys

from statgpt.common.config.logging import RedactingFilter


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
