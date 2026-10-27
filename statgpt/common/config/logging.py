import logging
import logging.config
import re
import sys

import uvicorn.logging
from aidial_sdk import logger as aidial_logger
from opentelemetry import trace

from statgpt.common.settings.logging import LoggingSettings


class SingleLineFormatter(uvicorn.logging.DefaultFormatter):
    def format(self, record):
        res = super().format(record).replace("\n", r"\n")
        return res


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return not re.search(r"(\s+)/health(\s+)", record.getMessage())


class RedactingFilter(logging.Filter):
    """Scrub secrets and PII from every log record before it is written.

    This is a last-resort boundary: if a payload is ever logged by accident — or by a
    third-party library, or inside an exception traceback — bearer tokens, JWTs,
    inline base64 images and email addresses are replaced before the record reaches a
    handler, so raw secrets/PII are never persisted.
    """

    _PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), "Bearer <redacted>"),
        (
            re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
            "<redacted-jwt>",
        ),
        (re.compile(r"(data:image/\w+;base64,)[^\s\"']+"), r"\1<base64_image>"),
        (
            re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            "<redacted-email>",
        ),
    )

    @classmethod
    def _redact(cls, text: str) -> str:
        for pattern, replacement in cls._PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            # A %-format/args mismatch makes getMessage() raise. Filters run before
            # emit()'s guarded block, so raising here would crash the log caller instead
            # of hitting logging's handleError(). Fail open and let emit() handle it.
            return True
        redacted = self._redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        if record.exc_info:
            # Render the traceback now and scrub it, so secrets/PII carried by an
            # exception's string representation are not emitted verbatim.
            record.exc_text = self._redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


class TraceIdFilter(logging.Filter):
    """Stamp every log record with the trace id of the request it was logged in.

    DIAL Core propagates its trace to the application in the W3C ``traceparent`` header (it does
    not forward ``X-DIAL-TRACE-ID``, which is a response header only), and the DIAL SDK's FastAPI
    instrumentation makes that trace the current span. The id logged here is therefore the one
    Core returns to its caller as ``X-DIAL-TRACE-ID``. Records logged outside a request get ``-``.
    """

    NO_TRACE_ID = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        span_context = trace.get_current_span().get_span_context()
        record.trace_id = (
            format(span_context.trace_id, "032x") if span_context.is_valid else self.NO_TRACE_ID
        )
        return True


_redaction_filter = RedactingFilter()
_trace_id_filter = TraceIdFilter()


class LoggingConfig:

    LOGGING_SETTINGS = LoggingSettings()

    @classmethod
    def configure_logging(cls):
        # Making the uvicorn and dial_sdk loggers delegate its logging to the root logger.
        # The uvicorn CLI gives `uvicorn.access` a non-propagating handler of its own before the
        # app is imported, so without this the access lines would skip the filters attached below.
        for logger in [
            logging.getLogger("uvicorn"),
            logging.getLogger("uvicorn.access"),
            aidial_logger,
        ]:
            logger.handlers = []
            logger.propagate = True

        # Setting up log levels
        for name in ["statgpt", "statgpt-ml", "__main__"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level)

        for name in ["statgpt.common.models.database"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_db)

        for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_uvicorn)

        # Filter out health check requests from uvicorn logs
        logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

        for name in ["httpcore", "httpx"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_httpcore)

        for name in ["openai"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_openai)

        # Configuring the root logger
        root = logging.getLogger()

        root_has_stderr_handler = any(
            isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr
            for handler in root.handlers
        )

        if not root_has_stderr_handler:
            formatter = uvicorn.logging.DefaultFormatter(
                fmt=cls.LOGGING_SETTINGS.format,
                datefmt=cls.LOGGING_SETTINGS.date_format,
                use_colors=True,
            )

            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            root.addHandler(handler)

        # configure statgpt-ml logger for multiline mode
        if not cls.LOGGING_SETTINGS.multiline_mode_enabled:
            statgpt_ml_logger = logging.getLogger("statgpt-ml")
            single_line_formatter = SingleLineFormatter(
                fmt=cls.LOGGING_SETTINGS.format,
                datefmt=cls.LOGGING_SETTINGS.date_format,
                use_colors=True,
            )
            console_single_line_handler = logging.StreamHandler()
            console_single_line_handler.setFormatter(single_line_formatter)
            statgpt_ml_logger.handlers = [console_single_line_handler]
            statgpt_ml_logger.propagate = False

        # Attach the filters at the write boundary. Every record reaching these handlers is
        # scrubbed of secrets/PII and carries the `trace_id` the log format references.
        # statgpt-ml uses its own handlers with propagation disabled, so it must be covered
        # explicitly.
        for handler in [*root.handlers, *logging.getLogger("statgpt-ml").handlers]:
            handler.addFilter(_redaction_filter)
            handler.addFilter(_trace_id_filter)


LoggingConfig.configure_logging()
logger = logging.getLogger("statgpt")
multiline_logger = logging.getLogger("statgpt-ml")
