"""The catch-all handler that logs unexpected errors while the request's trace is current.

uvicorn logs the traceback of an error that escapes the app only after the request's span has
ended, so that line has no trace id. The admin backend is reached directly, without DIAL Core,
so the trace id in its logs and audit records is the only way to tie a failure to a request.
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from statgpt.admin.exception_handlers import register_exception_handlers
from statgpt.common.config.logging import TraceIdFilter

_LOGGER_NAME = "statgpt.admin.exception_handlers"


def _app(trace_ids_seen_by_route: list[str]) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def _boom() -> None:
        span_context = trace.get_current_span().get_span_context()
        trace_ids_seen_by_route.append(format(span_context.trace_id, "032x"))
        raise RuntimeError("boom")

    return app


def test_an_unexpected_error_keeps_the_default_500(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
        response = TestClient(_app([]), raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    [record] = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert "GET /boom" in record.getMessage()
    # The text, not `exc_info`: the redaction filter renders the traceback and drops `exc_info`.
    assert "RuntimeError: boom" in caplog.text


def test_the_traceback_carries_the_trace_id_of_the_failed_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # As in the admin app: the instrumentation starts a trace per request, no DIAL Core involved.
    trace_ids_seen_by_route: list[str] = []
    app = _app(trace_ids_seen_by_route)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=TracerProvider())
    caplog.handler.addFilter(TraceIdFilter())
    try:
        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            TestClient(app, raise_server_exceptions=False).get("/boom")
    finally:
        FastAPIInstrumentor.uninstrument_app(app)

    [record] = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert trace_ids_seen_by_route
    assert record.trace_id == trace_ids_seen_by_route[0] != TraceIdFilter.NO_TRACE_ID
