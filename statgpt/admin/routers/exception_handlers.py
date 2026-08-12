"""Mapping of discovery domain errors to HTTP responses.

Inner layers raise domain errors; they become status codes here, at the edge. A router
cannot carry exception handlers, and repeating try/except in every handler is how one gets
missed and leaks as a 500.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from statgpt.admin.services.exceptions import (
    DiscoveryPayloadError,
    DiscoveryUploadFormatError,
    DiscoveryUploadTooLargeError,
    IndexingJobInProgressError,
)

_log = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DiscoveryPayloadError)
    async def _payload_error(_: Request, exc: DiscoveryPayloadError) -> JSONResponse:
        """A structurally unusable payload: 400, with one entry per offending record."""
        _log.info(f"Rejected discovery payload: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": exc.detail.model_dump(mode="json", by_alias=True)},
        )

    @app.exception_handler(DiscoveryUploadFormatError)
    async def _format_error(_: Request, exc: DiscoveryUploadFormatError) -> JSONResponse:
        _log.info(f"Rejected discovery upload: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(DiscoveryUploadTooLargeError)
    async def _too_large(_: Request, exc: DiscoveryUploadTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": str(exc)}
        )

    @app.exception_handler(IndexingJobInProgressError)
    async def _job_in_progress(_: Request, exc: IndexingJobInProgressError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})
