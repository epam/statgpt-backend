"""Mapping of discovery domain errors to HTTP responses.

Inner layers raise domain errors; they become status codes here, at the edge. A router
cannot carry exception handlers, and repeating try/except in every handler is how one gets
missed and leaks as a 500.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from statgpt.admin.services.exceptions import (
    DiscoveryDatasetConflictError,
    DiscoveryNotFoundError,
    DiscoveryPayloadError,
    DiscoveryRagNotConfiguredError,
    DiscoveryUploadFormatError,
    DiscoveryUploadTooLargeError,
    IndexingJobInProgressError,
)
from statgpt.common import schemas
from statgpt.common.services import GenericRagIngestionError

_log = logging.getLogger(__name__)


def _payload_error_response(detail: schemas.DiscoveryPayloadErrorDetail) -> JSONResponse:
    """Render the one 400 body shape the discovery endpoints use.

    A file that could not be read and a file whose cells are wrong are the same thing to a
    caller - the write was refused - so they share a shape and differ only in whether
    `problems` is populated. One shape is also what makes the 400 declarable in OpenAPI.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=schemas.DiscoveryPayloadErrorResponse(detail=detail).model_dump(
            mode="json", by_alias=True
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DiscoveryPayloadError)
    async def _payload_error(_: Request, exc: DiscoveryPayloadError) -> JSONResponse:
        """A structurally unusable payload: 400, with one entry per offending record."""
        _log.info(f"Rejected discovery payload: {exc}")
        return _payload_error_response(exc.detail)

    @app.exception_handler(DiscoveryUploadFormatError)
    async def _format_error(_: Request, exc: DiscoveryUploadFormatError) -> JSONResponse:
        """An unreadable file: 400, with nothing to itemize."""
        _log.info(f"Rejected discovery upload: {exc}")
        return _payload_error_response(
            schemas.DiscoveryPayloadErrorDetail(message=str(exc), problems=[])
        )

    @app.exception_handler(DiscoveryUploadTooLargeError)
    async def _too_large(_: Request, exc: DiscoveryUploadTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, content={"detail": str(exc)}
        )

    @app.exception_handler(DiscoveryNotFoundError)
    async def _not_found(_: Request, exc: DiscoveryNotFoundError) -> JSONResponse:
        """Registered on the base class, so every discovery id lookup maps to 404."""
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(DiscoveryDatasetConflictError)
    async def _conflict(_: Request, exc: DiscoveryDatasetConflictError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(IndexingJobInProgressError)
    async def _job_in_progress(_: Request, exc: IndexingJobInProgressError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(DiscoveryRagNotConfiguredError)
    async def _rag_not_configured(_: Request, exc: DiscoveryRagNotConfiguredError) -> JSONResponse:
        """A channel with nowhere to publish to: 409, naming what to configure.

        A conflict rather than a 400, for the same reason a job already in progress is: the
        request is well-formed, the channel is just not in a state that can serve it.
        """
        _log.info(f"Refused a discovery indexing job: {exc}")
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(GenericRagIngestionError)
    async def _rag_unreachable(_: Request, exc: GenericRagIngestionError) -> JSONResponse:
        """The RAG channel would not serve a call this request depends on: 502.

        Reachable since deleting a record withdraws its document synchronously. Everywhere
        else this error is raised inside an indexing job, which records it on the job row and
        never lets it near a response - so without this it would surface as a bare 500 and a
        caller could not tell a RAG channel that is down from a bug here.
        """
        _log.warning(f"Generic RAG call failed while serving a request: {exc}")
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})
