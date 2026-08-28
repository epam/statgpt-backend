"""The edge mapping from discovery domain errors to HTTP responses.

Worth testing directly: the services raise domain errors and never set a status code, so
these handlers are the only thing that decides what a caller sees. The 400 body in
particular is a contract the Admin Portal reads.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from statgpt.admin.exception_handlers import register_exception_handlers
from statgpt.admin.services.exceptions import (
    AdminServiceError,
    DiscoveryDatasetConflictError,
    DiscoveryDatasetNotFoundError,
    DiscoveryIndexingJobNotFoundError,
    DiscoveryPayloadError,
    DiscoveryRagNotConfiguredError,
    DiscoveryUploadFormatError,
    DiscoveryUploadTooLargeError,
    IndexingJobInProgressError,
)
from statgpt.common import schemas


def _client(error: AdminServiceError) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def _boom() -> None:
        raise error

    return TestClient(app, raise_server_exceptions=False)


def _problem(**kwargs: object) -> schemas.DiscoveryPayloadProblem:
    return schemas.DiscoveryPayloadProblem.model_validate(kwargs)


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (DiscoveryDatasetNotFoundError(7), 404),
        (DiscoveryIndexingJobNotFoundError(7), 404),
        (DiscoveryDatasetConflictError("already exists"), 409),
        (IndexingJobInProgressError(channel_id=1, job_id=2), 409),
        (IndexingJobInProgressError(channel_id=1), 409),
        (DiscoveryRagNotConfiguredError(channel_id=1), 409),
        (DiscoveryUploadTooLargeError("too big"), 413),
        (DiscoveryUploadFormatError("unreadable"), 400),
        (DiscoveryPayloadError(problems=[]), 400),
    ],
)
def test_each_domain_error_maps_to_its_status(
    error: AdminServiceError, expected_status: int
) -> None:
    response = _client(error).get("/boom")

    assert response.status_code == expected_status


def test_a_payload_error_reports_every_problem_at_its_cell() -> None:
    """The per-cell report is the point of refusing the write; it has to survive to the wire."""
    error = DiscoveryPayloadError(
        problems=[
            _problem(
                message="Agency / organization must not be empty.", field="agency", cell="D14"
            ),
            _problem(message="Dataset ID is missing.", field="dataset_id", row=15),
        ],
        truncated=True,
    )

    body = _client(error).get("/boom").json()

    detail = body["detail"]
    assert detail["truncated"] is True
    assert "2 problems" in detail["message"]
    # camelCase, like every other response the Admin Portal consumes.
    assert [(p["field"], p.get("cell"), p.get("row")) for p in detail["problems"]] == [
        ("agency", "D14", None),
        ("dataset_id", None, 15),
    ]


def test_an_unreadable_file_shares_the_payload_error_shape() -> None:
    """One 400 shape across the discovery endpoints, so the declared model is honest."""
    body = _client(DiscoveryUploadFormatError("The sheet is empty.")).get("/boom").json()

    assert body == {
        "detail": {"message": "The sheet is empty.", "problems": [], "truncated": False}
    }


def test_an_unconfigured_channel_says_what_to_configure() -> None:
    """The 409 has to be actionable: an administrator sees only this message."""
    body = _client(DiscoveryRagNotConfiguredError(channel_id=3)).get("/boom").json()

    assert "discoveryDatasets.details.applicationId" in body["detail"]


def test_a_conflict_names_the_colliding_record() -> None:
    detail = (
        "A discovery dataset with agency='imf', dataset id='dot' already exists in this channel."
    )

    body = _client(DiscoveryDatasetConflictError(detail)).get("/boom").json()

    assert body == {"detail": detail}
