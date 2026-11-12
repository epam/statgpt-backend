"""The request shape of `POST /channels/{channel_id}/discovery-datasets/upload`.

The service is replaced: these tests are about which files reach it, not what it does with them.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.testclient import TestClient

from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.routers.discovery_datasets import channel_discovery_datasets_router
from statgpt.admin.services import AdminPortalDiscoveryDatasetService
from statgpt.common import models, schemas

_URL = "/channels/7/discovery-datasets/upload"


async def _summarize(
    channel_id: int, files: list[UploadFile], mode: schemas.RecordUploadMode
) -> schemas.DiscoveryUploadSummary:
    names = [file.filename for file in files]
    return schemas.DiscoveryUploadSummary(
        created=len(names),
        rows_read=len(names),
        files=[schemas.DiscoveryUploadFileSummary(file=name, rows_read=1) for name in names],
    )


@pytest.fixture
def upload(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(side_effect=_summarize)
    monkeypatch.setattr(AdminPortalDiscoveryDatasetService, "upload", mock)
    return mock


@pytest.fixture
def app(upload: AsyncMock) -> FastAPI:
    app = FastAPI()
    channels = APIRouter(prefix="/channels")
    channels.include_router(channel_discovery_datasets_router)
    app.include_router(channels)
    app.dependency_overrides[require_jwt_auth] = lambda: None
    app.dependency_overrides[models.get_session] = lambda: None
    return app


def test_a_single_file_part_keeps_the_old_counters(app: FastAPI, upload: AsyncMock) -> None:
    response = TestClient(app).post(_URL, files={"file": ("a.xlsx", b"x")})

    assert response.status_code == 200
    body = response.json()
    assert (body["created"], body["updated"], body["unchanged"], body["deleted"]) == (1, 0, 0, 0)
    assert body["rowsRead"] == 1
    assert upload.await_args.args[2] is schemas.RecordUploadMode.UPSERT


def test_repeated_file_parts_reach_the_service_in_order(app: FastAPI) -> None:
    response = TestClient(app).post(
        _URL,
        params={"mode": "replace"},
        files=[("file", ("a.xlsx", b"x")), ("file", ("b.csv", b"y"))],
    )

    assert response.status_code == 200
    assert [entry["file"] for entry in response.json()["files"]] == ["a.xlsx", "b.csv"]


def test_the_openapi_schema_declares_an_array_of_files(app: FastAPI) -> None:
    """OpenAPI 3.1 spells a binary as `contentMediaType`, where 3.0 used `format: binary`."""
    body = app.openapi()["paths"]["/channels/{channel_id}/discovery-datasets/upload"]["post"]
    ref = body["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    form = app.openapi()["components"]["schemas"][ref.rsplit("/", 1)[-1]]

    file_schema = form["properties"]["file"]
    assert file_schema["type"] == "array"
    assert file_schema["items"] == {
        "type": "string",
        "contentMediaType": "application/octet-stream",
    }
