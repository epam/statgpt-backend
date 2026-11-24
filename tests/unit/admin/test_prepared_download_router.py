from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from aidial_client import DialException
from aidial_client.types.metadata import FileItem
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.user import User, require_jwt_auth
from statgpt.admin.exception_handlers import register_exception_handlers
from statgpt.admin.routers import discovery_datasets as router
from statgpt.admin.services import prepared_downloads
from statgpt.admin.services.exceptions import (
    DiscoveryDatasetsNotInChannelError,
    DiscoveryExportError,
)
from statgpt.common import models, schemas
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect
from statgpt.common.utils.dial.attachments import AttachmentsStorage

_PREPARE = "/channels/7/discovery-datasets/exports"
_DIRECT = "/channels/7/discovery-datasets/export"


@dataclass
class PreparedApi:
    app: FastAPI
    client: TestClient
    session: MagicMock
    storage: MagicMock
    render: AsyncMock
    open_stream: AsyncMock
    close_stream: AsyncMock
    user: User = field(default_factory=lambda: User("admin-a", "admin-a", "Admin A"))
    records: dict[UUID, models.PreparedDownload] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    channel_exists: bool = True

    def download(self, download_id: str, channel_id: int = 7) -> Response:
        return self.client.get(
            f"/channels/{channel_id}/discovery-datasets/exports/{download_id}/download"
        )


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[PreparedApi]:
    app = FastAPI()
    channels = APIRouter(prefix="/channels")
    channels.include_router(router.channel_discovery_datasets_router)
    app.include_router(channels)
    register_exception_handlers(app)
    session = MagicMock(spec=AsyncSession)
    storage = MagicMock(spec=AttachmentsStorage)
    render = AsyncMock(return_value=b"prepared snapshot")
    close = AsyncMock()
    state = PreparedApi(
        app,
        TestClient(app, raise_server_exceptions=False),
        session,
        storage,
        render,
        AsyncMock(),
        close,
    )

    def scalar(query):
        if query.column_descriptions[0]["entity"] is models.Channel:
            return 7 if state.channel_exists else None
        params = query.compile().params
        record = state.records.get(params["id_1"])
        if (
            record is not None
            and record.channel_id == params["channel_id_1"]
            and record.owner_user_id == params["owner_user_id_1"]
        ):
            return record
        return None

    session.scalar.side_effect = scalar
    session.add.side_effect = lambda record: state.records.__setitem__(record.id, record)

    async def upload(name: str, media_type: str, content: bytes) -> FileItem:
        url = f"files/test-bucket/{name}"
        state.files[url] = content
        return FileItem(bucket="test-bucket", url=url, node_type="ITEM", resource_type="FILE")

    storage.put_file.side_effect = upload
    storage.delete_file.side_effect = lambda url: state.files.pop(url, None)

    @asynccontextmanager
    async def session_context():
        yield session

    @asynccontextmanager
    async def storage_context(*args, **kwargs):
        yield storage

    async def open_stream(base_url: str, api_key: str, url: str):
        content = state.files[url]

        async def chunks():
            yield content

        return chunks(), "ignored/storage-type", close

    state.open_stream.side_effect = open_stream
    monkeypatch.setattr(router, "get_session_context_manager", session_context)
    monkeypatch.setattr(router.DiscoveryDatasetService, "export_discovery_datasets", render)
    monkeypatch.setattr(prepared_downloads, "attachments_storage_factory", storage_context)
    monkeypatch.setattr(router, "open_file_stream", state.open_stream)
    app.dependency_overrides[require_jwt_auth] = lambda: state.user
    app.dependency_overrides[cancel_on_disconnect] = lambda: None
    app.dependency_overrides[models.get_session] = lambda: session
    yield state
    state.client.close()


@pytest.mark.parametrize("export_format", ["csv", "xlsx"])
def test_preparation_stores_complete_file_and_returns_only_descriptor(
    api: PreparedApi, export_format: str
) -> None:
    response = api.client.post(_PREPARE, params={"format": export_format})
    assert response.status_code == 201
    assert set(response.json()) == {"id", "expiresAt"}
    assert response.headers["cache-control"] == "no-store"
    record = api.records[UUID(response.json()["id"])]
    assert record.filename == f"discovery_datasets.{export_format}"
    assert (
        record.media_type
        == router._EXPORT_MEDIA_TYPES[schemas.DiscoveryExportFormat(export_format)]
    )
    assert api.files[record.file_path] == b"prepared snapshot"
    assert record.file_path.endswith(f"prepared-downloads/{record.id}.{export_format}")
    api.session.commit.assert_awaited_once()
    assert api.render.await_args is not None
    _, filters, selected_format, include_validation = api.render.await_args.args
    assert filters == schemas.DiscoveryExportFilters()
    assert selected_format == schemas.DiscoveryExportFormat(export_format)
    assert include_validation is False


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"agency": "NSO"}, schemas.DiscoveryExportFilters(agency="NSO")),
        (
            {"validation_status": "INVALID"},
            schemas.DiscoveryExportFilters(
                validation_status=schemas.DiscoveryValidationStatus.INVALID
            ),
        ),
        (
            {"indexing_status": "FAILED"},
            schemas.DiscoveryExportFilters(indexing_status=schemas.DiscoveryIndexingStatus.FAILED),
        ),
        ([("id", "5"), ("id", "77")], schemas.DiscoveryExportFilters(ids=[5, 77])),
    ],
)
def test_preparation_uses_existing_filters(
    api: PreparedApi, params, expected: schemas.DiscoveryExportFilters
) -> None:
    response = api.client.post(_PREPARE, params=params)
    assert response.status_code == 201
    assert api.render.await_args is not None
    assert api.render.await_args.args[1] == expected


def test_preparation_passes_workbook_validation_option(api: PreparedApi) -> None:
    response = api.client.post(
        _PREPARE,
        params={"format": "xlsx", "include_validation": "true", "validation_status": "INVALID"},
    )
    assert response.status_code == 201
    assert api.render.await_args is not None
    assert api.render.await_args.args[2:] == (schemas.DiscoveryExportFormat.XLSX, True)


def test_preparation_accepts_500_ids(api: PreparedApi) -> None:
    assert api.client.post(_PREPARE, params=[("id", str(i)) for i in range(500)]).status_code == 201


@pytest.mark.parametrize(
    "params",
    [
        [("id", "5"), ("agency", "NSO")],
        [("id", str(i)) for i in range(501)],
        {"format": "xls"},
        {"validation_status": "UNKNOWN"},
        {"include_validation": "not-a-bool"},
    ],
)
def test_preparation_preserves_query_errors(api: PreparedApi, params) -> None:
    direct = api.client.get(_DIRECT, params=params)
    prepared = api.client.post(_PREPARE, params=params)
    assert direct.status_code == prepared.status_code == 422
    assert direct.json() == prepared.json()
    api.render.assert_not_awaited()
    api.storage.put_file.assert_not_awaited()


@pytest.mark.parametrize(
    "error, status_code",
    [
        (DiscoveryExportError("Description exceeds the Excel cell limit."), 400),
        (DiscoveryDatasetsNotInChannelError(7, [77]), 404),
    ],
)
def test_rendering_errors_preserve_direct_detail_and_never_upload(
    api: PreparedApi, error: Exception, status_code: int
) -> None:
    api.render.side_effect = error
    direct = api.client.get(_DIRECT, params={"format": "xlsx"})
    prepared = api.client.post(_PREPARE, params={"format": "xlsx"})
    assert direct.status_code == prepared.status_code == status_code
    assert direct.json() == prepared.json() == {"detail": str(error)}
    api.storage.put_file.assert_not_awaited()
    assert not api.records


def test_unknown_channel_does_not_render_or_upload(api: PreparedApi) -> None:
    api.channel_exists = False
    response = api.client.post(_PREPARE)
    assert response.status_code == 404
    assert "id" not in response.json()
    api.render.assert_not_awaited()
    api.storage.put_file.assert_not_awaited()


def test_upload_failure_has_sanitized_502_detail(api: PreparedApi) -> None:
    api.storage.put_file.side_effect = RuntimeError("secret-key files/internal-bucket/private.csv")
    response = api.client.post(_PREPARE)
    assert response.status_code == 502
    assert "secret-key" not in response.text
    assert "internal-bucket" not in response.text
    assert "id" not in response.json()
    api.session.commit.assert_not_awaited()


@pytest.mark.parametrize("export_format", ["csv", "xlsx"])
def test_repeated_download_returns_original_bytes_and_stored_headers(
    api: PreparedApi, export_format: str
) -> None:
    descriptor = api.client.post(_PREPARE, params={"format": export_format}).json()
    record = api.records[UUID(descriptor["id"])]
    deadline = record.expires_at
    api.render.return_value = b"source changed after preparation"
    for _ in range(2):
        response = api.download(descriptor["id"])
        assert response.status_code == 200
        assert response.content == b"prepared snapshot"
        assert response.headers["content-type"].startswith(record.media_type)
        assert (
            response.headers["content-disposition"]
            == f'attachment; filename="discovery_datasets.{export_format}"'
        )
        assert response.headers["cache-control"] == "no-store"
        assert record.expires_at == deadline
    api.render.assert_awaited_once()
    assert api.close_stream.await_count == 2


@pytest.mark.parametrize("mismatch", ["owner", "channel", "deleted_channel", "missing_id"])
def test_scope_mismatch_is_404_without_file_io(api: PreparedApi, mismatch: str) -> None:
    descriptor = api.client.post(_PREPARE).json()
    record = api.records[UUID(descriptor["id"])]
    channel_id = 7
    if mismatch == "owner":
        api.user = User("admin-b", "admin-b", "Admin B")
    elif mismatch == "channel":
        channel_id = 8
    elif mismatch == "deleted_channel":
        record.channel_id = None
    else:
        descriptor["id"] = str(uuid4())
    record.expires_at = datetime.now(UTC) - timedelta(hours=1)
    response = api.download(descriptor["id"], channel_id)
    assert response.status_code == 404
    assert response.json() == {"detail": "Prepared download not found."}
    api.open_stream.assert_not_awaited()


@pytest.mark.parametrize("seconds_past", [0, 1])
def test_expiry_boundary_is_410_without_file_io(
    api: PreparedApi, monkeypatch: pytest.MonkeyPatch, seconds_past: int
) -> None:
    descriptor = api.client.post(_PREPARE).json()
    deadline = api.records[UUID(descriptor["id"])].expires_at

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return deadline + timedelta(seconds=seconds_past)

    monkeypatch.setattr(prepared_downloads, "datetime", Clock)
    response = api.download(descriptor["id"])
    assert response.status_code == 410
    assert response.json() == {"detail": "Prepared download has expired. Prepare a new export."}
    api.open_stream.assert_not_awaited()


def test_refreshed_identity_can_retrieve_without_repreparation(api: PreparedApi) -> None:
    descriptor = api.client.post(_PREPARE).json()
    api.user = User("admin-a", "new-login-name", "Updated Admin A")
    assert api.download(descriptor["id"]).status_code == 200
    api.render.assert_awaited_once()


def test_authentication_failure_precedes_lookup(api: PreparedApi) -> None:
    descriptor = api.client.post(_PREPARE).json()
    api.session.scalar.reset_mock()

    def reject():
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    api.app.dependency_overrides[require_jwt_auth] = reject
    assert api.download(descriptor["id"]).status_code == 401
    api.session.scalar.assert_not_awaited()
    api.open_stream.assert_not_awaited()


@pytest.mark.parametrize(
    "failure, expected_status",
    [
        (DialException("missing storage path", status_code=HTTPStatus.NOT_FOUND), 404),
        (
            DialException("secret storage credentials", status_code=HTTPStatus.SERVICE_UNAVAILABLE),
            502,
        ),
        (OSError("internal storage path"), 502),
    ],
)
def test_storage_opening_errors_are_mapped_without_internal_detail(
    api: PreparedApi, failure: Exception, expected_status: int
) -> None:
    descriptor = api.client.post(_PREPARE).json()
    api.open_stream.side_effect = failure
    response = api.download(descriptor["id"])
    assert response.status_code == expected_status
    assert str(failure) not in response.text
    api.close_stream.assert_not_awaited()


def test_invalid_download_uuid_uses_normal_validation(api: PreparedApi) -> None:
    assert api.download("not-a-uuid").status_code == 422
    api.open_stream.assert_not_awaited()


def test_openapi_declares_descriptor_file_types_and_errors(api: PreparedApi) -> None:
    schema = api.app.openapi()
    preparation = schema["paths"]["/channels/{channel_id}/discovery-datasets/exports"]["post"]
    download = schema["paths"][
        "/channels/{channel_id}/discovery-datasets/exports/{download_id}/download"
    ]["get"]
    assert "requestBody" not in preparation
    assert {"201", "400", "401", "403", "404", "422", "502"} <= preparation["responses"].keys()
    assert {"200", "401", "403", "404", "410", "422", "502"} <= download["responses"].keys()
    assert set(download["responses"]["200"]["content"]) == set(router._EXPORT_MEDIA_TYPES.values())
    descriptor_schema = schema["components"]["schemas"]["PreparedDownloadResponse"]
    assert set(descriptor_schema["properties"]) == {"id", "expiresAt"}
