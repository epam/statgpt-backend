"""The request rules of `GET /channels/{channel_id}/discovery-datasets/export`.

The service is replaced: these tests are about what the route accepts and how it labels the
file, not about its contents.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.routers.discovery_datasets import channel_discovery_datasets_router
from statgpt.admin.services import AdminPortalDiscoveryDatasetService
from statgpt.common import models, schemas
from statgpt.common.utils import MediaTypes
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

_URL = "/channels/7/discovery-datasets/export"


@pytest.fixture
def export(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=b"content")
    monkeypatch.setattr(AdminPortalDiscoveryDatasetService, "export_discovery_datasets", mock)
    return mock


@pytest.fixture
def client(export: AsyncMock) -> TestClient:
    app = FastAPI()
    channels = APIRouter(prefix="/channels")
    channels.include_router(channel_discovery_datasets_router)
    app.include_router(channels)
    app.dependency_overrides[require_jwt_auth] = lambda: None
    app.dependency_overrides[cancel_on_disconnect] = lambda: None
    app.dependency_overrides[models.get_session] = lambda: None
    return TestClient(app)


def _call_args(export: AsyncMock) -> tuple[Any, ...]:
    return export.await_args.args  # type: ignore[union-attr]


def test_the_defaults_are_todays_csv(client: TestClient, export: AsyncMock) -> None:
    response = client.get(_URL)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(MediaTypes.CSV)
    assert 'filename="discovery_datasets.csv"' in response.headers["content-disposition"]
    channel_id, filters, export_format, include_validation = _call_args(export)
    assert channel_id == 7
    assert filters == schemas.DiscoveryExportFilters()
    assert export_format is schemas.DiscoveryExportFormat.CSV
    assert include_validation is False


def test_a_workbook_is_labelled_as_one(client: TestClient, export: AsyncMock) -> None:
    response = client.get(_URL, params={"format": "xlsx", "include_validation": "true"})

    assert response.status_code == 200
    assert response.headers["content-type"] == MediaTypes.XLSX
    assert 'filename="discovery_datasets.xlsx"' in response.headers["content-disposition"]
    _, _, export_format, include_validation = _call_args(export)
    assert export_format is schemas.DiscoveryExportFormat.XLSX
    assert include_validation is True


def test_ids_are_repeatable(client: TestClient, export: AsyncMock) -> None:
    response = client.get(_URL, params=[("id", "3"), ("id", "5")])

    assert response.status_code == 200
    assert _call_args(export)[1].ids == [3, 5]


@pytest.mark.parametrize(
    "other",
    [("agency", "IMF"), ("validation_status", "INVALID"), ("indexing_status", "FAILED")],
)
def test_ids_cannot_be_combined_with_the_filters(
    client: TestClient, export: AsyncMock, other: tuple[str, str]
) -> None:
    response = client.get(_URL, params=[("id", "3"), other])

    assert response.status_code == 422
    assert "cannot be combined" in response.json()["detail"]
    export.assert_not_awaited()


def test_at_most_500_ids(client: TestClient, export: AsyncMock) -> None:
    response = client.get(_URL, params=[("id", str(i)) for i in range(501)])

    assert response.status_code == 422
    export.assert_not_awaited()


def test_an_unknown_format_is_refused(client: TestClient, export: AsyncMock) -> None:
    assert client.get(_URL, params={"format": "xls"}).status_code == 422
