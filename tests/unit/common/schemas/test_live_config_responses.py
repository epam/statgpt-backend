import datetime
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from statgpt.common.schemas import ChannelDatasetExpanded, DataSet, DataSource, DataSourceType
from statgpt.common.schemas.base import ListResponse
from statgpt.common.schemas.dataset import Status
from statgpt.common.schemas.enums import PreprocessingStatusEnum


@pytest.mark.parametrize(
    "details",
    [
        {"sdmxConfig": {"id": "test", "name": "Test", "url": "https://example.invalid"}},
        {"locale": "en", "proxyConfig": {"agencies": []}},
    ],
)
def test_data_source_http_responses_preserve_stored_keys_and_enrichment(details):
    now = datetime.datetime(2026, 1, 1)
    source = DataSource(
        id=1,
        created_at=now,
        updated_at=now,
        title="Test",
        type_id=1,
        details=details,
        type=DataSourceType(id=1, created_at=now, updated_at=now, name="SDMX21", description=""),
    )
    app = FastAPI()

    @app.get("/source", response_model=DataSource)
    def single():
        return source

    @app.get("/sources", response_model=ListResponse[DataSource])
    def listed():
        return ListResponse[DataSource](data=[source], limit=10, offset=0, count=1, total=1)

    with TestClient(app) as client:
        response = client.get("/source").json()
        listing = client.get("/sources").json()
    assert response["details"] == details
    assert listing["data"][0] == response
    assert DataSource.model_validate(response).model_dump(mode="json")["details"] == details
    assert response["description"] == ""


@pytest.mark.parametrize(
    "overrides", [{}, {"isOfficial": False, "citation": None, "pinnedColumns": []}]
)
def test_dataset_and_nested_channel_dataset_http_responses_preserve_stored_keys(overrides):
    now = datetime.datetime(2026, 1, 1)
    details = {"urn": {"agencyId": "TEST", "resourceId": "DATA"}, **overrides}
    dataset = DataSet(
        id=1,
        id_=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        title="Test",
        data_source_id=1,
        data_source=None,
        details=details,
        status=Status(status="offline"),
    )
    nested = ChannelDatasetExpanded(
        id=2,
        created_at=now,
        updated_at=now,
        channel_id=1,
        dataset_id=1,
        dataset=dataset,
        preprocessing_status=PreprocessingStatusEnum.NOT_STARTED,
        clearing_status=PreprocessingStatusEnum.NOT_STARTED,
        last_completed_version=None,
        previous_completed_version=None,
        latest_version=None,
        last_auto_update_job=None,
    )
    app = FastAPI()

    @app.get("/dataset", response_model=DataSet)
    def single():
        return dataset

    @app.get("/channel-dataset", response_model=ChannelDatasetExpanded)
    def expanded():
        return nested

    with TestClient(app) as client:
        response = client.get("/dataset").json()
        expanded_response = client.get("/channel-dataset").json()
    assert response["details"] == details
    assert expanded_response["dataset"] == response
    assert DataSet.model_validate(response).model_dump(mode="json")["details"] == details
    assert response["description"] == ""
    assert response["status"]["details"] == ""
