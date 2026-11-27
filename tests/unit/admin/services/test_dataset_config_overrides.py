import datetime
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from statgpt.admin.services.dataset import AdminPortalDataSetService
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.sdmx.common.config import SdmxDataSetConfig
from statgpt.common.schemas.dataset import Status

_DETAILS = {
    "urn": {"agencyId": "TEST", "resourceId": "DATA"},
    "dimensions": {
        "INDICATOR": {"dimensionType": "INDICATOR"},
        "TIME_PERIOD": {"dimensionType": "TIME_PERIOD"},
        "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
    },
}


@pytest.fixture
def service() -> AdminPortalDataSetService:
    session = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    service = AdminPortalDataSetService(session)
    handler = MagicMock()
    handler.parse_data_set_config = SdmxDataSetConfig.model_validate
    handler.get_dataset = AsyncMock(
        return_value=SimpleNamespace(name="Test", description="", status=Status(status="offline"))
    )
    service._get_handler = AsyncMock(return_value=handler)
    service._propagate_config_to_channel_datasets = AsyncMock(return_value=[])

    def add(item):
        item.id = 1
        item.created_at = datetime.datetime(2026, 1, 1)
        item.updated_at = item.created_at

    session.add.side_effect = add
    return service


def _stored_dataset(details: dict[str, Any]) -> models.DataSet:
    now = datetime.datetime(2026, 1, 1)
    return models.DataSet(
        id=1,
        id_=uuid.uuid4(),
        source_id=3,
        title="Test",
        details=details,
        created_at=now,
        updated_at=now,
    )


def _dataset_schema(item: models.DataSet, details: dict[str, Any]) -> schemas.DataSet:
    return schemas.DataSet(
        id=item.id,
        id_=item.id_,
        created_at=item.created_at,
        updated_at=item.updated_at,
        title=item.title,
        data_source_id=item.source_id,
        data_source=None,
        details=details,
        status=Status(status="offline"),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"isOfficial": False, "citation": None, "pinnedColumns": []},
        {"indexer": {"indicator": {}}},
    ],
)
async def test_create_stores_sparse_config_and_generated_identity(service, overrides):
    details = {**_DETAILS, **overrides}
    request = schemas.DataSetBase(title="Test", data_source_id=3, details=details)
    auth = MagicMock(spec=AuthContext)

    result = await AdminPortalDataSetService.create_dataset.__wrapped__(service, request, auth)

    stored = service._session.add.call_args.args[0]
    assert stored.details == details
    assert stored.id_ == request.id_ == result.id_
    assert result.details == details
    assert "id_" not in request.model_fields_set
    assert SdmxDataSetConfig.model_validate(stored.details).urn.version == "latest"
    service._get_handler.return_value.get_dataset.assert_awaited_once_with(
        entity_id=request.id_, title="Test", config=details, auth_context=auth, allow_offline=True
    )


async def test_replacement_removes_overrides_without_changing_identity(service, mocker):
    stored = _stored_dataset({**_DETAILS, "isOfficial": True, "pinnedColumns": ["INDICATOR"]})
    identity = stored.id_
    service.get_model_by_id = AsyncMock(return_value=stored)
    mocker.patch(
        "statgpt.admin.services.dataset.DataSetSerializer.db_to_schema",
        return_value=_dataset_schema(stored, _DETAILS),
    )
    auth = MagicMock(spec=AuthContext)

    result = await AdminPortalDataSetService.update.__wrapped__(
        service, stored.id, schemas.DataSetUpdateRequest(details=_DETAILS), auth
    )

    assert stored.details == _DETAILS
    assert stored.id_ == result.dataset.id_ == identity
    service._propagate_config_to_channel_datasets.assert_awaited_once_with(
        stored, service._get_handler.return_value, auth_context=auth
    )


@pytest.mark.parametrize(
    "data",
    [schemas.DataSetUpdateRequest(title="Renamed"), schemas.DataSetUpdateRequest(details=None)],
)
async def test_metadata_and_null_updates_preserve_stored_config(service, mocker, data):
    stored = _stored_dataset({**_DETAILS, "isOfficial": False})
    original = stored.details.copy()
    service.get_model_by_id = AsyncMock(return_value=stored)
    mocker.patch(
        "statgpt.admin.services.dataset.DataSetSerializer.db_to_schema",
        return_value=_dataset_schema(stored, original),
    )

    await AdminPortalDataSetService.update.__wrapped__(
        service, stored.id, data, MagicMock(spec=AuthContext)
    )

    assert stored.details == original


async def test_invalid_config_is_rejected_before_persistence(service):
    request = schemas.DataSetBase(title="Test", data_source_id=3, details={"urn": _DETAILS["urn"]})

    with pytest.raises(HTTPException) as exc:
        await AdminPortalDataSetService.create_dataset.__wrapped__(
            service, request, MagicMock(spec=AuthContext)
        )

    assert exc.value.status_code == 422
    service._session.add.assert_not_called()
