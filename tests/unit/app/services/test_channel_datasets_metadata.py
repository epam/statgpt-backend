import datetime
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.app.services.channel_datasets_metadata import build_channel_datasets_metadata
from statgpt.common.schemas import (
    ChannelDatasetExpandedWithLastUpdatedAt,
    ChannelDatasetVersion,
    DataSet,
)
from statgpt.common.schemas.dataset import Status
from statgpt.common.schemas.enums import PreprocessingStatusEnum


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {
            "urn": {"agencyId": "TEST", "resourceId": "DATA", "version": "1.0"},
            "isOfficial": False,
            "citation": None,
        },
    ],
)
async def test_metadata_uses_resolved_snapshot_without_materializing_live_defaults(
    mocker, snapshot
):
    now = datetime.datetime(2026, 1, 1)
    live = {"urn": {"agencyId": "TEST", "resourceId": "DATA"}}
    dataset = DataSet(
        id=1,
        id_=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        title="Test",
        data_source_id=1,
        data_source=None,
        details=live,
        status=Status(status="offline"),
    )
    version = ChannelDatasetVersion(
        id=1,
        created_at=now,
        updated_at=now,
        channel_dataset_id=1,
        version=1,
        preprocessing_status=PreprocessingStatusEnum.COMPLETED,
        creation_reason="Test",
        reason_for_failure=None,
        pointer_to=None,
        indexing_config_hash=None,
        structure_metadata=None,
        structure_hash=None,
        indicator_dimensions_hash=None,
        non_indicator_dimensions_hash=None,
        special_dimensions_hash=None,
        resolved_config=snapshot,
    )
    channel_dataset = ChannelDatasetExpandedWithLastUpdatedAt(
        id=1,
        created_at=now,
        updated_at=now,
        channel_id=1,
        dataset_id=1,
        dataset=dataset,
        preprocessing_status=PreprocessingStatusEnum.COMPLETED,
        clearing_status=PreprocessingStatusEnum.NOT_STARTED,
        last_completed_version=version,
        previous_completed_version=None,
        latest_version=version,
        last_auto_update_job=None,
    )
    original_snapshot = deepcopy(snapshot)

    @asynccontextmanager
    async def session_context():
        yield MagicMock()

    mocker.patch(
        "statgpt.app.services.channel_datasets_metadata.get_readonly_session_context_manager",
        session_context,
    )
    service = mocker.patch(
        "statgpt.app.services.channel_datasets_metadata.DataSetService"
    ).return_value
    service.get_channel_dataset_schemas_with_last_updated = AsyncMock(
        return_value=[channel_dataset]
    )

    response = await build_channel_datasets_metadata(
        SimpleNamespace(id=1, deployment_id="test", title="Test"), MagicMock()
    )

    assert response.datasets[0].dataset.details == (snapshot if snapshot is not None else live)
    assert dataset.details == live
    assert version.resolved_config == original_snapshot
    assert response.model_dump(mode="json")["datasets"][0]["dataset"]["details"] == (
        snapshot if snapshot is not None else live
    )
