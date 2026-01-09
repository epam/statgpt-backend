from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from statgpt.app.application import service_endpoints
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas.channel_dataset import ChannelDatasetExpanded, ChannelDatasetVersion
from statgpt.common.schemas.dataset import DataSet, Status
from statgpt.common.schemas.enums import PreprocessingStatusEnum


@dataclass
class _DummyChannel:
    id: int
    deployment_id: str
    title: str
    description: str
    llm_model: str
    details: dict[str, Any]


class _DummyChannelFacade:
    def __init__(self, channel: _DummyChannel, channel_config: ChannelConfig):
        self._channel = channel
        self._channel_config = channel_config

    @property
    def channel(self) -> _DummyChannel:
        return self._channel

    @property
    def channel_config(self) -> ChannelConfig:
        return self._channel_config


class _DummySystemAuthContext:
    # Minimal auth context stub for direct endpoint calls (FastAPI Depends isn't resolved in unit tests).
    is_system = True
    dial_access_token = None
    api_key = "test"


def _minimal_channel_config_dict() -> dict[str, Any]:
    return {
        "supreme_agent": {
            "name": "Test Bot",
            "domain": "test",
            "terminology_domain": "test",
        },
        "data_query": {
            "name": "data_query",
            "description": "Data query tool",
            "enabled": True,
            "details": {},
        },
    }


def _dummy_channel_dataset_expanded() -> ChannelDatasetExpanded:
    now = datetime.now(timezone.utc)
    dataset = DataSet(
        id=10,
        created_at=now,
        updated_at=now,
        id_=uuid.uuid4(),
        title="Dataset 1",
        description="",
        data_source_id=1,
        data_source=None,
        details={
            "dimensions": {
                "COUNTRY": {"dimensionType": "NON_INDICATOR", "subtype": "REGION"},
                "TIME_PERIOD": {"dimensionType": "TIME_PERIOD"},
                "INDEX_TYPE": {"dimensionType": "INDICATOR", "isRequired": True},
            }
        },
        status=Status(status="online", details=""),
    )
    latest_version = ChannelDatasetVersion(
        id=100,
        created_at=now,
        updated_at=now,
        channel_dataset_id=20,
        version=1,
        preprocessing_status=PreprocessingStatusEnum.COMPLETED,
        creation_reason="test",
        reason_for_failure=None,
        pointer_to=None,
        indicators_config_hash=None,
        non_indicators_config_hash=None,
        special_dimensions_config_hash=None,
        structure_metadata=None,
        structure_hash=None,
        indicator_dimensions_hash=None,
        non_indicator_dimensions_hash=None,
        special_dimensions_hash=None,
    )
    return ChannelDatasetExpanded(
        id=20,
        created_at=now,
        updated_at=now,
        channel_id=1,
        dataset_id=10,
        preprocessing_status=PreprocessingStatusEnum.COMPLETED,
        clearing_status=PreprocessingStatusEnum.NOT_STARTED,
        dataset=dataset,
        latest_version=latest_version,
        last_completed_version=latest_version,
        previous_completed_version=None,
    )


@pytest.mark.asyncio
async def test_channel_metadata_returns_tools(monkeypatch):
    channel = _DummyChannel(
        id=1,
        deployment_id="dep_1",
        title="Channel 1",
        description="desc",
        llm_model="model",
        details=_minimal_channel_config_dict(),
    )
    channel_config = ChannelConfig.model_validate(_minimal_channel_config_dict())
    facade = _DummyChannelFacade(channel, channel_config)

    async def _get_channel(_session: Any, deployment_id: str) -> _DummyChannelFacade:
        assert deployment_id == "dep_1"
        return facade

    monkeypatch.setattr(service_endpoints.ChannelServiceFacade, "get_channel", _get_channel)

    res = await service_endpoints.channel_metadata("dep_1", session=None)  # type: ignore[arg-type]
    assert res.deployment_id == "dep_1"
    assert res.title == "Channel 1"
    assert [t.name for t in res.tools] == ["data_query"]


@pytest.mark.asyncio
async def test_channel_datasets_metadata_returns_all_datasets(monkeypatch):
    channel = _DummyChannel(
        id=1,
        deployment_id="dep_2",
        title="Channel 2",
        description="desc",
        llm_model="model",
        details=_minimal_channel_config_dict(),
    )
    channel_config = ChannelConfig.model_validate(_minimal_channel_config_dict())
    facade = _DummyChannelFacade(channel, channel_config)

    async def _get_channel(_session: Any, deployment_id: str) -> _DummyChannelFacade:
        assert deployment_id == "dep_2"
        return facade

    async def _get_channel_dataset_schemas(
        self: Any,
        *,
        limit: int | None,
        offset: int,
        channel_id: int,
        auth_context: Any,
    ) -> list[ChannelDatasetExpanded]:
        assert limit is None
        assert offset == 0
        assert channel_id == 1
        assert auth_context.is_system is True
        return [_dummy_channel_dataset_expanded()]

    monkeypatch.setattr(service_endpoints.ChannelServiceFacade, "get_channel", _get_channel)
    monkeypatch.setattr(
        service_endpoints.DataSetService,
        "get_channel_dataset_schemas",
        _get_channel_dataset_schemas,
    )

    res = await service_endpoints.channel_datasets_metadata(
        "dep_2",
        auth_context=_DummySystemAuthContext(),  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
    )
    assert res.deployment_id == "dep_2"
    assert res.n_datasets == 1
    assert (
        res.datasets[0].dataset.details["dimensions"]["INDEX_TYPE"]["dimensionType"] == "INDICATOR"
    )
