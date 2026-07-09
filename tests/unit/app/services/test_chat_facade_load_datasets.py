"""Unit tests for ChannelServiceFacade._load_datasets ordering and filtering."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from statgpt.app.services.chat_facade import ChannelServiceFacade, VersionedDataSet


class _FakeHandler:
    """Handler stub whose behavior is driven by per-dataset config flags."""

    async def is_dataset_available(self, config: dict, auth_context: Any) -> bool:
        return config["available"]

    async def get_dataset(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: dict,
        auth_context: Any,
        allow_offline: bool = False,
        allow_cached: bool = False,
        force_refresh: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(title=title, status=SimpleNamespace(status=config["status"]))


def _dataset_model(
    db_id: int, title: str, *, available: bool = True, status: str = "online", source_id: int = 100
) -> SimpleNamespace:
    return SimpleNamespace(
        id=db_id,
        id_=uuid.uuid4(),
        title=title,
        details={"available": available, "status": status},
        source_id=source_id,
    )


async def _run_load_datasets(
    dataset_models: list[SimpleNamespace], handler: Any
) -> list[VersionedDataSet]:
    """Run `_load_datasets` with the DB layer and handler resolution mocked out."""
    version = SimpleNamespace(resolved_config=None)
    last_versions = {
        model.id: SimpleNamespace(last_completed_version=version) for model in dataset_models
    }
    dataset_service = MagicMock()
    dataset_service.get_latest_successful_dataset_versions_for_channel = AsyncMock(
        return_value=last_versions
    )
    dataset_service.get_datasets_models = AsyncMock(return_value=dataset_models)

    data_sources = [
        SimpleNamespace(
            id=source_id, type=SimpleNamespace(id=source_id, name=f"TYPE_{source_id}"), details={}
        )
        for source_id in {model.source_id for model in dataset_models}
    ]
    data_source_service = MagicMock()
    data_source_service.get_data_sources_schemas = AsyncMock(return_value=data_sources)

    @asynccontextmanager
    async def fake_session_cm():
        yield MagicMock()

    channel = MagicMock()
    channel.id = 1
    facade = ChannelServiceFacade(channel=channel)

    with (
        patch(
            "statgpt.app.services.chat_facade.get_readonly_session_context_manager",
            fake_session_cm,
        ),
        patch("statgpt.app.services.chat_facade.DataSetService", return_value=dataset_service),
        patch(
            "statgpt.app.services.chat_facade.DataSourceService",
            return_value=data_source_service,
        ),
        patch.object(ChannelServiceFacade, "_get_handler_class", AsyncMock(return_value=handler)),
    ):
        return await facade._load_datasets(auth_context=MagicMock())


class TestLoadDatasets:

    @pytest.mark.asyncio
    async def test_preserves_input_order(self) -> None:
        models = [_dataset_model(i, f"ds{i}") for i in range(1, 6)]

        result = await _run_load_datasets(models, _FakeHandler())

        assert [v.data.title for v in result] == ["ds1", "ds2", "ds3", "ds4", "ds5"]
        assert all(isinstance(v, VersionedDataSet) for v in result)

    @pytest.mark.asyncio
    async def test_filters_unavailable_and_offline_datasets(self) -> None:
        models = [
            _dataset_model(1, "ds1"),
            _dataset_model(2, "ds2", available=False, source_id=200),
            _dataset_model(3, "ds3", status="offline"),
            _dataset_model(4, "ds4", source_id=200),
        ]

        result = await _run_load_datasets(models, _FakeHandler())

        assert [v.data.title for v in result] == ["ds1", "ds4"]

    @pytest.mark.asyncio
    async def test_all_datasets_filtered_out(self) -> None:
        models = [
            _dataset_model(1, "ds1", available=False),
            _dataset_model(2, "ds2", status="offline"),
        ]

        result = await _run_load_datasets(models, _FakeHandler())

        assert result == []

    @pytest.mark.asyncio
    async def test_order_preserved_when_later_datasets_finish_first(self) -> None:
        release_first = asyncio.Event()

        class _StaggeredHandler(_FakeHandler):
            async def get_dataset(self, entity_id, title, config, auth_context, **kwargs):
                if title == "ds1":
                    await release_first.wait()  # ds1 completes only after ds3
                result = await super().get_dataset(entity_id, title, config, auth_context)
                if title == "ds3":
                    release_first.set()
                return result

        models = [_dataset_model(i, f"ds{i}") for i in range(1, 4)]

        result = await _run_load_datasets(models, _StaggeredHandler())

        assert [v.data.title for v in result] == ["ds1", "ds2", "ds3"]
