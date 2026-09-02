"""Deleting a discovery record withdraws the document it published.

The reconciliation itself is `withdraw_documents`, covered in `test_discovery_publisher`. What
is checked here is the orchestration around it: that it runs before the rows go, that records
are grouped by the channel whose RAG application holds their documents, and that a channel with
nowhere to publish to is not stopped from deleting its records.
"""

import datetime
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services import discovery_dataset as service_module
from statgpt.admin.services.discovery_dataset import AdminPortalDiscoveryDatasetService
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.common import models
from statgpt.common.schemas import DiscoveryIndexingStatus, DiscoveryValidationStatus
from statgpt.common.services import GenericRagIngestionError, record_key

_APPLICATION = "discovery-rag-app"
_TIMESTAMP = datetime.datetime(2026, 8, 26, tzinfo=datetime.timezone.utc)


def _stored(
    item_id: int = 1,
    channel_id: int = 1,
    agency: str = "Bank Indonesia (BI)",
    dataset_id: str = "TABEL1_1",
) -> models.DiscoveryDataset:
    """A stand-in for a stored row, carrying the columns the service reads."""
    values: dict[str, object] = {name: "" for name in COLUMN_FIELDS}
    values.update(
        id=item_id,
        channel_id=channel_id,
        agency=agency,
        dataset_id=dataset_id,
        validation_status=DiscoveryValidationStatus.VALID,
        validation_issues=None,
        validated_at=None,
        indexing_status=DiscoveryIndexingStatus.INDEXED,
        indexed_at=None,
        index_error=None,
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
    )
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


@pytest.fixture
def calls() -> list[str]:
    """The order the steps ran in, which is what most of these tests are about."""
    return []


@pytest.fixture
def rag_client() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    return client


@pytest.fixture
def withdrawn() -> list[tuple[str, set[tuple[str, str]]]]:
    """One entry per `withdraw_documents` call: (channel deployment id, record keys)."""
    return []


@pytest.fixture(autouse=True)
def _rag(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    rag_client: AsyncMock,
    withdrawn: list[tuple[str, set[tuple[str, str]]]],
) -> None:
    """Stand in for everything between the service and the RAG channel."""

    def channel_service(_session: Any) -> Any:
        return SimpleNamespace(
            get_model_by_id=AsyncMock(
                side_effect=lambda channel_id: SimpleNamespace(
                    id=channel_id, deployment_id=f"channel-{channel_id}"
                )
            )
        )

    def db_to_schema(channel: Any) -> Any:
        # Channel 2 stands for one with no publish target configured.
        application_id = None if channel.id == 2 else f"{_APPLICATION}-{channel.id}"
        return SimpleNamespace(details=SimpleNamespace(discovery_application_id=application_id))

    async def withdraw(client: Any, channel: str, keys: Any, **_: Any) -> int:
        calls.append(f"withdraw:{channel}")
        withdrawn.append((channel, set(keys)))
        return len(keys)

    monkeypatch.setattr(service_module, "ChannelService", channel_service)
    monkeypatch.setattr(
        service_module, "ChannelSerializer", SimpleNamespace(db_to_schema=db_to_schema)
    )
    monkeypatch.setattr(
        service_module.GenericRagIngestionClient,
        "for_application",
        classmethod(lambda cls, application_id: rag_client),
    )
    monkeypatch.setattr(service_module, "withdraw_documents", withdraw)


def _service(
    calls: list[str], stored: Sequence[models.DiscoveryDataset]
) -> AdminPortalDiscoveryDatasetService:
    """A service whose session hands back `stored` for the select and records what ran."""
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())

    async def execute(statement: Any) -> Any:
        operation = "select" if statement.is_select else "delete"
        calls.append(operation)
        result = MagicMock()
        result.scalars.return_value.all.return_value = list(stored)
        return result

    service._session.execute = AsyncMock(side_effect=execute)  # type: ignore[method-assign]
    service._session.commit = AsyncMock(side_effect=lambda: calls.append("commit"))  # type: ignore[method-assign]
    service._session.delete = AsyncMock(side_effect=lambda _: calls.append("delete"))  # type: ignore[method-assign]
    return service


async def test_documents_are_withdrawn_before_the_rows_are_deleted(calls: list[str]) -> None:
    """A crash in between must leave a record with no document, never the reverse.

    A record with no document is what the next indexing run republishes; a document with no
    record is invisible to everything except an orphan sweep, and goes on being retrievable.
    """
    service = _service(calls, [_stored()])

    await service.delete_records_bulk(channel_id=1)

    assert calls == ["select", "withdraw:channel-1", "delete", "commit"]


async def test_deleted_records_are_returned_after_the_commit_expired_them(
    calls: list[str],
) -> None:
    """The response is built while the instances are still readable.

    A commit expires them, and reading an expired attribute in an async session raises
    instead of lazy-loading - so a response assembled afterwards would blow up.
    """
    service = _service(calls, [_stored(item_id=7, dataset_id="TABEL7")])

    deleted = await service.delete_records_bulk(channel_id=1)

    assert [(item.id, item.dataset_id) for item in deleted] == [(7, "TABEL7")]


async def test_records_are_withdrawn_from_the_channel_that_published_them(
    calls: list[str], withdrawn: list[tuple[str, set[tuple[str, str]]]]
) -> None:
    """Ids are global, so a bulk delete by id can span channels."""
    service = _service(
        calls,
        [
            _stored(item_id=1, channel_id=1, dataset_id="TABEL1"),
            _stored(item_id=2, channel_id=3, dataset_id="TABEL2"),
            _stored(item_id=3, channel_id=1, dataset_id="TABEL3"),
        ],
    )

    await service.delete_records_bulk(item_ids=[1, 2, 3])

    assert dict(withdrawn) == {
        "channel-1": {
            record_key("Bank Indonesia (BI)", "TABEL1"),
            record_key("Bank Indonesia (BI)", "TABEL3"),
        },
        "channel-3": {record_key("Bank Indonesia (BI)", "TABEL2")},
    }


async def test_a_channel_with_no_publish_target_still_deletes_its_records(
    calls: list[str], rag_client: AsyncMock
) -> None:
    """Nothing was ever published, so refusing over a document that cannot exist is wrong."""
    service = _service(calls, [_stored(channel_id=2)])

    deleted = await service.delete_records_bulk(channel_id=2)

    assert len(deleted) == 1
    assert calls == ["select", "delete", "commit"]
    rag_client.__aenter__.assert_not_awaited()


async def test_a_failed_withdrawal_leaves_the_rows_in_place(
    calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the record while its document survives is what this exists to prevent."""

    async def withdraw(*_: Any, **__: Any) -> int:
        raise GenericRagIngestionError("document deletion", "boom", 503)

    monkeypatch.setattr(service_module, "withdraw_documents", withdraw)
    service = _service(calls, [_stored()])

    with pytest.raises(GenericRagIngestionError):
        await service.delete_records_bulk(channel_id=1)

    assert calls == ["select"]


async def test_a_channel_with_no_records_costs_no_round_trip(
    calls: list[str], rag_client: AsyncMock
) -> None:
    service = _service(calls, [])

    assert await service.delete_records_bulk(channel_id=1) == []
    assert calls == ["select"]
    rag_client.__aenter__.assert_not_awaited()


async def test_deleting_one_record_withdraws_its_document(
    calls: list[str], withdrawn: list[tuple[str, set[tuple[str, str]]]]
) -> None:
    """The single-record route goes through the same step as the bulk ones."""
    item = _stored(item_id=5, dataset_id="TABEL5")
    service = _service(calls, [item])
    service.get_record_model_by_id = AsyncMock(return_value=item)  # type: ignore[method-assign]

    await service.delete(5)

    assert calls == ["withdraw:channel-1", "delete", "commit"]
    assert withdrawn == [("channel-1", {record_key("Bank Indonesia (BI)", "TABEL5")})]
