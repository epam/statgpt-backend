"""Tests for the discovery dataset calls of the admin client.

The paths and query parameters are the contract with the admin API, and a refused upload has
to arrive as the per-cell report rather than as a response body.
"""

import httpx
import pytest

from statgpt.cli.shared.admin_client import AdminAPIError, AdminClient, DiscoveryPayloadError
from statgpt.common.schemas import DiscoveryIndexingStatus, DiscoveryUploadMode

_CHANNEL_ID = 7


def _client(handler) -> tuple[AdminClient, list[httpx.Request]]:
    """Build a client over a mock transport, recording the requests it makes."""
    requests: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(_record)
    return AdminClient(httpx.AsyncClient(transport=transport), "http://admin.invalid"), requests


@pytest.mark.asyncio
async def test_stats_are_requested_per_channel_and_parsed() -> None:
    payload = {
        "total": 3,
        "by_validation_status": {"VALID": 2, "INVALID": 1, "NOT_VALIDATED": 0},
        "by_indexing_status": {"INDEXED": 2, "NEW": 1, "OUTDATED": 0, "FAILED": 0},
    }
    client, requests = _client(lambda request: httpx.Response(200, json=payload))

    stats = await client.get_discovery_stats(_CHANNEL_ID)

    assert requests[0].url.path == "/admin/api/v1/channels/7/discovery-datasets/stats"
    assert stats.total == 3
    assert stats.by_indexing_status[DiscoveryIndexingStatus.INDEXED] == 2


@pytest.mark.asyncio
async def test_upload_posts_the_file_with_its_name_and_mode(tmp_path) -> None:
    """The API classifies a file by its bytes and then its name, so both have to travel."""
    file = tmp_path / "records.csv"
    file.write_text("agency,dataset_id\nBPS,TABEL1_1\n", encoding="utf-8")

    summary = {"created": 1, "updated": 0, "unchanged": 0, "deleted": 0, "rows_read": 1}
    client, requests = _client(lambda request: httpx.Response(200, json=summary))

    result = await client.upload_discovery_datasets(
        _CHANNEL_ID, str(file), DiscoveryUploadMode.REPLACE
    )

    request = requests[0]
    assert request.url.path == "/admin/api/v1/channels/7/discovery-datasets/upload"
    assert request.url.params["mode"] == "replace"
    assert b'filename="records.csv"' in request.content
    assert result.created == 1


@pytest.mark.asyncio
async def test_a_refused_upload_carries_the_per_cell_report(tmp_path) -> None:
    file = tmp_path / "records.csv"
    file.write_text("agency,dataset_id\n,TABEL1_1\n", encoding="utf-8")

    body = {
        "detail": {
            "message": "2 problem(s) found",
            "problems": [
                {"message": "must not be empty", "field": "agency", "row": 2, "cell": "D2"},
                {"message": "duplicate record", "field": "dataset_id", "row": 3},
            ],
            "truncated": False,
        }
    }
    client, _ = _client(lambda request: httpx.Response(400, json=body))

    with pytest.raises(DiscoveryPayloadError) as excinfo:
        await client.upload_discovery_datasets(_CHANNEL_ID, str(file))

    detail = excinfo.value.detail
    assert detail.message == "2 problem(s) found"
    assert [problem.cell for problem in detail.problems] == ["D2", None]
    assert str(excinfo.value) == "2 problem(s) found"


@pytest.mark.asyncio
async def test_a_400_without_the_report_shape_stays_a_generic_error(tmp_path) -> None:
    """Not every 400 carries the report - one raised before the handler runs does not."""
    file = tmp_path / "records.csv"
    file.write_text("agency,dataset_id\n", encoding="utf-8")

    client, _ = _client(lambda request: httpx.Response(400, json={"detail": "bad request"}))

    with pytest.raises(AdminAPIError) as excinfo:
        await client.upload_discovery_datasets(_CHANNEL_ID, str(file))

    assert not isinstance(excinfo.value, DiscoveryPayloadError)
    assert "bad request" in str(excinfo.value)


@pytest.mark.asyncio
async def test_indexing_is_triggered_with_force_and_polled_by_job_id() -> None:
    job = {
        "id": 17,
        "channel_id": _CHANNEL_ID,
        "status": "QUEUED",
        "createdAt": "2026-08-25T10:00:00Z",
        "updatedAt": "2026-08-25T10:00:00Z",
    }
    client, requests = _client(lambda request: httpx.Response(202, json=job))

    triggered = await client.trigger_discovery_indexing(_CHANNEL_ID, force=True)

    assert requests[0].url.path == "/admin/api/v1/channels/7/discovery-datasets/indexing-jobs"
    assert requests[0].url.params["force"] == "true"
    assert triggered.id == 17

    client, requests = _client(lambda request: httpx.Response(200, json=job))
    await client.get_discovery_indexing_job(17)
    assert requests[0].url.path == "/admin/api/v1/discovery-datasets/indexing-jobs/17"


@pytest.mark.asyncio
async def test_a_conflict_reports_the_reason_the_api_gave() -> None:
    """A 409 is a state the operator can act on, so it must not arrive as an HTTP dump."""
    client, _ = _client(
        lambda request: httpx.Response(
            409, json={"detail": {"error": "indexing job 3 is already running"}}
        )
    )

    with pytest.raises(AdminAPIError) as excinfo:
        await client.trigger_discovery_indexing(_CHANNEL_ID)

    assert "indexing job 3 is already running" in str(excinfo.value)
    assert "Response body" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_clear_deletes_the_channels_records_and_parses_the_bare_list() -> None:
    """The endpoint answers with the deleted records themselves, not a `data` envelope."""
    record = {
        "id": 1,
        "channelId": _CHANNEL_ID,
        "agency": "Bank Indonesia (BI)",
        "datasetId": "TABEL1_1",
        "validationStatus": "VALID",
        "indexingStatus": "INDEXED",
        "createdAt": "2026-08-26T00:00:00Z",
        "updatedAt": "2026-08-26T00:00:00Z",
    }
    client, requests = _client(lambda request: httpx.Response(200, json=[record]))

    deleted = await client.clear_discovery_datasets(_CHANNEL_ID)

    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/admin/api/v1/channels/7/discovery-datasets/bulk"
    assert [item.dataset_id for item in deleted] == ["TABEL1_1"]


@pytest.mark.asyncio
async def test_clear_reports_a_rag_channel_that_would_not_serve_the_withdrawal() -> None:
    """Deleting withdraws documents, so an unreachable RAG channel fails the call."""
    client, _ = _client(
        lambda request: httpx.Response(502, json={"detail": "Generic RAG document deletion failed"})
    )

    with pytest.raises(AdminAPIError) as exc_info:
        await client.clear_discovery_datasets(_CHANNEL_ID)

    assert exc_info.value.status_code == 502
