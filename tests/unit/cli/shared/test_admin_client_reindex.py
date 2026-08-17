"""Tests that a rejected dataset does not stop a bulk reindex."""

import httpx
import pytest

from statgpt.cli.shared.admin_client import AdminAPIError, AdminClient
from statgpt.cli.shared.batch_report import BatchItemStatus

_CHANNEL_ID = 7
_DATASET_IDS = [1, 2, 3, 4]
_LABELS = {1: "AG:DF_A(1.0)", 2: "AG:DF_B(1.0)", 3: "AG:DF_C(1.0)", 4: "AG:DF_D(1.0)"}


def _client(handler) -> tuple[AdminClient, list[str]]:
    """Build a client over a mock transport, recording the paths it requests."""
    requested: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return handler(request)

    transport = httpx.MockTransport(_record)
    return AdminClient(httpx.AsyncClient(transport=transport), "http://admin.invalid"), requested


def _dataset_id_of(request: httpx.Request) -> int:
    return int(request.url.path.rstrip("/").split("/")[-2])


@pytest.mark.asyncio
async def test_a_rejected_dataset_does_not_stop_the_others() -> None:
    """The reported failure: the first 4xx used to abort before the later datasets."""

    def handler(request: httpx.Request) -> httpx.Response:
        if _dataset_id_of(request) == 2:
            return httpx.Response(
                400, json={"detail": {"dataset_urn": "AG:DF_B(1.0)", "error": "unsupported DSD"}}
            )
        return httpx.Response(202, json={})

    client, requested = _client(handler)
    report = await client.reload_channel_indicators(
        channel_id=_CHANNEL_ID, dataset_ids=_DATASET_IDS, labels=_LABELS
    )

    assert len(requested) == 4, "every dataset must be submitted, including those after the failure"

    by_status = {item.item_id: item.status for item in report.items}
    assert by_status == {
        "AG:DF_A(1.0)": BatchItemStatus.OK,
        "AG:DF_B(1.0)": BatchItemStatus.FAILED,
        "AG:DF_C(1.0)": BatchItemStatus.OK,
        "AG:DF_D(1.0)": BatchItemStatus.OK,
    }
    assert report.has_failures


@pytest.mark.asyncio
async def test_the_failure_message_carries_the_reason_from_the_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"detail": {"dataset_urn": "AG:DF_A(1.0)", "error": "unsupported DSD"}}
        )

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(
        channel_id=_CHANNEL_ID, dataset_ids=[1], labels=_LABELS
    )

    assert report.failed[0].message == "HTTP 400: unsupported DSD"


@pytest.mark.asyncio
async def test_a_detail_list_is_joined() -> None:
    """The channel-wide endpoint answers with a list of InvalidConfigurationError dicts."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": [{"error": "first"}, {"error": "second"}]})

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(
        channel_id=_CHANNEL_ID, dataset_ids=[1], labels=_LABELS
    )

    assert report.failed[0].message == "HTTP 400: first; second"


@pytest.mark.asyncio
async def test_a_non_json_error_body_still_yields_a_usable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(
        channel_id=_CHANNEL_ID, dataset_ids=[1], labels=_LABELS
    )

    assert report.failed[0].message == "HTTP 502: <html>Bad Gateway</html>"


@pytest.mark.asyncio
async def test_an_empty_error_body_still_yields_a_usable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(
        channel_id=_CHANNEL_ID, dataset_ids=[1], labels=_LABELS
    )

    assert report.failed[0].message == "HTTP 500"


@pytest.mark.asyncio
async def test_an_auth_failure_aborts_instead_of_repeating_per_dataset() -> None:
    """401 is not a per-dataset problem, so it must not be reported four times."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Not authenticated"})

    client, requested = _client(handler)

    with pytest.raises(AdminAPIError):
        await client.reload_channel_indicators(
            channel_id=_CHANNEL_ID, dataset_ids=_DATASET_IDS, labels=_LABELS
        )

    assert len(requested) == 1


@pytest.mark.asyncio
async def test_datasets_without_a_label_fall_back_to_their_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={})

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(channel_id=_CHANNEL_ID, dataset_ids=[9])

    assert report.items[0].item_id == "9"


@pytest.mark.asyncio
async def test_channel_wide_reindex_posts_once() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={})

    client, requested = _client(handler)
    report = await client.reload_channel_indicators(channel_id=_CHANNEL_ID, dataset_ids=None)

    assert requested == [f"/admin/api/v1/channels/{_CHANNEL_ID}/datasets/reload-indicators"]
    assert not report.has_failures


@pytest.mark.asyncio
async def test_channel_wide_reindex_records_its_rejection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": [{"error": "unsupported DSD"}]})

    client, _ = _client(handler)
    report = await client.reload_channel_indicators(channel_id=_CHANNEL_ID, dataset_ids=None)

    assert report.failed[0].message == "HTTP 400: unsupported DSD"
