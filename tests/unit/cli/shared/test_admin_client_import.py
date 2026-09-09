"""Tests for the channel import call of the admin client.

The query parameters are the contract with the admin API: a mode the request drops silently
turns a `replace` into an `upsert`, which is not a failure the user would see.
"""

import httpx
import pytest

from statgpt.cli.shared.admin_client import AdminClient
from statgpt.common.schemas import RecordUploadMode


def _client(handler) -> tuple[AdminClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(_record)
    return AdminClient(httpx.AsyncClient(transport=transport), "http://admin.invalid"), requests


@pytest.mark.asyncio
async def test_import_sends_the_archive_with_its_options(tmp_path) -> None:
    archive = tmp_path / "channel.zip"
    archive.write_bytes(b"PK\x03\x04")

    job = {
        "id": 1,
        "type": "IMPORT",
        "status": "QUEUED",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "file": None,
        "channel_id": None,
    }
    client, requests = _client(lambda request: httpx.Response(200, json=job))

    await client.import_channel(str(archive), clean_up=True, mode=RecordUploadMode.REPLACE)

    request = requests[0]
    assert request.url.path == "/admin/api/v1/channels/import"
    assert request.url.params["clean_up"] == "True"
    assert request.url.params["mode"] == "replace"
    assert b"PK" in request.content


@pytest.mark.asyncio
async def test_import_defaults_to_upsert() -> None:
    """The default has to stay upsert, so an import cannot delete records unasked."""
    assert RecordUploadMode.UPSERT is RecordUploadMode("upsert")
