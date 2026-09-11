"""Tests for the channel import call of the admin client.

The query parameters are the contract with the admin API: a mode the request drops silently
turns a `replace` into an `upsert`, which is not a failure the user would see. The two
collections have their own mode, so the request has to carry both, and carry them apart.
"""

import httpx
import pytest

from statgpt.cli.shared.admin_client import AdminClient
from statgpt.common.schemas import RecordUploadMode

_JOB = {
    "id": 1,
    "type": "IMPORT",
    "status": "QUEUED",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "file": None,
    "channel_id": None,
}


def _client(handler) -> tuple[AdminClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(_record)
    return AdminClient(httpx.AsyncClient(transport=transport), "http://admin.invalid"), requests


def _archive(tmp_path) -> str:
    archive = tmp_path / "channel.zip"
    archive.write_bytes(b"PK\x03\x04")
    return str(archive)


@pytest.mark.asyncio
async def test_import_sends_the_archive_with_its_options(tmp_path) -> None:
    client, requests = _client(lambda request: httpx.Response(200, json=_JOB))

    await client.import_channel(
        _archive(tmp_path),
        clean_up=True,
        discovery_datasets_mode=RecordUploadMode.REPLACE,
        glossary_terms_mode=RecordUploadMode.REPLACE,
    )

    request = requests[0]
    assert request.url.path == "/admin/api/v1/channels/import"
    assert request.url.params["clean_up"] == "True"
    assert request.url.params["discovery_datasets_mode"] == "replace"
    assert request.url.params["glossary_terms_mode"] == "replace"
    assert b"PK" in request.content


@pytest.mark.asyncio
async def test_the_two_collections_are_sent_their_own_mode(tmp_path) -> None:
    """The point of two parameters: replacing the datasets must not touch the glossary."""
    client, requests = _client(lambda request: httpx.Response(200, json=_JOB))

    await client.import_channel(
        _archive(tmp_path),
        discovery_datasets_mode=RecordUploadMode.REPLACE,
        glossary_terms_mode=RecordUploadMode.UPSERT,
    )

    params = requests[0].url.params
    assert params["discovery_datasets_mode"] == "replace"
    assert params["glossary_terms_mode"] == "upsert"


@pytest.mark.asyncio
async def test_both_modes_default_to_upsert(tmp_path) -> None:
    """An import cannot delete records unasked, whichever collection they belong to."""
    client, requests = _client(lambda request: httpx.Response(200, json=_JOB))

    await client.import_channel(_archive(tmp_path))

    params = requests[0].url.params
    assert params["discovery_datasets_mode"] == "upsert"
    assert params["glossary_terms_mode"] == "upsert"
