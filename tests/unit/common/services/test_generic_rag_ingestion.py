"""Tests for the Generic RAG channel ingestion client."""

from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from statgpt.common.services import GenericRagIngestionClient, GenericRagIngestionError

_BASE_URL = "http://core:8080/v1/deployments/generic-rag-app/route"

Handler = Callable[[httpx.Request], httpx.Response]


class _Metadata(BaseModel):
    agency: str
    statgpt_channel: str


def _client(handler: Handler) -> GenericRagIngestionClient:
    """A client whose own managed HTTP client serves a mocked transport."""
    client = GenericRagIngestionClient(base_url=_BASE_URL, api_key=SecretStr("test-key"))
    client._http.client  # noqa: B018  - create the lazy client before replacing its transport
    client._http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _document(document_id: int, **metadata: object) -> dict[str, object]:
    return {
        "id": document_id,
        "url": f"files/doc{document_id}.md",
        "display_name": f"doc{document_id}.md",
        "mime_type": "text/markdown",
        "size": 10,
        "metadata": metadata,
        "status": "ready",
    }


def _page(results: list[dict[str, object]], total_count: int, offset: int = 0) -> dict[str, object]:
    return {"total_count": total_count, "offset": offset, "limit": 500, "results": results}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ addressing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_requests_go_to_the_application_channel_route_with_the_api_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    async with _client(handler) as client:
        await client.delete_document(7)

    (request,) = requests
    assert request.method == "DELETE"
    assert str(request.url) == f"{_BASE_URL}/channel/documents/7"
    assert request.headers["Api-Key"] == "test-key"


async def test_for_application_builds_the_dial_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from statgpt.common.services.generic_rag import ingestion

    monkeypatch.setattr(ingestion.dial_settings, "url", "http://core:8080/", raising=False)

    client = ingestion.GenericRagIngestionClient.for_application("generic-rag-app")

    assert client.base_url == "http://core:8080/v1/deployments/generic-rag-app/route"


async def test_for_application_supports_a_custom_application_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom application id carries its bucket as path segments and must not be escaped."""
    from statgpt.common.services.generic_rag import ingestion

    monkeypatch.setattr(ingestion.dial_settings, "url", "http://core:8080", raising=False)

    client = ingestion.GenericRagIngestionClient.for_application("applications/bucket42/rag")

    assert client.base_url == "http://core:8080/v1/deployments/applications/bucket42/rag/route"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ listing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_listing_pages_to_exhaustion() -> None:
    pages = [
        _page([_document(1), _document(2)], total_count=3),
        _page([_document(3)], total_count=3, offset=2),
    ]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("offset"))
        return httpx.Response(200, json=pages[len(seen) - 1])

    async with _client(handler) as client:
        documents = await client.list_documents()

    assert [d.id for d in documents] == [1, 2, 3]
    assert seen == ["0", "2"]


async def test_listing_stops_on_an_empty_page() -> None:
    """A service that caps `limit` below the request must not spin forever."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_page([], total_count=99))

    async with _client(handler) as client:
        assert await client.list_documents() == []

    assert calls == 1


async def test_listing_tolerates_unknown_fields_and_statuses() -> None:
    document = _document(1) | {"status": "quarantined", "some_new_field": True}

    async with _client(lambda _: httpx.Response(200, json=_page([document], 1))) as client:
        (parsed,) = await client.list_documents()

    assert parsed.status == "quarantined"
    assert parsed.is_failed is False


async def test_a_document_in_error_state_reads_as_failed() -> None:
    document = _document(1) | {"status": "error"}

    async with _client(lambda _: httpx.Response(200, json=_page([document], 1))) as client:
        (parsed,) = await client.list_documents()

    assert parsed.is_failed is True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ upload ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_upload_sends_the_file_and_the_metadata_as_json() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=_document(5))

    async with _client(handler) as client:
        document = await client.upload_document(
            filename="Bank Indonesia (BI) - TABEL1_1.md",
            content=b"# I.1. Broad Money",
            mime_type="text/markdown",
            metadata=_Metadata(agency="Bank Indonesia (BI)", statgpt_channel="statgpt-gtdc"),
        )

    assert document.id == 5
    (request,) = requests
    body = request.content.decode()
    assert request.method == "POST"
    assert request.url.path == "/v1/deployments/generic-rag-app/route/channel/documents"
    assert 'name="attachment"; filename="Bank Indonesia (BI) - TABEL1_1.md"' in body
    assert "# I.1. Broad Money" in body
    assert '{"agency":"Bank Indonesia (BI)","statgpt_channel":"statgpt-gtdc"}' in body


@pytest.mark.parametrize(("overwrite", "expected"), [(True, "true"), (False, "false")])
async def test_upload_asks_to_overwrite_only_when_told_to(overwrite: bool, expected: str) -> None:
    """The service derives the storage path from the file name, and refuses to replace a file
    that is already there unless asked."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=_document(5))

    async with _client(handler) as client:
        await client.upload_document(
            filename="x.md",
            content=b"x",
            mime_type="text/markdown",
            metadata=_Metadata(agency="a", statgpt_channel="c"),
            overwrite=overwrite,
        )

    assert requests[0].url.params.get("overwrite") == expected


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ update ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_update_replaces_the_content_and_the_metadata_of_one_document() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_document(7))

    async with _client(handler) as client:
        document = await client.update_document(
            7,
            filename="Bank Indonesia (BI) - TABEL1_1.md",
            content=b"# I.1. Broad Money",
            mime_type="text/markdown",
            metadata=_Metadata(agency="Bank Indonesia (BI)", statgpt_channel="statgpt-gtdc"),
        )

    assert document.id == 7
    (request,) = requests
    body = request.content.decode()
    assert request.method == "PUT"
    assert request.url.path == "/v1/deployments/generic-rag-app/route/channel/documents/7"
    assert "# I.1. Broad Money" in body
    assert '{"agency":"Bank Indonesia (BI)","statgpt_channel":"statgpt-gtdc"}' in body


async def test_a_failed_update_names_that_operation() -> None:
    """The reason is written to the record, so it has to say which call failed."""
    async with _client(lambda _: httpx.Response(404, text="no such document")) as client:
        with pytest.raises(GenericRagIngestionError, match="document update") as caught:
            await client.update_document(
                7,
                filename="x.md",
                content=b"x",
                mime_type="text/markdown",
                metadata=_Metadata(agency="a", statgpt_channel="c"),
            )

    assert caught.value.status_code == 404


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ metadata schema ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_filterable_fields_come_from_the_schema_not_the_dimensions() -> None:
    """`dimensions` only lists values documents carry, so an empty channel lists none."""
    payload = {
        "schema": {
            "properties": {
                "agency": {"type": "string", "enable_filtering": True},
                "reference_area": {"type": "string", "enable_filtering": True},
                "description": {"type": "string"},
            }
        },
        "dimensions": {},
    }

    async with _client(lambda _: httpx.Response(200, json=payload)) as client:
        schema = await client.get_metadata_schema()

    assert schema.filterable_fields == {"agency", "reference_area"}


async def test_filterable_fields_of_a_schema_without_properties() -> None:
    async with _client(lambda _: httpx.Response(200, json={"schema": {}, "dimensions": {}})) as c:
        schema = await c.get_metadata_schema()

    assert schema.filterable_fields == set()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ failures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_error_response_names_the_operation_and_the_status() -> None:
    async with _client(lambda _: httpx.Response(422, text="bad metadata")) as client:
        with pytest.raises(GenericRagIngestionError, match="document upload failed with HTTP 422"):
            await client.upload_document(
                filename="x.md",
                content=b"x",
                mime_type="text/markdown",
                metadata=_Metadata(agency="a", statgpt_channel="c"),
            )


async def test_a_transport_failure_is_reported_as_the_same_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(GenericRagIngestionError, match="document listing failed"):
            await client.list_documents()


async def test_an_unparsable_body_is_reported_as_a_failure_of_that_call() -> None:
    async with _client(lambda _: httpx.Response(200, text="not json")) as client:
        with pytest.raises(GenericRagIngestionError, match="unexpected response body"):
            await client.get_metadata_schema()


async def test_a_long_error_body_is_truncated() -> None:
    """The reason lands in a database column, so an HTML error page must not arrive whole."""
    async with _client(lambda _: httpx.Response(500, text="x" * 5000)) as client:
        with pytest.raises(GenericRagIngestionError) as exc_info:
            await client.delete_document(1)

    assert exc_info.value.detail.endswith("... (truncated)")
    assert len(exc_info.value.detail) < 600
