"""Tests for the Generic RAG channel search client."""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from statgpt.common.services import GenericRagChannelError, GenericRagSearchClient

_BASE_URL = "http://core:8080/v1/deployments/generic-rag-app/route"

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> GenericRagSearchClient:
    """A client whose own managed HTTP client serves a mocked transport."""
    client = GenericRagSearchClient(base_url=_BASE_URL, api_key=SecretStr("user-key"))
    client._http.client  # noqa: B018  - create the lazy client before replacing its transport
    client._http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _document(document_id: int, **metadata: object) -> dict[str, object]:
    return {
        "id": document_id,
        "url": f"files/doc{document_id}.txt",
        "display_name": f"doc{document_id}.txt",
        "mime_type": "text/plain",
        "size": 10,
        "metadata": metadata,
        "status": "ready",
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ search ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_search_posts_the_query_and_limit_with_the_callers_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        await client.search_documents("gdp of france", limit=7)

    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == f"{_BASE_URL}/channel/documents/search"
    assert request.headers["Api-Key"] == "user-key"
    assert json.loads(request.content) == {"query": "gdp of france", "limit": 7}


async def test_indexes_are_sent_only_when_configured() -> None:
    """An omitted `indexes` leaves the choice of indexes to the channel."""
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        await client.search_documents("q", limit=1)
        await client.search_documents("q", limit=1, indexes=["semantic"])

    assert "indexes" not in json.loads(bodies[0])
    assert json.loads(bodies[1])["indexes"] == ["semantic"]


async def test_search_returns_the_documents_in_the_order_the_service_gave_them() -> None:
    """The order is the only relevance signal: the endpoint returns no scores."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[_document(5, agency="IMF"), _document(2), _document(9, agency="OECD")]
        )

    async with _client(handler) as client:
        documents = await client.search_documents("q", limit=10)

    assert [document.id for document in documents] == [5, 2, 9]
    assert documents[0].metadata == {"agency": "IMF"}


async def test_a_body_that_is_not_a_document_array_fails_as_a_search_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        with pytest.raises(GenericRagChannelError, match="document search"):
            await client.search_documents("q", limit=1)


async def test_an_http_failure_names_the_search_and_carries_the_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    async with _client(handler) as client:
        with pytest.raises(GenericRagChannelError) as excinfo:
            await client.search_documents("q", limit=1)

    assert excinfo.value.status_code == 403
    assert "document search" in str(excinfo.value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ download ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_download_returns_the_document_body_as_text() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="Quarterly national accounts, 1995-2024.")

    async with _client(handler) as client:
        body = await client.download_document(11)

    assert body == "Quarterly national accounts, 1995-2024."
    (request,) = requests
    assert request.method == "GET"
    assert str(request.url) == f"{_BASE_URL}/channel/documents/11/download"


async def test_for_application_authenticates_as_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from statgpt.common.services.generic_rag import client as client_module

    monkeypatch.setattr(client_module.dial_settings, "url", "http://core:8080/", raising=False)

    class _AuthContext:
        api_key = "caller-key"

    client = GenericRagSearchClient.for_application("generic-rag-app", _AuthContext())  # type: ignore[arg-type]

    assert client.base_url == "http://core:8080/v1/deployments/generic-rag-app/route"
    assert client._api_key.get_secret_value() == "caller-key"
