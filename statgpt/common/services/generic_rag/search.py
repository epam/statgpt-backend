"""Read access to the documents of a Generic RAG channel.

What a chat-time lookup needs: rank documents against a query, then read the body of the ones
it keeps. The ranking endpoint returns documents without their content - our discovery documents
carry the description as the body and everything else as metadata - so the two calls together
are what reconstitutes a full record.

Authenticates with the caller's own DIAL key: this runs inside a chat turn, on behalf of a user.
"""

from typing import Self

import httpx
from pydantic import RootModel, SecretStr

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.generic_rag import GenericRagDocument, GenericRagDocumentSearchRequest

from .client import BaseGenericRagChannelClient


class _DocumentList(RootModel[list[GenericRagDocument]]):
    """The bare JSON array `POST /channel/documents/search` answers with.

    `_parse` validates one model against a body, and a list is not one, so the array is given a
    model of its own - a body that is not an array then fails as a failure of the search call
    rather than as a `TypeError` somewhere downstream.
    """


class GenericRagSearchClient(BaseGenericRagChannelClient):
    """The document read endpoints of one Generic RAG channel."""

    @classmethod
    def for_application(cls, application_id: str, auth_context: AuthContext) -> Self:
        """Target the channel of a DIAL application, as the user behind `auth_context`.

        The channel endpoints authorize on the DIAL key alone, so the user must have access to
        the application; a user who does not gets an HTTP error rather than an empty result.
        """
        return cls(
            base_url=cls.application_route(application_id),
            api_key=SecretStr(auth_context.api_key),
        )

    async def search_documents(
        self, query: str, limit: int, indexes: list[str] | None = None
    ) -> list[GenericRagDocument]:
        """Documents relevant to `query`, best first.

        The service fuses the ranks of every index it searched, so the position in this list is
        the only relevance signal available - no scores are returned.
        """
        request = GenericRagDocumentSearchRequest(query=query, limit=limit, indexes=indexes)
        response = await self._request(
            "document search",
            "POST",
            "/documents/search",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        return self._parse(_DocumentList, response, "document search").root

    async def download_document(self, document_id: int) -> str:
        """The document's body as text.

        Discovery documents are plain UTF-8 text, so this is decoded here rather than handed
        back as bytes for every caller to decode identically.
        """
        response: httpx.Response = await self._request(
            "document download", "GET", f"/documents/{document_id}/download"
        )
        return response.text
