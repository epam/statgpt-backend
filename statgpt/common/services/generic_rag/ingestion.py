"""Write access to the documents of a Generic RAG channel.

The read side of the Generic RAG integration goes through chat completions; this is the
other half - what an administrator-triggered indexing run needs to put records into a
channel and take them out again.

Lives in `common` because it is shared: Grade C publishes uploaded workbook records and
Grade B will publish crawled registry records into the same contract, and `common` cannot
import from `admin`.
"""

import logging
from typing import Self, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from statgpt.common.schemas.generic_rag import (
    GenericRagDocument,
    GenericRagDocumentPage,
    GenericRagMetadataSchema,
)
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import ManagedHttpClient

_log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(60.0)

_PAGE_SIZE = 500
"""How many documents to ask for per listing page.

The API declares no maximum and defaults to 25, which would page a few hundred records
into more round trips than the run needs.
"""

_MAX_DETAIL_CHARS = 500
"""Cap on how much of a failure body is quoted.

The reason ends up in `index_error` or `reason_for_failure`, so an HTML error page must not
arrive there in full.
"""

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class GenericRagIngestionError(Exception):
    """A request to the Generic RAG channel API failed.

    Carries the operation and, when there was a response, its status: the message is what a
    run records on the record or the job row, so it has to say which call failed and what
    came back.
    """

    def __init__(self, operation: str, detail: str, status_code: int | None = None) -> None:
        self.operation = operation
        self.detail = detail
        self.status_code = status_code
        code = f" with HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"Generic RAG {operation} failed{code}: {detail}")


class GenericRagIngestionClient:
    """The document endpoints of one Generic RAG channel.

    A channel is addressed by the DIAL application that fronts it, so one client instance
    speaks to exactly one channel. Own the instance for the length of a run - it holds an
    HTTP connection pool - and close it, ideally by using it as an async context manager.

    Documents are identified by a service-assigned integer, never by a key the caller chooses,
    so a caller that needs to find its own document again has to recognize it by the metadata
    `list_documents` returns. An `overwrite` upload is keyed on the file name rather than on
    that id, which is the closest thing to an upsert the channel offers.
    """

    def __init__(self, base_url: str, api_key: SecretStr) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = ManagedHttpClient(_HTTP_TIMEOUT)

    @classmethod
    def for_application(cls, application_id: str) -> Self:
        """Target the channel of a DIAL application, by application id.

        Authenticates with the deployment's own DIAL key rather than a user token: this runs
        in a background job, which has no user, and the Generic RAG application forwards an
        auth token only on its chat path.
        """
        base_url = f"{dial_settings.url.rstrip('/')}/v1/deployments/{application_id}/route"
        return cls(base_url=base_url, api_key=dial_settings.api_key)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ endpoints ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def get_metadata_schema(self) -> GenericRagMetadataSchema:
        """Read the metadata JSON-schema the channel accepts, and its filterable fields."""
        response = await self._request("metadata read", "GET", "/metadata")
        return self._parse(GenericRagMetadataSchema, response, "metadata read")

    async def list_documents(self) -> list[GenericRagDocument]:
        """Every document in the channel, paged to exhaustion."""
        documents: list[GenericRagDocument] = []
        offset = 0
        while True:
            response = await self._request(
                "document listing",
                "GET",
                "/documents",
                params={"offset": offset, "limit": _PAGE_SIZE},
            )
            page = self._parse(GenericRagDocumentPage, response, "document listing")
            documents.extend(page.results)
            offset += len(page.results)
            # An empty page also ends the loop: the service may cap `limit` below what was
            # asked for, and without this a `total_count` that never becomes reachable
            # would spin forever.
            if not page.results or offset >= page.total_count:
                return documents

    async def upload_document(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        metadata: BaseModel,
        overwrite: bool = False,
    ) -> GenericRagDocument:
        """Add a document. `metadata` must satisfy the channel's metadata JSON-schema.

        The service echoes `filename` back as the document's `display_name`, and derives the
        storage path from it, so uploading the same name twice targets the same file.

        `overwrite` decides what happens when that file is already there: without it the
        upload is refused, and with it the file is replaced and the document that owns it
        updated in place, keeping its id. Pass it when the caller owns the name it is
        uploading under - otherwise a file left behind by a document that no longer exists
        blocks that name for good.
        """
        response = await self._request(
            "document upload",
            "POST",
            "/documents",
            params={"overwrite": overwrite},
            files={"attachment": (filename, content, mime_type)},
            # The endpoint takes the metadata as a form field holding a JSON document, so it
            # is serialized once, here at the boundary.
            data={"metadata": metadata.model_dump_json()},
        )
        return self._parse(GenericRagDocument, response, "document upload")

    async def delete_document(self, document_id: int) -> None:
        await self._request("document deletion", "DELETE", f"/documents/{document_id}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ transport ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _request(self, operation: str, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self._base_url}/channel{path}"
        try:
            response = await self._http.client.request(
                method, url, headers={"Api-Key": self._api_key.get_secret_value()}, **kwargs
            )
        except httpx.HTTPError as e:
            raise GenericRagIngestionError(operation, f"{type(e).__name__}: {e}") from e

        if response.is_success:
            return response

        _log.warning(f"Generic RAG {operation} at {url} returned HTTP {response.status_code}")
        raise GenericRagIngestionError(operation, self._detail(response.text), response.status_code)

    @staticmethod
    def _parse(model: type[_ModelT], response: httpx.Response, operation: str) -> _ModelT:
        """Validate a response body, reporting a surprise as a failure of that operation.

        A body that does not parse means the caller cannot proceed, and it is far more
        useful to name the call than to surface a bare `ValidationError` from a job log.
        """
        try:
            return model.model_validate(response.json())
        except ValueError as e:
            raise GenericRagIngestionError(
                operation, f"unexpected response body: {e}", response.status_code
            ) from e

    @staticmethod
    def _detail(body: str) -> str:
        body = body.strip()
        if len(body) <= _MAX_DETAIL_CHARS:
            return body
        return f"{body[:_MAX_DETAIL_CHARS]}... (truncated)"
