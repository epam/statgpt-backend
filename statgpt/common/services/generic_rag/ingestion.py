"""Write access to the documents of a Generic RAG channel.

The read side of the Generic RAG integration is `search` (and, for file RAG, chat completions);
this is the other half - what an administrator-triggered indexing run needs to put records into
a channel and take them out again.

Lives in `common` because it is shared: Grade C publishes uploaded workbook records and
Grade B will publish crawled registry records into the same contract, and `common` cannot
import from `admin`.
"""

from typing import Self

from pydantic import BaseModel

from statgpt.common.schemas.generic_rag import GenericRagDocument, GenericRagDocumentPage
from statgpt.common.settings.dial import dial_settings

from .client import BaseGenericRagChannelClient, GenericRagChannelError

GenericRagIngestionError = GenericRagChannelError
"""Kept as the name callers of this module already raise and catch."""

_PAGE_SIZE = 500
"""How many documents to ask for per listing page.

The API declares no maximum and defaults to 25, which would page a few hundred records
into more round trips than the run needs.
"""


class GenericRagIngestionClient(BaseGenericRagChannelClient):
    """The document write endpoints of one Generic RAG channel.

    Documents are identified by a service-assigned integer, never by a key the caller chooses,
    so a caller that needs to find its own document again has to recognize it by the metadata
    `list_documents` returns. An `overwrite` upload is keyed on the file name rather than on
    that id, which is the closest thing to an upsert the channel offers.
    """

    @classmethod
    def for_application(cls, application_id: str) -> Self:
        """Target the channel of a DIAL application, by application id.

        Authenticates with the deployment's own DIAL key rather than a user token: this runs
        in a background job, which has no user, and the Generic RAG application forwards an
        auth token only on its chat path.
        """
        return cls(base_url=cls.application_route(application_id), api_key=dial_settings.api_key)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ endpoints ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    async def update_document(
        self,
        document_id: int,
        *,
        filename: str,
        content: bytes,
        mime_type: str,
        metadata: BaseModel,
    ) -> GenericRagDocument:
        """Replace a document's content and metadata, keeping the document itself.

        The closest thing to an in-place refresh the channel offers, and the reason to prefer
        it over deleting and re-uploading: the document keeps its id, and it is never briefly
        absent from the channel.

        What it cannot do is rename. The service writes the new content to the url the
        document already has and leaves `display_name` alone, so `filename` travels only as
        the multipart part name. A caller that needs the stored name to change has to delete
        and upload instead.

        The content is re-indexed only if it actually differs - the service compares etags -
        so refreshing an unchanged record is cheap.
        """
        response = await self._request(
            "document update",
            "PUT",
            f"/documents/{document_id}",
            files={"attachment": (filename, content, mime_type)},
            data={"metadata": metadata.model_dump_json()},
        )
        return self._parse(GenericRagDocument, response, "document update")

    async def delete_document(self, document_id: int) -> None:
        """Remove a document, its file, its chunks and its entries in every index."""
        await self._request("document deletion", "DELETE", f"/documents/{document_id}")
