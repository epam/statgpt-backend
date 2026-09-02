"""Wire schemas of the Generic RAG channel API.

Modeled on the application's own `/openapi/channel` spec (v0.2.0), reached through the DIAL
application route `/v1/deployments/{application_id}/route/channel/...`. Only the contract:
what StatGPT puts into such a channel is `discovery_document`.

These models mirror an external contract, so they are plain `BaseModel`s rather than
`BaseYamlModel`s: that base camelizes field names, and this API speaks snake_case
(`display_name`, `total_count`). Every field but the id is defaulted and unknown fields are
ignored, so a new field or status value on the service side does not fail a whole run.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenericRagDocumentStatus(StrEnum):
    """Where a document is in the channel's parse-and-index pipeline.

    Not used as a field type: the service may add values, and a listing must not fail to
    parse because of one. Compare against `GenericRagDocument.status` instead.
    """

    CREATED = "created"
    PROCESSING = "processing"
    PROCESSED = "processed"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class GenericRagDocument(BaseModel):
    """A document stored in a Generic RAG channel."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="ignore")

    id: int
    """Assigned by the service, unique within the channel. There is no client-supplied key."""

    display_name: str = ""
    """User-facing name, taken from the uploaded file name."""

    url: str = ""
    mime_type: str = ""
    size: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Whatever was sent on upload, having satisfied the channel's metadata JSON-schema."""

    status: str = ""
    """One of `GenericRagDocumentStatus`, as a raw string."""

    @property
    def is_failed(self) -> bool:
        """True when the service could not parse or index this document.

        Such a document holds no retrievable content, so a publisher must treat it as
        absent and upload the record again.
        """
        return self.status == GenericRagDocumentStatus.ERROR

    @property
    def is_terminal(self) -> bool:
        """True when the service has finished with this document, one way or the other.

        Everything else - `created`, `processing`, `processed`, `indexing` - means parsing
        or indexing is still under way, so the document's fate is not yet decided and a
        publisher that needs to know has to look again later.
        """
        return self.status in (GenericRagDocumentStatus.READY, GenericRagDocumentStatus.ERROR)


class GenericRagDocumentPage(BaseModel):
    """One page of `GET /channel/documents`."""

    model_config = ConfigDict(extra="ignore")

    total_count: int = 0
    offset: int = 0
    limit: int = 0
    results: list[GenericRagDocument] = Field(default_factory=list)


class GenericRagMetadataSchema(BaseModel):
    """The channel's document-metadata JSON-schema and its filterable dimensions.

    Configured on the DIAL application, not by us: a publisher can only read it back to
    check that the fields it relies on are declared.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="ignore", populate_by_name=True)

    json_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    """The JSON-schema a document's metadata must satisfy.

    Aliased because a field literally named `schema` shadows a `BaseModel` attribute.
    """

    dimensions: dict[str, list[str]] = Field(default_factory=dict)
    """Filterable field names mapped to the values currently present in the channel."""

    @property
    def filterable_fields(self) -> set[str]:
        """Fields a retrieval request may filter on.

        Read from the schema rather than from `dimensions`: a field is declared filterable
        by `enable_filtering`, while `dimensions` only lists the values documents happen to
        carry, so it is empty for a field nothing has populated yet - and for every field of
        an empty channel.
        """
        properties = self.json_schema.get("properties")
        if not isinstance(properties, dict):
            return set()
        return {
            name
            for name, definition in properties.items()
            if isinstance(definition, dict) and definition.get("enable_filtering")
        }


class GenericRagDocumentFilter(BaseModel):
    """One entry of a matcher's `filters`: a document matches when every field set here matches.

    The service builds this model per channel out of the metadata fields declared
    `enable_filtering`, and rejects a field that is not one of them, so only fields
    `DiscoveryDocumentMetadata` marks filterable may be added here. A field left `None` is
    omitted from the request rather than sent as a null.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    statgpt_channel: str | None = None
    """Deployment id of the StatGPT channel that published the document."""


class GenericRagDocumentMatcher(BaseModel):
    """The `matcher` of a search request: which documents the search is allowed to consider.

    A document matches when it satisfies *any* of `filters`; an empty list means every document
    in the channel. Applied before retrieval, so a request's `limit` is spent on matching
    documents alone.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    filters: list[GenericRagDocumentFilter] = Field(default_factory=list)


class GenericRagDocumentSearchRequest(BaseModel):
    """The body of `POST /channel/documents/search`.

    The service builds this schema dynamically per channel, so this is the static subset StatGPT
    sends.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    query: str
    """Free text the indexes are searched with."""

    limit: int = 5
    """Upper bound on results, applied both per index and to the rank-fused list."""

    indexes: list[str] | None = None
    """Which document indexes to search, in the order the stages run.

    `None` leaves the choice to the channel, which uses every index flagged
    `include_in_hybrid`.
    """

    matcher: GenericRagDocumentMatcher | None = None
    """Which documents the search may return. `None` leaves the whole channel in scope."""
