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


FILTERABLE: dict[str, Any] = {"enable_filtering": True}
"""Marks a metadata field the RAG channel must be able to filter documents by."""

RETRIEVABLE: dict[str, Any] = {"enable_in_mcp_retrieve_chunks": True}
"""Marks a metadata field worth returning alongside a retrieved chunk."""

FILTERABLE_AND_RETRIEVABLE: dict[str, Any] = FILTERABLE | RETRIEVABLE


class DiscoveryDocumentMetadata(BaseModel):
    """Metadata attached to one discovery dataset document.

    Every workbook field except the description, which the document body carries. Shared by
    both discovery grades, so one index and one search behavior serve both.

    Values are sent verbatim, as submitted: the ';'-separated lists inside cells are not
    parsed here any more than they are in the database.

    This model is the contract. The RAG channel enforces its own metadata JSON-schema, which
    is configured on the DIAL application rather than pushed from here, so
    `channel_json_schema()` renders what that application must declare and
    `filterable_fields()` is what a run checks it against - one definition, so the two cannot
    drift apart silently.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    grade: str = Field(json_schema_extra=FILTERABLE)
    """Which discovery grade produced the record - see `DiscoveryGrade`."""

    statgpt_channel: str = Field(json_schema_extra=FILTERABLE)
    """Deployment id of the StatGPT channel the record belongs to.

    Scopes reconciliation: several channels, and both grades, can share one RAG channel, so a
    run must be able to tell its own documents from everyone else's. The deployment id rather
    than the row id, because that is the channel's identity that survives being moved between
    environments.
    """

    agency: str = Field(json_schema_extra=FILTERABLE_AND_RETRIEVABLE)
    """Publisher. A search pre-filter key."""

    reference_area: str = Field(default="", json_schema_extra=RETRIEVABLE)
    """Countries covered, verbatim as submitted. Displayed in a referral.

    Not the filter axis. The cell is free text - one country, a ';'-separated list, or a group
    label - so equality against the whole string never matches a question about one member of a
    multi-country dataset. `reference_area_values` carries the axis a filter can match.
    """

    reference_area_values: list[str] = Field(default_factory=list, json_schema_extra=FILTERABLE)
    """The country pre-filter axis: `reference_area` split into its entries.

    An array rather than a string because the service matches an array field by containment, so
    one filter value reaches every dataset whose list holds it - which is what makes a
    multi-country dataset findable from a question about one of its countries.

    Built by `parse_reference_area`, which also stamps the sentinel that keeps a record with no
    country scope reachable from any country's question.
    """

    frequency_coverage: str = Field(default="", json_schema_extra=FILTERABLE)
    """The frequencies the dataset publishes at."""

    dataset_id: str = Field(default="", json_schema_extra=RETRIEVABLE)
    name: str = Field(default="", json_schema_extra=RETRIEVABLE)
    url: str = ""
    regional_coverage: str = ""
    excluded_regional_values: str = ""
    time_coverage: str = ""
    indicators_coverage: str = ""
    missing_indicators: str = ""

    @classmethod
    def filterable_fields(cls) -> set[str]:
        """Fields the target channel has to declare filterable for search to work."""
        return {
            name
            for name, field in cls.model_fields.items()
            if isinstance(field.json_schema_extra, dict)
            and field.json_schema_extra.get("enable_filtering")
        }

    @classmethod
    def channel_json_schema(cls) -> dict[str, Any]:
        """The metadata JSON-schema to configure on the Generic RAG application.

        Pydantic's own schema, minus what only means something here: the titles and the
        docstrings this module keeps for its own readers, which would otherwise land in a
        deployment's configuration as several paragraphs of internal rationale.
        `additionalProperties` stays open, so a field added here still reaches an application
        whose configuration has not caught up yet.
        """
        schema = cls.model_json_schema()
        for definition in schema.get("properties", {}).values():
            for noise in ("title", "description", "default"):
                definition.pop(noise, None)
        schema.pop("description", None)
        schema["title"] = "DiscoveryDatasetMetadataSchema"
        schema["additionalProperties"] = True
        return schema
