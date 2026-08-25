"""The document a discovery dataset record is published as.

The Generic RAG channel's own wire contract lives in `generic_rag`; this is what StatGPT
puts into it. Kept apart because the two change for different reasons: that one follows the
service's API, this one follows the discovery workbook.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _flags(*, filterable: bool = False, retrievable: bool = False) -> dict[str, Any]:
    """Build the `json_schema_extra` marking what the RAG channel may do with a field.

    A fresh dict per field: Pydantic keeps whatever it is given on the `FieldInfo`, and a
    shared one would be the same object on every field that carries it.

    `filterable` lets a retrieval request pre-filter documents by the field; `retrievable`
    returns it alongside a retrieved chunk.
    """
    extra: dict[str, Any] = {}
    if filterable:
        extra["enable_filtering"] = True
    if retrievable:
        extra["enable_in_mcp_retrieve_chunks"] = True
    return extra


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

    grade: str = Field(json_schema_extra=_flags(filterable=True))
    """Which discovery grade produced the record - see `DiscoveryGrade`.

    Typed as `str` rather than as the enum on purpose: this model renders the JSON-schema an
    external application is configured with, and an enum-typed field renders as a `$ref`
    into a `$defs` block, which is a needlessly brittle thing to ask a deployment to carry.
    """

    statgpt_channel: str = Field(json_schema_extra=_flags(filterable=True))
    """Deployment id of the StatGPT channel the record belongs to.

    Scopes reconciliation: several channels, and both grades, can share one RAG channel, so a
    run must be able to tell its own documents from everyone else's. The deployment id rather
    than the row id, because that is the channel's identity that survives being moved between
    environments.
    """

    agency: str = Field(json_schema_extra=_flags(filterable=True, retrievable=True))
    """Publisher. A search pre-filter key."""

    reference_area: str = Field(
        default="", json_schema_extra=_flags(filterable=True, retrievable=True)
    )
    """Countries covered. A search pre-filter key."""

    frequency_coverage: str = Field(default="", json_schema_extra=_flags(filterable=True))
    """The frequencies the dataset publishes at."""

    dataset_id: str = Field(default="", json_schema_extra=_flags(retrievable=True))
    name: str = Field(default="", json_schema_extra=_flags(retrievable=True))
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
