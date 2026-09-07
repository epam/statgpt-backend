"""The documents StatGPT publishes into a Generic RAG channel.

The Generic RAG channel's own wire contract lives in `generic_rag`; this is what StatGPT
puts into it. Kept apart because the two change for different reasons: that one follows the
service's API, these follow the discovery workbook.

Two kinds of document: a discovery dataset record, and one entry of the reference-area
vocabulary a query's countries are resolved against. They live in different channels and
share only the base class below.
"""

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

REFERENCE_AREA_KIND = "reference_area"
"""What a vocabulary document's `kind` says, and what a run recognizes its own documents by."""


class ReferenceAreaRole(StrEnum):
    """How a record used a reference area: as the data's subject, or as its counterpart.

    The two are separate fields on a discovery document and separate axes at chat time, so a
    vocabulary entry has to say which of them it can serve - a label no record ever names as a
    partner is not a value the partner axis may filter on.

    Only a source of constants: the metadata field itself stays a plain `list[str]`, for the
    reason its docstring gives.
    """

    SUBJECT = "subject"
    PARTNER = "partner"


def _filterable() -> dict[str, Any]:
    """The `json_schema_extra` letting a retrieval request pre-filter documents by a field.

    A fresh dict per field: Pydantic keeps whatever it is given on the `FieldInfo`, and a
    shared one would be the same object on every field that carries it.
    """
    return {"enable_filtering": True}


class ChannelDocumentMetadata(BaseModel):
    """Metadata of one document StatGPT publishes, whatever kind of document it is.

    A subclass is the contract for its channel. The channel enforces its own metadata
    JSON-schema, which is configured on the DIAL application rather than pushed from here, so
    `channel_json_schema()` renders what that application must declare and
    `filterable_fields()` is what a run checks it against - one definition, so the two cannot
    drift apart silently.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_title: ClassVar[str]
    """Title of the rendered JSON-schema, so a configuration says which channel it belongs to."""

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
        schema["title"] = cls.schema_title
        schema["additionalProperties"] = True
        return schema


class DiscoveryDocumentMetadata(ChannelDocumentMetadata):
    """Metadata attached to one discovery dataset document.

    Every workbook field except the description, which the document body carries. Shared by
    both discovery grades, so one index and one search behavior serve both.

    Cells are sent verbatim, as submitted. The ';'-separated lists inside them are additionally
    published as `parsed_` arrays, because a filter can only match a whole value: the array is
    what a request narrows on, while the cell is what a template renders and an index searches.
    """

    schema_title: ClassVar[str] = "DiscoveryDatasetMetadataSchema"

    grade: str = Field(json_schema_extra=_filterable())
    """Which discovery grade produced the record - see `DiscoveryGrade`.

    Typed as `str` rather than as the enum on purpose: this model renders the JSON-schema an
    external application is configured with, and an enum-typed field renders as a `$ref`
    into a `$defs` block, which is a needlessly brittle thing to ask a deployment to carry.
    """

    statgpt_channel: str = Field(json_schema_extra=_filterable())
    """Deployment id of the StatGPT channel the record belongs to.

    Scopes reconciliation: several channels, and both grades, can share one RAG channel, so a
    run must be able to tell its own documents from everyone else's. The deployment id rather
    than the row id, because that is the channel's identity that survives being moved between
    environments.
    """

    agency: str = Field(json_schema_extra=_filterable())
    """Publisher. A search pre-filter key, and a single value already, so it has no array."""

    reference_area: str = Field(default="")
    """Countries covered, as the cell was submitted. Rendered and indexed, not filtered on."""

    parsed_reference_areas: list[str] = Field(default_factory=list, json_schema_extra=_filterable())
    """The subject areas of `reference_area`, one value per entry. A search pre-filter key.

    A plain, non-nullable `list[str]`: the service derives its request model from this schema,
    and both an optional array and an enum-typed one make that derivation raise, which takes
    down every search on the channel rather than just this field.

    Defaulted, like every array here, so a document published before these fields existed
    still parses as a search hit instead of being dropped from the results.
    """

    parsed_partner_reference_areas: list[str] = Field(
        default_factory=list, json_schema_extra=_filterable()
    )
    """The areas `reference_area` marks as partners rather than subjects.

    A separate field rather than a role marker on a shared one: the two are asked about
    differently, and a query naming a country almost always means it as a subject.
    """

    frequency_coverage: str = Field(default="")
    """The frequencies the dataset publishes at, as the cell was submitted."""

    parsed_frequencies: list[str] = Field(default_factory=list, json_schema_extra=_filterable())
    """The frequencies of `frequency_coverage`, folded onto the template's vocabulary."""

    dataset_id: str = ""
    name: str = ""
    url: str = ""
    regional_coverage: str = ""
    excluded_regional_values: str = ""
    time_coverage: str = ""
    indicators_coverage: str = ""
    missing_indicators: str = ""


class ReferenceAreaDocumentMetadata(ChannelDocumentMetadata):
    """Metadata of one reference-area vocabulary document.

    The vocabulary exists so a query's areas can be resolved by search rather than by asking a
    model to pick from a channel's several hundred area labels at once. One document per
    distinct label, whose body is the label itself.

    One document whatever its roles: a label records use both as a subject area and as a
    partner area is published once, naming both. The roles are what lets the two chat-time area
    axes search this one channel separately, each offered only the labels its own field holds.
    """

    schema_title: ClassVar[str] = "ReferenceAreaMetadataSchema"

    kind: str = Field(default=REFERENCE_AREA_KIND, json_schema_extra=_filterable())
    """What this document is. Filterable so the channel can be shared with other kinds later.

    A plain `str` rather than a `Literal`: the service turns this schema into its own request
    model, and a constant renders as an enum there - a needless way to make a single-valued
    field a source of derivation failures.
    """

    statgpt_channel: str = Field(json_schema_extra=_filterable())
    """Deployment id of the StatGPT channel whose records produced this label."""

    roles: list[str] = Field(default_factory=list, json_schema_extra=_filterable())
    """Which of `ReferenceAreaRole` this channel's records use the label in. Sorted.

    A plain, non-nullable `list[str]` rather than a list of the enum, for the same reason
    `kind` is not a `Literal`: the service derives its request model from this schema, and an
    enum-typed array makes that derivation raise on every search of the channel.

    Defaulted, so a document published before roles existed still parses. Such a document
    matches no role filter and is re-uploaded by the next indexing run.
    """

    value: str = ""
    """The label, as the records spell it. Also the document's body."""
