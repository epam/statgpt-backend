"""Schemas of the Grade C discovery read path.

Three groups: what StatGPT sends the Generic RAG application to retrieve with, what a retrieved
candidate looks like once chunks are folded back into datasets, and what the relevance judge
returns.

The request models are deliberately separate from the publications ones in `file_rags/`. Those
model `publication_type` and `publication_date` because that is what a publications channel
filters on; discovery filters on the fields `DiscoveryDocumentMetadata` declares. Sharing one
model would mean generalizing the publications request payload, which buys nothing here and puts
a working retrieval path at risk.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from statgpt.common.schemas.generic_rag import GenericRagDocument

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ retrieval request ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DiscoveryFilterEntry(BaseModel):
    """One metadata filter entry.

    The service AND's the fields set within an entry and OR's the entries of the list, so a
    country union is one entry per country, each carrying the same grade and channel.

    `grade` and `statgpt_channel` are on every entry rather than hoisted out because there is
    nowhere to hoist them to: the filter list is the only structure the service offers, and an
    entry missing them would match another channel's documents.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    grade: str
    statgpt_channel: str

    reference_area_values: str | None = None
    """One published country value, or None to leave the country axis unfiltered.

    A single string even though the document's field is a list: the service matches an array
    field by containment, so one value here matches any document whose list contains it.
    """


class DiscoveryDocumentSelector(BaseModel):
    """The `explicit` document selector - filters supplied per request."""

    type: Literal["explicit"] = "explicit"
    filters: list[DiscoveryFilterEntry] = Field(default_factory=list)


class DiscoveryRetrieverConfig(BaseModel):
    """The retriever override.

    `type` is omitted so the application keeps whatever retriever it is configured with; only the
    document selector is being overridden.
    """

    document_selector: DiscoveryDocumentSelector


class DiscoveryGenerationConfig(BaseModel):
    """Ask for retrieval without an answer.

    `retrieval_only` is the application's `RetrievalOnlyAnswerGenerator`, which returns the
    retrieval results as attachments and never calls an LLM. Discovery owns its own selection
    step, so a generated answer would be a second, unowned prompt between the record and the
    user - and prose generated from dataset descriptions is exactly the hallucination a referral
    must not carry.
    """

    type: Literal["retrieval_only"] = "retrieval_only"


class DiscoverySearchConfiguration(BaseModel):
    """The `custom_fields.configuration` payload of a discovery retrieval request."""

    retriever: DiscoveryRetrieverConfig
    generation: DiscoveryGenerationConfig = Field(default_factory=DiscoveryGenerationConfig)

    def as_extra_body(self) -> dict:
        """Render the payload the chat-completions call carries."""
        return {"custom_fields": {"configuration": self.model_dump(mode="json", exclude_none=True)}}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ candidates ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DiscoveryCandidate(BaseModel):
    """One discovery dataset, folded from the chunks that matched it.

    Every workbook field is present, not only the ones a chunk happened to contain: the judge
    reads the two negative fields to rule a dataset out, and the referral needs the URL whichever
    section of the record matched.

    The fields come from the document's metadata rather than from the retrieval response. The
    application's attachments carry the chunk text, the document's display name, and a URL
    pointing at the stored markdown file - never the document metadata - so the metadata is
    recovered from the channel's document listing.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int
    display_name: str

    agency: str = ""
    dataset_id: str = ""
    name: str = ""
    url: str = ""
    reference_area: str = ""
    regional_coverage: str = ""
    excluded_regional_values: str = ""
    time_coverage: str = ""
    frequency_coverage: str = ""
    indicators_coverage: str = ""
    missing_indicators: str = ""

    chunks: list[str] = Field(default_factory=list)
    """Text of the chunks that matched, in retrieval order. Shown to the judge, not to the user."""

    @property
    def label(self) -> str:
        """How the dataset is named in a referral and in the judge's candidate list."""
        return self.name or self.dataset_id or self.display_name

    @classmethod
    def from_document(cls, document: GenericRagDocument, chunks: list[str]) -> "DiscoveryCandidate":
        """Build a candidate from a listed document and the chunks that matched it.

        Metadata values are read defensively: the channel echoes back whatever was uploaded, and
        a document published before a field existed simply lacks it.
        """
        metadata = document.metadata

        def _text(key: str) -> str:
            value = metadata.get(key)
            return value if isinstance(value, str) else ""

        return cls(
            document_id=document.id,
            display_name=document.display_name,
            agency=_text("agency"),
            dataset_id=_text("dataset_id"),
            name=_text("name"),
            url=_text("url"),
            reference_area=_text("reference_area"),
            regional_coverage=_text("regional_coverage"),
            excluded_regional_values=_text("excluded_regional_values"),
            time_coverage=_text("time_coverage"),
            frequency_coverage=_text("frequency_coverage"),
            indicators_coverage=_text("indicators_coverage"),
            missing_indicators=_text("missing_indicators"),
            chunks=chunks,
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ judge ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DiscoverySelection(BaseModel):
    """One dataset the judge chose to surface."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    index: int = Field(description="1-based number of the dataset in the candidate list.")
    """Referenced by number rather than by id, so a misspelled id cannot lose a selection."""

    reason: str = Field(
        default="",
        description=(
            "One sentence on why this dataset answers the question, grounded in the text of the"
            " record itself."
        ),
    )
    missing: str = Field(
        default="",
        description=(
            "What the record's own text says the dataset does not contain, when that touches what"
            " was asked. Empty when nothing relevant is missing."
        ),
    )


class DiscoveryJudgement(BaseModel):
    """The judge's verdict over one candidate list."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    selections: list[DiscoverySelection] = Field(
        default_factory=list,
        description=(
            "The datasets worth showing, most relevant first. Empty when none of the candidates"
            " is relevant to the question."
        ),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class DiscoveryReferralItem(BaseModel):
    """A selected dataset together with why it was selected."""

    candidate: DiscoveryCandidate
    reason: str = ""
    missing: str = ""


class DiscoverySearchResult(BaseModel):
    """What a discovery search produced, including enough to explain an empty result."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    items: list[DiscoveryReferralItem] = Field(default_factory=list)
    """The datasets to refer to, in the judge's order. Empty means refer to nothing."""

    grounded_areas: list[str] = Field(default_factory=list)
    """Country values the filter was narrowed to. Empty means the search was unfiltered."""

    unmatched_areas: list[str] = Field(default_factory=list)
    """Countries the request named that the channel holds no value for."""

    retrieved: int = 0
    """Distinct datasets retrieved before the judge ran. Separates "found nothing" from
    "found things and rejected them all"."""

    @property
    def has_referral(self) -> bool:
        return bool(self.items)
