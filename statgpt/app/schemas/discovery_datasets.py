"""What the chat-time discovery datasets lookup works with and reports.

A candidate is one retrieved document reassembled into a whole record: the metadata the search
endpoint returns plus the description it does not. The eval attachment is what a reviewer needs
to judge a run, and is deliberately not part of the tool state echoed back on later turns.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from statgpt.common.schemas import DiscoveryDocumentMetadata, DiscoveryPreFilterAxis

_PUBLISHING_METADATA_FIELDS = frozenset({"grade", "statgpt_channel"})
"""Metadata that scopes publishing rather than describing a dataset.

Kept out of both the prompt and the template context: it means nothing to a judge or a reader.
"""


class DiscoveryCandidateForLlm(BaseModel):
    """One candidate as the relevance prompt sees it.

    These keys are the prompt's contract: the default relevance prompt names them, and they are
    what the YAML handed to the judge contains.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int
    """What the model refers to a candidate by, and echoes back in its verdict."""

    name: str
    agency: str
    reference_area: str
    time_coverage: str
    frequency_coverage: str
    indicators_coverage: str
    missing_indicators: str
    description: str


class DiscoveryCandidate(BaseModel):
    """One retrieved discovery document, ready to be judged and rendered."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int
    """The RAG channel's own id. What the model refers to a candidate by."""

    rank: int
    """1-based position among the documents this channel published.

    The only relevance signal the endpoint gives - it returns no scores. Counted over the hits
    that survived retrieval, so the numbers a user reads have no gaps.
    """

    display_name: str = ""
    metadata: DiscoveryDocumentMetadata
    description: str = ""
    """The document body. Empty when its download failed - the candidate is still judged."""

    def for_llm(self) -> DiscoveryCandidateForLlm:
        """The candidate as the relevance prompt sees it."""
        return DiscoveryCandidateForLlm(
            document_id=self.document_id,
            name=self.metadata.name or self.display_name,
            agency=self.metadata.agency,
            reference_area=self.metadata.reference_area,
            time_coverage=self.metadata.time_coverage,
            frequency_coverage=self.metadata.frequency_coverage,
            indicators_coverage=self.metadata.indicators_coverage,
            missing_indicators=self.metadata.missing_indicators,
            description=self.description,
        )

    def template_context(self, reason: str = "") -> dict[str, Any]:
        """Placeholders available to the item template.

        Every metadata field that describes the dataset, plus what only the retrieval knows.
        Metadata is spread first so a field can never shadow `description` or `rank`.
        """
        context: dict[str, Any] = self.metadata.model_dump(
            mode="json", exclude=set(_PUBLISHING_METADATA_FIELDS)
        )
        context.update(
            document_id=self.document_id,
            rank=self.rank,
            display_name=self.display_name,
            description=self.description,
            reason=reason,
        )
        return context


class DiscoveryAxisSelection(BaseModel):
    """One pre-filter sub-chain's structured output: the values the query names on its axis."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    values: list[str] = Field(
        default_factory=list,
        description=(
            "Values from the provided list that the user's query asks for on this axis, copied"
            " exactly. Empty when the query does not restrict this axis."
        ),
    )


class DiscoveryPreFilterAxisReport(BaseModel):
    """What one axis of the pre-filter offered, chose and kept.

    All three lists are recorded, because they answer different questions: `offered` says what
    the model could have picked, `selected` what it did pick, and `grounded` what the discovery
    channel actually holds - a value that appears in `selected` alone is one that would have
    failed the search request had it been sent.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    axis: DiscoveryPreFilterAxis
    offered: list[str] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list)
    grounded: list[str] = Field(default_factory=list)
    error: str | None = None
    """Why this axis had nothing to offer: unconfigured, or a vocabulary that could not be read.

    Not set when the axis was offered a vocabulary and the query simply named none of it - that
    is the ordinary case, and `selected` already says so.
    """


class DiscoveryPreFilterReport(BaseModel):
    """What the pre-filter did to one lookup's search.

    Emitted whether or not the search was narrowed, so a turn that fell back is as visible as
    one that filtered.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    enabled: bool = True
    """Whether the channel asks for a pre-filter at all."""

    axes: list[DiscoveryPreFilterAxisReport] = Field(default_factory=list)

    filters: int = 0
    """How many filter entries the matcher carried: the product of the surviving axes."""

    applied: bool = False
    """Whether the documents the judge saw came from the narrowed search or from the fallback."""

    fallback_reason: str | None = None
    """Why the lookup fell back to searching every document this channel published."""


class SelectedDiscoveryDataset(BaseModel):
    """A candidate the relevance judge kept, with the reason it gave for keeping it."""

    candidate: DiscoveryCandidate
    reason: str = ""


class DiscoveryRelevanceItem(BaseModel):
    """The model's verdict on one candidate."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int = Field(description="`document_id` of the candidate being judged.")
    reason: str = Field(
        default="",
        description="One sentence explaining why this document may or may not answer the user's question.",
    )
    relevant: bool = Field(description="Whether this dataset is worth showing for the query.")


class DiscoveryRelevanceResponse(BaseModel):
    """The relevance judge's structured output."""

    items: list[DiscoveryRelevanceItem] = Field(
        default_factory=list, description="One entry per candidate, in any order."
    )


class DiscoveryDatasetsEvalAttachment(BaseModel):
    """Everything one lookup did, for offline evaluation.

    Emitted whether or not the lookup produced anything, so a run that found nothing - or
    failed - is as visible as one that succeeded.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    query: str = ""
    """The data query tool argument the lookup searched with."""

    pre_filter: DiscoveryPreFilterReport | None = None
    """How the search was narrowed, or `None` when the lookup never got that far."""

    candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    """Retrieved documents in rank order, descriptions included."""

    llm_response: DiscoveryRelevanceResponse | None = None
    """The judge's verdicts, or `None` when it was never called."""

    selected_document_ids: list[int] = Field(default_factory=list)
    rendered: str | None = None
    """Exactly what was appended to the data query response, or `None` if nothing was."""

    error: str | None = None
    """Why the lookup produced nothing, when it was a failure rather than an empty result."""


class DiscoveryDatasetsOutcome(BaseModel):
    """What the runner hands back to the data query tool."""

    rendered: str | None = None
    """The block to append to the tool response, or `None` to append nothing."""

    eval_attachment: DiscoveryDatasetsEvalAttachment = Field(
        default_factory=DiscoveryDatasetsEvalAttachment
    )
