"""What the chat-time discovery datasets lookup works with and reports.

A candidate is one retrieved document reassembled into a whole record: the metadata the search
endpoint returns plus the description it does not. The eval attachment is everything a reviewer
needs to judge a run - what was retrieved, what the model made of it, and what the user saw -
and is deliberately not part of the tool state, which is echoed back on every later turn.
"""

from pydantic import BaseModel, ConfigDict, Field

from statgpt.common.schemas import DiscoveryDocumentMetadata

_TEMPLATE_ONLY_KEYS = ("items",)
"""Placeholders the item template must not claim: `{items}` belongs to the wrapper."""


class DiscoveryCandidate(BaseModel):
    """One retrieved discovery document, ready to be judged and rendered."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int
    """The RAG channel's own id. What the model refers to a candidate by."""

    rank: int
    """1-based position in the search result.

    The only relevance signal the endpoint gives: it fuses the ranks of every index it searched
    and returns no scores.
    """

    display_name: str = ""
    metadata: DiscoveryDocumentMetadata
    description: str = ""
    """The document body. Empty when its download failed - the candidate is still judged."""

    def to_llm_dict(self) -> dict[str, object]:
        """The candidate as the relevance prompt sees it."""
        return {
            "document_id": self.document_id,
            "name": self.metadata.name or self.display_name,
            "agency": self.metadata.agency,
            "reference_area": self.metadata.reference_area,
            "time_coverage": self.metadata.time_coverage,
            "frequency_coverage": self.metadata.frequency_coverage,
            "indicators_coverage": self.metadata.indicators_coverage,
            "missing_indicators": self.metadata.missing_indicators,
            "description": self.description,
        }

    def template_context(self, reason: str = "") -> dict[str, object]:
        """Placeholders available to the item template.

        Every metadata field, plus what only the retrieval knows. Metadata is spread first so a
        field can never shadow `description` or `rank`, and `{items}` is dropped so an item
        template cannot recurse into the wrapper's placeholder.
        """
        context: dict[str, object] = self.metadata.model_dump(mode="json")
        context.update(
            document_id=self.document_id,
            rank=self.rank,
            display_name=self.display_name,
            description=self.description,
            reason=reason,
        )
        for key in _TEMPLATE_ONLY_KEYS:
            context.pop(key, None)
        return context


class DiscoveryRelevanceItem(BaseModel):
    """The model's verdict on one candidate."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    document_id: int = Field(description="`document_id` of the candidate being judged.")
    reason: str = Field(  # Not shown to the user by default.
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
