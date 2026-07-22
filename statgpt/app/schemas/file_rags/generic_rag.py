from __future__ import annotations

import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field

from .dial_rag import RagFilterDial, SortOrder


class GenericDateInterval(BaseModel):
    start: datetime.date | None = None
    end: datetime.date | None = None


class GenericSingleFilter(BaseModel):
    """A single metadata filter. A document must satisfy all set criteria to match."""

    publication_type: str | None = None
    publication_date: GenericDateInterval | None = None


def _default_sort_by() -> list[Literal["publication_date"]]:
    return ["publication_date"]


class GenericTopN(BaseModel):
    sort_by: list[Literal["publication_date"]] = Field(
        default_factory=_default_sort_by, min_length=1
    )
    order: SortOrder = SortOrder.desc
    limit: int = Field(gt=0)


class GenericExplicitDocumentSelector(BaseModel):
    type: Literal["explicit"] = "explicit"
    filters: list[GenericSingleFilter] = Field(default_factory=list)
    top_n: GenericTopN | None = None


class GenericRetrieverConfig(BaseModel):
    # NOTE: the retriever `type` is intentionally omitted so the generic-rag
    # application falls back to its own configured default retriever.
    document_selector: GenericExplicitDocumentSelector


class GenericRagConfiguration(BaseModel):
    """Partial `custom_fields.configuration` payload sent to the Generic RAG application.

    Only the fields we need to override are set; everything else (retriever type,
    answer generation, top_k) is merged from the application's server-side defaults.
    """

    retriever: GenericRetrieverConfig

    @classmethod
    def from_rag_filter_dial(cls, rag_filter: RagFilterDial) -> Self:
        filters = [
            GenericSingleFilter(
                publication_type=single.publication_type,
                publication_date=(
                    GenericDateInterval(
                        start=single.publication_date.start,
                        end=single.publication_date.end,
                    )
                    if single.publication_date is not None
                    else None
                ),
            )
            for single in rag_filter.filters
        ]

        top_n = (
            GenericTopN(
                sort_by=list(rag_filter.top_n.sort_by),
                order=rag_filter.top_n.order,
                limit=rag_filter.top_n.limit,
            )
            if rag_filter.top_n is not None
            else None
        )

        return cls(
            retriever=GenericRetrieverConfig(
                document_selector=GenericExplicitDocumentSelector(filters=filters, top_n=top_n),
            )
        )
