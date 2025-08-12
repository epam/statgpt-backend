from __future__ import annotations

import itertools as it
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMSelectionCandidateBase(BaseModel, ABC):
    """
    Base abstraction to separate logic of:
    - storing candidiates extracted from vector storage,
    - and using these candidates in LLM selection chains.
    Specifically, it:
    - provides '_id' field not tied to id used in extracted candidates model (query_id).
      it could contain any arbitrary value, like simple index
    - allows to specify different string formatting logic (to place candidates in LLM prompt)
      from the same extracted candidates model (containing data)
      by using different subclasses of this model.
    """

    @property
    @abstractmethod
    def _id(self) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def candidates_to_llm_string(candidates: list[LLMSelectionCandidateBase]) -> str:
        raise NotImplementedError


class BatchedSelectionOutputBase(BaseModel, ABC):
    @classmethod
    @abstractmethod
    def combine_batch_outputs(cls, batch_outputs: list[t.Self]) -> t.Self:
        raise NotImplementedError

    @abstractmethod
    def get_selected_ids(self) -> set[str]:
        raise NotImplementedError


class SelectedCandidates(BatchedSelectionOutputBase):
    ids: list[str] = Field(
        description="The IDs of the relevant items. Could be empty if there are no relevant items"
    )

    @classmethod
    def combine_batch_outputs(cls, batch_outputs: list[t.Self]) -> t.Self:
        ids_flat = list(it.chain.from_iterable([candidate.ids for candidate in batch_outputs]))
        return cls(ids=ids_flat)

    def get_selected_ids(self) -> set[str]:
        return set(self.ids)


class CandidatesRelevancyMapping(BatchedSelectionOutputBase):
    id_2_relevancy: dict[str, int] = Field(
        description=(
            "Mapping from candidate ID to binary relevancy score: 1 if relevant, 0 otherwise."
        )
    )

    @classmethod
    def combine_batch_outputs(cls, batch_outputs: list[t.Self]) -> t.Self:
        id_2_relevancy = {}
        for candidate in batch_outputs:
            id_2_relevancy.update(candidate.id_2_relevancy)
        return cls(id_2_relevancy=id_2_relevancy)

    def get_selected_ids(self) -> set[str]:
        return {k for k, v in self.id_2_relevancy.items() if v == 1}
