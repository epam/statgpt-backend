""" Retrieval Augmented Matcher (RAM) chain """

from abc import ABC, abstractmethod
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.vectorstore.base import VectorStore


class RetrievalAugmentedMatcherFactoryBase(ABC):
    """
    Retrieval Augmented Matcher (RAM) chain base class.

    Used to select number of items from a long list of available items,
    when putting all available items to LLM context is unreasonable:
    either context will get too large, or LLM will not properly attend for all items.
    """

    class Keys:
        QUERY = 'ram_query'
        CANDIDATES = 'ram_candidates'
        CANDIDATES_STR = 'ram_candidates_str'
        LLM_RESPONSE_RAW = 'ram_llm_response_raw'
        LLM_RESPONSE_PARSED = 'ram_llm_response_parsed'
        OUTPUT = 'ram_output'

    KEYS_TO_CLEANUP = {Keys.CANDIDATES, Keys.LLM_RESPONSE_PARSED}

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: ChatPromptTemplate,
        output_parser: BaseOutputParser,
        vector_store: VectorStore,
        top_n: int,
    ):
        self._llm = llm
        self._output_parser = output_parser
        self._vector_store = vector_store
        self._top_n = top_n
        self._prompt_template = prompt_template

    async def _vector_search(self, d: dict):
        query = d[self.Keys.QUERY]
        candidates = await self._vector_store.search_with_similarity_score(
            query=query, k=self._top_n
        )
        d[self.Keys.CANDIDATES] = candidates
        return d

    @classmethod
    @abstractmethod
    def _format_candidates_for_llm_prompt(cls, candidates: list[tuple[Document, float]]) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _grounding(cls, d: dict):
        """Perform grounding, i.e. remove hallucinations from LLM selections"""
        raise NotImplementedError

    @classmethod
    def _cleanup(cls, d: dict):
        """Remove intermediate chain parameters in the end of the chain"""
        d = {k: v for k, v in d.items() if k not in cls.KEYS_TO_CLEANUP}
        return d

    def create_chain(self) -> Runnable:
        return (
            self._vector_search
            | RunnablePassthrough.assign(
                **{
                    self.Keys.CANDIDATES_STR: lambda d: self._format_candidates_for_llm_prompt(
                        candidates=d[self.Keys.CANDIDATES]
                    )
                }
            )
            | RunnablePassthrough.assign(
                **{
                    self.Keys.LLM_RESPONSE_RAW: self._prompt_template
                    | self._llm
                    | StrOutputParser()
                }
            )
            | RunnablePassthrough.assign(
                **{
                    self.Keys.LLM_RESPONSE_PARSED: itemgetter(self.Keys.LLM_RESPONSE_RAW)
                    | self._output_parser
                }
            )
            | self._grounding
            | self._cleanup
        )
