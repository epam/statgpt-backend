from __future__ import annotations

import typing as t

from aidial_sdk.chat_completion import Stage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough
from pydantic import Field

import statgpt.chains.data_query.v2.query.utils as query_utils
from common.config import multiline_logger as logger
from common.data.base import DataSet
from common.data.base.query import DimensionQuery
from common.schemas import LLMModelConfig
from common.schemas.data_query_tool import SpecialDimensionsProcessor
from common.utils.models import get_chat_model
from common.utils.timer import debug_timer
from statgpt.config import ChainParameters, StateVarsConfig
from statgpt.schemas import LLMSelectionCandidateBase, SelectedCandidates
from statgpt.schemas.query_builder import (
    DatasetAvailabilityQueriesType,
    LLMSelectionDimensionCandidate,
    SpecialDimensionChainOutput,
)
from statgpt.services import ScoredDimensionCandidate
from statgpt.utils.callbacks import StageCallback

from .base import SpecialDimensionChainFactoryBase


def processor_stage_name_prefix(processor: SpecialDimensionsProcessor) -> str:
    return f'Special Dimension ({processor.id}, type: {processor.type})'


class LLMCandidateLHCL(LLMSelectionDimensionCandidate):
    # override `dedup_key` to only consider `name` field for deduplication
    dedup_key: t.ClassVar[tuple[str, ...]] = ('name',)

    @classmethod
    def candidates_to_llm_string(cls, candidates: list[LLMCandidateLHCL]) -> str:
        # do not include `dimension` column
        df = cls._candidates_to_df(candidates)
        df.drop_duplicates(cls.dedup_key, inplace=True)
        df.sort_values('score', ascending=False, inplace=True)
        return cls._format_df(df)


class LHCLChainState(ChainParameters):
    # data query fields
    normalized_query: str = Field(default="", description="Summarized conversation")
    datasets_dict: dict[str, DataSet] = {}
    # availability queries
    weak_queries: DatasetAvailabilityQueriesType = {}
    strong_queries: DatasetAvailabilityQueriesType = {}
    # candidates
    vector_candidates: list[ScoredDimensionCandidate] = []
    llm_candidates: list[LLMSelectionDimensionCandidate] = []
    llm_response: SelectedCandidates | None = None  # could be None only during initialization

    def get_llm_response(self) -> SelectedCandidates:
        if self.llm_response is None:
            raise ValueError("llm_response is not expected to be None")
        return self.llm_response


class LHCLInnerSelectionChainFactory:
    def __init__(
        self,
        llm_model_config: LLMModelConfig,
        system_prompt: str,
        user_prompt: str,
        candidates_key: str,
        processor: SpecialDimensionsProcessor,
    ):
        super().__init__()
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._candidates_key = candidates_key
        self._processor = processor

    def _get_candidates(self, inputs: dict) -> list[LLMSelectionCandidateBase]:
        return inputs[self._candidates_key]

    def _get_llm_response(self, inputs: dict) -> SelectedCandidates:
        return inputs["llm_response"]

    def _format_candidates(self, inputs: dict) -> str:
        candidates = self._get_candidates(inputs)
        if not candidates:
            return ''
        # NOTE: we assume all candidates are of the same type
        text = candidates[0].candidates_to_llm_string(candidates)
        return text

    def _get_candidates_formatted(self, inputs: dict) -> str:
        return inputs['candidates_formatted']

    def _display_formatted_candidates_in_stage(self, inputs: dict) -> dict:
        chain_state = LHCLChainState(**inputs)
        show_debug_stages = chain_state.state.get(StateVarsConfig.SHOW_DEBUG_STAGES)

        if not show_debug_stages:
            return inputs

        stage_name = (
            f'[DEBUG] {processor_stage_name_prefix(self._processor)}, Candidates, formatted for LLM'
        )
        with chain_state.choice.create_stage(name=stage_name) as stage:
            candidates_formatted = self._get_candidates_formatted(inputs)
            content = f'```yaml\n{candidates_formatted}\n```'
            stage.append_content(content)

        return inputs

    def _remove_hallucinations(self, inputs: dict) -> dict:
        candidates = self._get_candidates(inputs)
        llm_response = self._get_llm_response(inputs)

        candidates_ids = {x._id for x in candidates}
        response_ids = set(llm_response.ids)

        hallucinations = response_ids.difference(candidates_ids)
        if hallucinations:
            logger.warning(
                f"!HALLUCINATION in Selection chain! "
                f"{len(hallucinations)} unexpected ids found: {hallucinations}"
            )
            inputs["llm_response"] = SelectedCandidates(
                ids=list(response_ids.intersection(candidates_ids))
            )
        return inputs

    def _display_llm_response_in_stage(self, inputs: dict) -> dict:
        chain_state = LHCLChainState(**inputs)
        show_debug_stages = chain_state.state.get(StateVarsConfig.SHOW_DEBUG_STAGES)

        if not show_debug_stages:
            return inputs

        stage_name = (
            f'[DEBUG] {processor_stage_name_prefix(self._processor)}, LLM Response (grounded)'
        )
        with chain_state.choice.create_stage(name=stage_name) as stage:
            llm_response = self._get_llm_response(inputs)
            content = f'```json\n{llm_response.model_dump_json(indent=2)}\n```'
            stage.append_content(content)

        return inputs

    def _route_based_on_candidates_presence(self, inputs: dict) -> Runnable | SelectedCandidates:
        chain_state = LHCLChainState(**inputs)

        candidates = self._get_candidates(inputs)
        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. '
                'Will return empty list of selected ids.'
            )
            return SelectedCandidates(ids=[])

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", self._user_prompt),
            ],
        )

        llm = get_chat_model(
            api_key=chain_state.auth_context.api_key,
            model_config=self._llm_model_config,
        ).with_structured_output(schema=SelectedCandidates, method='json_schema')
        # NOTE: can experiment with output model having 'reasoning' field to improve quality

        chain = (
            RunnablePassthrough.assign(candidates_formatted=self._format_candidates)
            | self._display_formatted_candidates_in_stage
            | RunnablePassthrough.assign(llm_response=prompt_template | llm)
            | self._remove_hallucinations
            | self._display_llm_response_in_stage
            | self._get_llm_response
        )

        return chain

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)


class LHCLChainFactory(SpecialDimensionChainFactoryBase):
    """
    Chain factory to select terms from Long Hierarchical Code List dimension type.
    """

    def __init__(self, processor: SpecialDimensionsProcessor):
        super().__init__(processor)
        self._top_k = processor.top_k
        self._llm_selection_chain = LHCLInnerSelectionChainFactory(
            llm_model_config=processor.llm_model_config,
            system_prompt=processor.prompt.system_message,
            user_prompt=processor.prompt.user_message,
            candidates_key='llm_candidates',
            processor=self._processor,
        ).create_chain()

    async def _get_candidates_from_vector_store(
        self, inputs: dict
    ) -> list[ScoredDimensionCandidate]:
        chain_state = LHCLChainState(**inputs)
        data_service = chain_state.data_service
        query = chain_state.normalized_query
        # NOTE: use datasets from current version of strong queries (after nonindicators selection)
        # as filter for vector search
        datasets = set(chain_state.strong_queries.keys())

        with debug_timer(f"{self._name}._get_candidates_from_vector_store"):
            candidates_all = await data_service.search_special_dimension_scored(
                query,
                special_dimension_processor=self._processor,
                auth_context=chain_state.auth_context,
                datasets=datasets,
                k=self._top_k,
            )

        candidates_dedup = list(set(candidates_all))
        candidates_dedup = sorted(candidates_dedup, key=lambda x: x.score, reverse=True)

        return candidates_dedup

    def _format_final_response(self, inputs: dict) -> SpecialDimensionChainOutput:
        chain_state = LHCLChainState(**inputs)
        strong_queries = chain_state.strong_queries

        dataset_queries = {}
        for ds_id, availability_query in strong_queries.items():
            dims_dict = availability_query.dimensions_queries_dict
            if not dims_dict:
                continue
            if len(dims_dict) != 1:
                raise ValueError(
                    'expected each dataset having exactly one '
                    f'{self._processor.id} special dimension. found: {dims_dict.keys()}'
                )
            dim, query = next(iter(dims_dict.items()))
            dim_query = DimensionQuery.from_query(query=query, dimension_id=dim)
            dataset_queries[ds_id] = dim_query

        res = SpecialDimensionChainOutput(
            processor_id=self._processor.id,
            processor_type=self._processor.type,
            dataset_queries=dataset_queries,
            llm_response=chain_state.get_llm_response(),
        )
        return res

    def _parse_chain_state(self, inputs: dict) -> dict:
        """keep only required fields from input dict and drop others for clarity"""
        return LHCLChainState(**inputs).model_dump()

    @staticmethod
    async def _populate_weak_queries_stage(stage: Stage, inputs: dict) -> None:
        with debug_timer('LHCL._populate_weak_queries_stage'):
            chain_state = LHCLChainState(**inputs)
            await query_utils.populate_queries_stage(
                stage=stage,
                queries=chain_state.weak_queries,
                auth_context=chain_state.auth_context,
                datasets_dict=chain_state.datasets_dict,
            )

    @staticmethod
    async def _populate_strong_queries_stage(stage: Stage, inputs: dict) -> None:
        with debug_timer('LHCL._populate_strong_queries_stage'):
            chain_state = LHCLChainState(**inputs)
            await query_utils.populate_queries_stage(
                stage=stage,
                queries=chain_state.strong_queries,
                auth_context=chain_state.auth_context,
                datasets_dict=chain_state.datasets_dict,
            )

    @staticmethod
    def _prepare_llm_candidates(inputs: dict) -> list[LLMCandidateLHCL]:
        chain_state = LHCLChainState(**inputs)
        vector_candidates = chain_state.vector_candidates
        res = [
            LLMCandidateLHCL.from_scored_dimension_candidate(candidate=c, index=ix)
            for ix, c in enumerate(vector_candidates)
        ]
        return res

    @staticmethod
    def _filter_candidates_by_llm_response(inputs: dict) -> list[ScoredDimensionCandidate]:
        chain_state = LHCLChainState(**inputs)
        llm_candidates = chain_state.llm_candidates

        selected_ids = chain_state.get_llm_response().get_selected_ids()
        selected_ids_expanded = LLMCandidateLHCL.propagate_selection_status_to_duplicates(
            candidates=llm_candidates, selected_ids=selected_ids
        )

        filtered = [c for c in llm_candidates if c._id in selected_ids_expanded]
        filtered_casted = [c.to_scored_dimension_candidate() for c in filtered]

        return filtered_casted

    @staticmethod
    def _candidates_to_queries(inputs: dict) -> DatasetAvailabilityQueriesType:
        with debug_timer('LHCL._candidates_to_queries'):
            chain_state = LHCLChainState(**inputs)
            queries = query_utils.dimension_candidates_to_queries(
                candidates=chain_state.vector_candidates,
                date_time_query=None,
                dataset_2_dim_2_all_values_term=None,
                dataset_ids_to_be_present=None,
            )
            return queries

    def create_chain(self) -> Runnable:
        return (
            RunnableLambda(self._parse_chain_state)
            | (
                RunnablePassthrough.assign(
                    vector_candidates=self._get_candidates_from_vector_store,
                )
                | RunnablePassthrough.assign(weak_queries=self._candidates_to_queries)
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            f"{processor_stage_name_prefix(self._processor)}, Candidates, by dataset",
                            self._populate_weak_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | (
                RunnablePassthrough.assign(llm_candidates=self._prepare_llm_candidates)
                | RunnablePassthrough.assign(llm_response=self._llm_selection_chain)
                | RunnablePassthrough.assign(
                    vector_candidates=self._filter_candidates_by_llm_response
                )
                | RunnablePassthrough.assign(
                    # NOTE: this will create new instance of strong_queries
                    # ignoring already present strong_queries (non-indicators).
                    # it's ok, since this chain's outputs (strong_queries) will be combined
                    # with strong queries from indicators selection chain
                    # (containing both non-indicators and indicators).
                    strong_queries=self._candidates_to_queries
                )
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            f'{processor_stage_name_prefix(self._processor)}, Strong Queries',
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | self._format_final_response
        )
