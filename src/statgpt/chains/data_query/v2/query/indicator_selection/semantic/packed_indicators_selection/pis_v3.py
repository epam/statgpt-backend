from __future__ import annotations

import typing as t

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, ConfigDict, Field

from common.config import LLMModelsConfig, logger
from common.data.sdmx.common.indicator import ComplexIndicatorComponentDetails
from statgpt.chains.candidates_selection_batched import (
    BatchedSelectionInnerChainFactory,
    BatchedSelectionOutputBase,
    CandidatesSelectionBatchedChainFactory,
)
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.schemas.query_builder import ChainState, DatasetDimQueries
from statgpt.services import ScoredIndicatorCandidate

from . import PackedIndicatorsSelectionV1ChainFactory
from .pis_v1_v2 import LLMResponseBase


class IndicatorCandidatesLLMFormatter:
    _raw_data: list[RawDataItem] | None = None

    class RawDataItem(BaseModel):
        index: int
        candidate: ScoredIndicatorCandidate  # NOTE: probably not needed
        details: dict[str, ComplexIndicatorComponentDetails]

    def __init__(self, dataset_id_2_name: dict[str, str]):
        self.dataset_id_2_name = dataset_id_2_name

    @property
    def raw_data(self):
        if self._raw_data is None:
            raise ValueError('raw_data is not yet initialized. use the "run" method first')
        return self._raw_data

    def init_raw_data(
        self, candidates: list[ScoredIndicatorCandidate], start_ix: int, batch_size: int
    ):
        # ensure there is no overlapping between the candidates indices
        if len(candidates) > batch_size:
            raise ValueError(
                f'batch size ({batch_size}) is smaller than the number of candidates '
                f'passed to batch inner chain ({len(candidates)}).'
            )

        res = []
        for ix, cand in enumerate(candidates, start=start_ix):
            details = cand.indicator.get_components_details()
            details_dict = {chr(ord('A') + i): detail for i, detail in enumerate(details)}
            raw_data_item = self.RawDataItem(index=ix, candidate=cand, details=details_dict)
            res.append(raw_data_item)
        self._raw_data = res

    @staticmethod
    def _data2text(data: list[RawDataItem]):
        lines = []
        for item in data:
            details_gen = (
                f'[{key}] {value.dimension_name}. {value.name}'
                for key, value in sorted(item.details.items(), key=lambda x: x[0])
            )
            details_str = ' '.join(details_gen)
            lines.append(f'{item.index}. {details_str}')
        res = '\n'.join(lines)
        return res

    def run(self, candidates: list[ScoredIndicatorCandidate], start_ix: int, batch_size: int):
        self.init_raw_data(candidates=candidates, start_ix=start_ix, batch_size=batch_size)
        text = self._data2text(data=self.raw_data)
        return text

    @staticmethod
    def _data2text_w_selection(data: list[RawDataItem], selected_parts: dict[int, list[str]]):
        lines = []
        for item in data:
            details_gen = (
                f'[{key}] {value.dimension_name}. {value.name}'
                for key, value in sorted(item.details.items(), key=lambda x: x[0])
            )
            details_str = ' '.join(details_gen)

            selected_parts_str = ', '.join(selected_parts.get(item.index, []))

            lines.append(f'{item.index}. {details_str} -> [{selected_parts_str}]')
        res = '\n'.join(lines)
        return res

    def format_candidates_and_selection(self, candidate_ix_2_relevant_parts: dict[int, list[str]]):
        selected_candidates_ixs = set(
            key for key, value in candidate_ix_2_relevant_parts.items() if value
        )
        selected_data_items = [
            item for item in self.raw_data if item.index in selected_candidates_ixs
        ]
        nonselected_data_items = [
            item for item in self.raw_data if item.index not in selected_candidates_ixs
        ]
        selected_text = (
            self._data2text_w_selection(
                data=selected_data_items, selected_parts=candidate_ix_2_relevant_parts
            )
            if selected_data_items
            else None
        )
        nonselected_text = (
            self._data2text(data=nonselected_data_items) if nonselected_data_items else None
        )
        return {'selected': selected_text, 'nonselected': nonselected_text}


class PackedIndicatorsSelectionV3InnerChainFactory(
    BatchedSelectionInnerChainFactory,
    PackedIndicatorsSelectionV1ChainFactory,
):
    SYSTEM_PROMPT = '''\
You are expert in Economics and SDMX helping user to build a data query for SDMX data source, \
focusing only on indicator dimensions.

You are provided with the user query \
and a list of indicator candidates extracted using embeddings (vector) search.
Each candidate is a complex indicator,
presented as concatenation of multiple parts,
where each part consists of dimension name and a single value for this dimension.
Each part has the following format: "[<part_id>] <dimension_name>. <dimension_value_name>"

Please note that user may request multiple indicators.
Also, user may request indicators indirectly, by their synonyms or related terms.

Your task is to:
1. identify which indicators are requested by user.
2. map EACH candidate to list of parts RELEVANT to at least 1 indicator requested by user.
Ignore dimension values that are not explicitly mentioned in the user query.
However, mind that user may request indicators indirectly.

You MUST answer with following JSON instance:
{{
    "requested_indicators": [list of indicators (as strings) requested by user],
    "candidate_ix_2_relevant_parts": {{
        <candidate_ix>: [
            list of part ids (as letters, without brackets) relevant to the user query.
            1st item must be a very short (under 7 words) summary of candidate's parts.
            include summary for each candidate.
        ]
    }}
}}

NOTES:
- it is allowed to select ONLY SUBSET or even NO parts for a candidate.
remember to select only RELEVANT parts!
- you must map EVERY candidate index to list of relevant parts
- you must NOT skip any relevant parts - we need to ensure 100% recall!
- first element of each part ids list MUST be a short summary of the candidate.
while writing this summary, you MUST think about the candidate parts' relevancy
to indicators requested by user
- if user requested multiple indicators, try to find relevant parts for all of them
'''

    # TODO: change "yaml_candidates" key to "formatted_indicator_candidates"
    # since we don't use yaml in this chain
    USER_PROMPT = '''\
user query: {normalized_query}

candidates:

{yaml_candidates}
'''

    LLM_FORMATTER_KEY = 'llm_formatter'

    class LLMResponse(LLMResponseBase):
        requested_indicators: list[str]
        candidate_ix_2_relevant_parts: dict[int, list[str]] = Field(
            default={},
            description="Mapping from indicator candidate index to relevant part ids",
        )

        def get_queries(self):
            pass  # not used

    class CombinedOutput(BaseModel):
        queries: DatasetDimQueries
        selection_status_str: str

        model_config = ConfigDict(arbitrary_types_allowed=True)

    class BatchedSelectionOutput(BatchedSelectionOutputBase):
        batch_ix: int
        queries: DatasetDimQueries
        candidates_selection_debug_data: dict[str, str | None]

        @classmethod
        def combine_batch_outputs(
            cls, batch_outputs: list[t.Self]
        ) -> PackedIndicatorsSelectionV3InnerChainFactory.CombinedOutput:
            """
            NOTE: we hack a bit and expect batch_outputs to be of different type than self.
            reason is it's easier to combine DatasetDimQueries other than
            combining LLMResponse and then converting it to DatasetDimQueries
            without access to the formatter storing mapping from dimension codes to dimension values.
            """
            if not batch_outputs:
                # should never happen, since we check for candidates presence, but just in case
                return PackedIndicatorsSelectionV3InnerChainFactory.CombinedOutput(
                    queries=DatasetDimQueries(queries={}),
                    selection_status_str='',
                )

            # combine queries

            batch_outputs[0].queries
            combined_dict = batch_outputs[0].queries.queries
            for bout in batch_outputs[1:]:
                dataset_dim_queries = bout.queries
                for dataset_id, dim_queries in dataset_dim_queries.queries.items():
                    ds_queries = combined_dict.setdefault(dataset_id, {})
                    for dim_id, dim_values in dim_queries.items():
                        ds_queries.setdefault(dim_id, []).extend(dim_values)

            # keep only unique dim values
            for dataset_id, dim_queries in combined_dict.items():
                for dim_id in dim_queries.keys():
                    dim_queries[dim_id] = sorted(set(dim_queries[dim_id]))

            logger.info(f'combined indicator queries: {combined_dict}')

            # combine debug data

            sorted_debug_data = [
                bout.candidates_selection_debug_data
                for bout in sorted(batch_outputs, key=lambda x: x.batch_ix)
            ]
            selected_text_parts = [
                data['selected'] for data in sorted_debug_data if data['selected']
            ]
            nonselected_text_parts = [
                data['nonselected'] for data in sorted_debug_data if data['nonselected']
            ]
            selected_text = (
                '\n'.join(selected_text_parts) if selected_text_parts else '- Nothing selected'
            )
            nonselected_text = (
                '\n'.join(nonselected_text_parts) if nonselected_text_parts else '- Nothing'
            )
            debug_text = (
                f'### Selected\n\n```\n{selected_text}\n```'
                '\n\n'
                f'### Not Selected\n\n```\n{nonselected_text}\n```'
            )

            return PackedIndicatorsSelectionV3InnerChainFactory.CombinedOutput(
                queries=DatasetDimQueries(queries=combined_dict),
                selection_status_str=debug_text,
            )

        def get_selected_ids(self):
            # not needed, since we return queries and not single ids
            raise NotImplementedError

    def __init__(
        self,
        candidates_key: str,
        llm_api_base: str | None = None,
        llm_model_name: str | None = None,
        llm_temperature: float = 0.0,
    ):
        # override
        # llm_model_name = LLMModelsConfig.GPT_4_O_2024_08_06
        llm_model_name = LLMModelsConfig.GPT_4_TURBO_2024_04_09
        super().__init__(
            candidates_key=candidates_key,
            llm_api_base=llm_api_base,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
        )

        # override, since we don't use parser and format instructions
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT),
            ],
        )

    @classmethod
    def get_output_type(cls):
        return cls.BatchedSelectionOutput

    async def _populate_candidates_stage(self, inputs: dict):
        pass  # not using

    def _format_candidates(self, inputs: dict):
        batch_ix = self.get_batch_ix(inputs)
        batch_size = self.get_batch_size(inputs)
        candidates = self._get_candidates(inputs)
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        dataset_id_2_name = {ds.entity_id: ds.name for ds in datasets_dict.values()}
        formatter = IndicatorCandidatesLLMFormatter(dataset_id_2_name=dataset_id_2_name)
        text = formatter.run(candidates, start_ix=1 + batch_ix * batch_size, batch_size=batch_size)
        inputs['yaml_candidates'] = text
        inputs[self.LLM_FORMATTER_KEY] = formatter
        return inputs

    def _remove_hallucinations(self, inputs: dict) -> dict:
        parsed_response: self.LLMResponse = inputs[self.PARSED_RESPONSE_KEY]
        formatter: IndicatorCandidatesLLMFormatter = inputs[self.LLM_FORMATTER_KEY]
        ix2data = {item.index: item for item in formatter.raw_data}

        # first hallucination check

        res_filtered = parsed_response.candidate_ix_2_relevant_parts
        missed_ixs = set(ix2data.keys()).difference(res_filtered.keys())
        if missed_ixs:
            logger.warning(
                f"!HALLUCINATION in batched indicators selection chain (inner)! "
                f"{len(missed_ixs)} indices missing: {missed_ixs}"
            )
            for ix in missed_ixs:
                res_filtered[ix] = []

        # second hallucination check

        hallucinated_indices = set(res_filtered.keys()).difference(ix2data.keys())
        if hallucinated_indices:
            logger.warning(
                f"!HALLUCINATION in batched indicators selection chain (inner)! "
                f"{len(hallucinated_indices)} unexpected indices found: {hallucinated_indices}"
            )
            res_filtered = {ix: res_filtered[ix] for ix in ix2data.keys()}

        # third hallucination check

        # use a copy of keys to avoid accidental errors with modifying the dict during iteration
        for ix in list(res_filtered.keys()):
            # NOTE: we convert to uppercase to avoid case sensitivity issues
            selected_detail_codes = set(x.upper() for x in res_filtered[ix])
            # remove placeholder, used for "let's think dot by dot
            # selected_detail_codes.discard('...')

            available_detail_codes = ix2data[ix].details.keys()
            hallucinated_codes = selected_detail_codes.difference(available_detail_codes)
            if hallucinated_codes:
                # there will be indicator summaries for each candidate.
                # thus we don't need to produce lots of logs.
                # logger.warning(
                #     f"!HALLUCINATION in batched indicators selection chain (inner)! "
                #     f"{len(hallucinated_codes)} unexpected detail codes found: {hallucinated_codes}"
                # )
                res_filtered[ix] = list(
                    set(selected_detail_codes).intersection(available_detail_codes)
                )
            else:
                res_filtered[ix] = selected_detail_codes

        filtered_llm_response = self.LLMResponse(
            requested_indicators=parsed_response.requested_indicators,
            candidate_ix_2_relevant_parts=res_filtered,
        )
        inputs[self.PARSED_RESPONSE_KEY] = filtered_llm_response
        return inputs

    def _convert_to_queries(self, inputs: dict) -> DatasetDimQueries:
        """
        Convert LLMResponse to DatasetDimQueries here, in inner batched chain,
        so that we don't need to map dimension codes to actual dimension values
        after combining batch outputs (we would need Formatter for that)
        """
        parsed_response: self.LLMResponse = inputs[self.PARSED_RESPONSE_KEY]
        formatter: IndicatorCandidatesLLMFormatter = inputs[self.LLM_FORMATTER_KEY]
        ix2data = {item.index: item for item in formatter.raw_data}

        dataset_dim_queries_dict = {}
        for ix, detail_codes in parsed_response.candidate_ix_2_relevant_parts.items():
            raw_data_item = ix2data[ix]
            dataset_id = raw_data_item.candidate.dataset_id
            for detail_code in detail_codes:
                detail = raw_data_item.details[detail_code]
                # use set to store dim values to avoid duplicates.
                # but we'll need to convert it to list later
                dataset_dim_queries_dict.setdefault(dataset_id, {}).setdefault(
                    detail.dimension_id, set()
                ).add(detail.query_id)
        # convert dim values from set to list
        for dataset_id, dim_queries in dataset_dim_queries_dict.items():
            for dim_id in dim_queries:
                dim_queries[dim_id] = list(dim_queries[dim_id])

        res = DatasetDimQueries(queries=dataset_dim_queries_dict)
        return res

    def _pack_results(self, inputs: dict):
        batch_ix = self.get_batch_ix(inputs)
        queries = self._convert_to_queries(inputs)
        parsed_response: self.LLMResponse = inputs[self.PARSED_RESPONSE_KEY]
        formatter: IndicatorCandidatesLLMFormatter = inputs[self.LLM_FORMATTER_KEY]
        candidates_selection_debug_data = formatter.format_candidates_and_selection(
            candidate_ix_2_relevant_parts=parsed_response.candidate_ix_2_relevant_parts
        )
        res = PackedIndicatorsSelectionV3InnerChainFactory.BatchedSelectionOutput(
            batch_ix=batch_ix,
            queries=queries,
            candidates_selection_debug_data=candidates_selection_debug_data,
        )
        return res

    def _create_chain_inner(self, llm):
        chain = (
            self._format_candidates
            | RunnablePassthrough.assign(**{self.PARSED_RESPONSE_KEY: self._prompt_template | llm})
            | self._remove_hallucinations
            | self._pack_results
        )
        return chain


class PackedIndicatorsSelectionV3ChainFactory(PackedIndicatorsSelectionV1ChainFactory):
    """
    Do not unpack indicators. For each complex (packed) indicator select relevant dimension values.

    NOTE: subclass from PackedIndicatorsSelectionV1ChainFactory
    and not from PackedIndicatorsSelectionV2ChainFactory to avoid the need to refactor
    in case PackedIndicatorsSelectionV2ChainFactory is deleted.
    """

    SELECTION_OUTPUT_KEY = 'pis3__selection_output'

    def __init__(
        self,
        candidates_key: str,
        llm_api_base: str | None = None,
        llm_model_name: str | None = None,
        llm_temperature: float = 0.0,
    ):
        super().__init__(
            candidates_key=candidates_key,
            llm_api_base=llm_api_base,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
        )

        # override, since we don't use parser and format instructions
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT),
            ],
        )

    def _show_combined_queries_stage(self, inputs: dict):
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES) or False

        # it's a debug stage. thus we check whether to show it
        if not show_debug_stages:
            return inputs

        choice = ChainParameters.get_choice(inputs)
        selection_output: PackedIndicatorsSelectionV3InnerChainFactory.CombinedOutput = inputs[
            self.SELECTION_OUTPUT_KEY
        ]

        with choice.create_stage('[DEBUG] Indicators selection: combined queries') as stage:
            stage.append_content(selection_output.selection_status_str)

        return inputs

    def _get_queries(self, inputs) -> DatasetDimQueries:
        return inputs[self.SELECTION_OUTPUT_KEY].queries

    def _route_based_on_candidates_presence(self, inputs: dict):
        candidates = self._get_candidates(inputs)

        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. Will return empty mapping.'
            )
            return DatasetDimQueries()

        # create inner chain
        inner_chain_factory = PackedIndicatorsSelectionV3InnerChainFactory(
            candidates_key=self._candidates_key,
            llm_api_base=self._llm_api_base,
            llm_model_name=self._llm_model_name,
            llm_temperature=self._llm_temperature,
        )

        # wrap it in batched chain
        batched_chain = CandidatesSelectionBatchedChainFactory(
            inner_chain_factory=inner_chain_factory,
            candidates_key=self._candidates_key,
            batch_size=20,  # TODO: move to config
        ).create_chain()

        chain = (
            RunnablePassthrough.assign(**{self.SELECTION_OUTPUT_KEY: batched_chain})
            | self._show_combined_queries_stage
            | self._get_queries
        )

        return chain

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)
