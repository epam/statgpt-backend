from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field, model_validator

from common import utils
from common.config import logger
from common.schemas import LLMModelConfig
from common.settings.dial import dial_settings
from common.utils.models import get_chat_model
from statgpt.chains import ChainFactory
from statgpt.chains.data_query.v2.query.utils import DatasetDimQueriesSimpleDictFormatter
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetDimensionTermNameType,
    DatasetDimQueries,
    DatasetDimQueriesType,
)
from statgpt.services import ScoredIndicatorCandidate
from statgpt.utils.formatters import (
    CitationFormatterConfig,
    DatasetFormatterConfig,
    SimpleDatasetFormatter,
)


class DatasetDimQueriesByRelevancy(BaseModel):
    exact: DatasetDimQueries
    child: DatasetDimQueries

    @model_validator(mode='before')
    @classmethod
    def transform_structure(cls, values: dict) -> dict:
        """
        Transform structure to DatasetDimQueries instances.
        Reason is we want to avoid asking LLM to generate nested queries -
        the key "queries" is redundant.
        But we need this keyword to create DatasetDimQueries instances.
        So we insert this key here, before the model validation.
        """
        # insert "queries" key
        res = {k: {'queries': v} for k, v in values.items()}
        # now we need to check if all fields are present in the input
        for key in ['exact', 'child']:
            if key not in res:
                # use default value
                res[key] = DatasetDimQueries(queries={}).model_dump()
        return res

    def combine_with_priority(self):
        """
        Combine exact and child queries with following priority rules:
        1. use exact matches if they are valid. ignore child matches.
        2. if there are no valid exact matches, use child matches.
        """
        if self.exact.is_valid():
            return self.exact
        return self.child


class IndicatorCandidatesLLMFormatter:
    def __init__(self, dataset_id_2_name: dict[str, str]):
        self.dataset_id_2_name = dataset_id_2_name

    @classmethod
    def get_candidate_details_by_dataset(
        cls, candidates: list[ScoredIndicatorCandidate]
    ) -> DatasetDimensionTermNameType:
        cand_by_dataset = {}
        for c in candidates:
            cand_by_dataset.setdefault(c.dataset_id, []).append(c)

        dataset_data = {}
        for dataset_id, dataset_candidates in cand_by_dataset.items():
            # NOTE: can remove this type hint once VSCode stops complaining
            dataset_candidates: list[ScoredIndicatorCandidate]
            indicators = [x.indicator for x in dataset_candidates]

            dataset_dim_values = {}
            for ind in indicators:
                # all dataset indicators have same structure (components)
                components_details = ind.get_components_details()
                for cur_detail in components_details:
                    dataset_dim_values.setdefault(cur_detail.dimension_id, {})[
                        cur_detail.query_id
                    ] = cur_detail.name

            # there is at least 1 item
            first_ind_comp_details = indicators[0].get_components_details()
            cur_dataset_dimensions = {
                details.dimension_id: {
                    'name': details.dimension_name,
                    'values': dataset_dim_values[details.dimension_id],
                }
                for details in first_ind_comp_details
            }

            dataset_data[dataset_id] = cur_dataset_dimensions
        return dataset_data

    def _data2text(self, candidate_details_by_dataset: DatasetDimensionTermNameType) -> str:
        lines = []
        for dataset_id, dimension_data in candidate_details_by_dataset.items():
            dataset_name = self.dataset_id_2_name[dataset_id]
            lines.append(
                f'Dataset id: "{dataset_id}", dataset name: "{dataset_name}". '
                f'Dimensions (keys are dimension IDs):'
            )
            cur_text = utils.write_yaml_to_stream(dimension_data)
            lines.append(cur_text)
        res = '\n'.join(lines)
        return res

    def run(self, candidates: list[ScoredIndicatorCandidate]):
        data = self.get_candidate_details_by_dataset(candidates)
        res = self._data2text(data)
        return res


class LLMResponseBase(BaseModel, ABC):
    @abstractmethod
    def get_queries(self) -> DatasetDimQueriesType:
        pass

    async def populate_stage(self, inputs: dict) -> None:
        logger.info(f'default {type(self).__name__}.populate_stage(): doing nothing')


class PackedIndicatorsSelectionV1ChainFactory(ChainFactory):
    SYSTEM_PROMPT = '''\
You are expert in Economics and SDMX helping user to build a data query for sdmx data source.
You are provided with the user query and a yaml containing list of available dimension values for each available dataset.
Your task is to build SDMX query ONLY FROM the provided dimension values required in the use query, query may be empty.

{format_instructions}

## FLOW
1. relevancy summary.
analyze provided available dimension values.
summarize and reason (in natural language), which of them user would want to receive in the SDMX query.
in the summary:
    - mention dimension values in the following format: "value name [value id]"
    - group dimension values by dataset and dimension
    - DO NOT focus only on the MOST relevant values. reason about ALL relevant values.
2. based on your relevance summary, build queries

NOTES:
- select dimension values ONLY IF THEY WERE REQUIRED by user!
- DO NOT INCLUDE the dimension if it does not have values explicitly required by user
- you must maintain high precision and recall!
- it is FORBIDDEN to include dimension values absent in the provided list of candidates

YOUR RESPONSE MUST NOT INCLUDE ANYTHING ELSE EXCEPT THE REQUIRED JSON INSTANCE.

Example: "currency" dimension filter must be included for the query '"GDP in USD",
but it MUST NOT BE INCLUDED for "GDP for USA" query since it LACKS EXPLICIT currency specification.
'''

    USER_PROMPT = '''\
user query: {normalized_query}

candidates:

{yaml_candidates}
'''

    PARSED_RESPONSE_KEY = 'pis__parsed_response'

    class LLMResponse(LLMResponseBase):
        relevancy_summary: str = ''
        dataset_queries: DatasetDimQueriesType = Field(
            default={},
            description=(
                'mapping from dataset id (not name!!!) to query. '
                'query is a mapping from dimension id '
                'to list of dimension value ids required in the user query. '
            ),
        )

        def get_queries(self) -> DatasetDimQueriesType:
            return self.dataset_queries

        async def populate_stage(self, inputs: dict) -> None:
            state = ChainParameters.get_state(inputs)
            if not state.get(StateVarsConfig.SHOW_DEBUG_STAGES):
                return

            logger.info(f'{type(self).__name__}.populate_stage()')

            choice = ChainParameters.get_choice(inputs)
            chain_state = ChainState(**inputs)
            datasets = chain_state.datasets_dict

            lines = []

            lines.append('### Reasoning')
            lines.append(self.relevancy_summary)

            lines.append('### Queries')
            formatter = DatasetDimQueriesSimpleDictFormatter(
                datasets=datasets, auth_context=chain_state.auth_context
            )
            line = await formatter.format_multidataset_queries(queries=self.get_queries())
            lines.append(line)

            concat = '\n'.join(lines)

            with choice.create_stage('[DEBUG] Indicators Relevancy Scores') as stage:
                stage.append_content(concat)

    def __init__(
        self,
        candidates_key: str,
        llm_model_config: LLMModelConfig,
        llm_api_base: str | None = None,
    ):
        self._llm_response_class = self.LLMResponse

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT),
            ],
        )
        parser = PydanticOutputParser(pydantic_object=self._llm_response_class)
        self._prompt_template = self._prompt_template.partial(
            format_instructions=parser.get_format_instructions()
        )

        self._candidates_key = candidates_key
        self._llm_api_base = llm_api_base or dial_settings.url
        self._llm_model_config = llm_model_config

    def _get_candidates(self, inputs: dict) -> list[ScoredIndicatorCandidate]:
        return inputs[self._candidates_key]

    async def _populate_candidates_stage(self, inputs: dict):
        """
        NOTE: here we create a stage manually, without using the callback
        since callback requires 'choice' to be present in chain inputs,
        which is not always the case in subclasses.
        """
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES) or False

        # it's a debug stage
        if not show_debug_stages:
            return

        with choice.create_stage('[DEBUG] Indicator Candidates for LLM selection') as stage:
            candidates_formatted = inputs['yaml_candidates']
            content = f'```yaml\n{candidates_formatted}\n```'
            stage.append_content(content)

    def _create_chain_inner(self, llm):
        async def async_lambda(inputs):
            return await inputs[self.PARSED_RESPONSE_KEY].populate_stage(inputs)

        chain = (
            self._format_candidates
            | RunnablePassthrough.assign(_=self._populate_candidates_stage)
            | RunnablePassthrough.assign(**{self.PARSED_RESPONSE_KEY: self._prompt_template | llm})
            | RunnablePassthrough.assign(_=async_lambda)
            | self._remove_hallucinations
        )
        return chain

    def _route_based_on_candidates_presence(self, inputs: dict) -> Runnable | DatasetDimQueries:
        candidates = self._get_candidates(inputs)

        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. Will return empty mapping.'
            )
            return DatasetDimQueries()

        auth_context = ChainParameters.get_auth_context(inputs)

        llm = get_chat_model(
            api_key=auth_context.api_key,
            azure_endpoint=self._llm_api_base,
            model_config=self._llm_model_config,
        ).with_structured_output(self._llm_response_class, method='json_mode')
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )

        chain = self._create_chain_inner(llm=llm)

        return chain

    def _format_candidates(self, inputs: dict):
        candidates = self._get_candidates(inputs)
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        dataset_id_2_name = {ds.entity_id: ds.name for ds in datasets_dict.values()}
        formatter = IndicatorCandidatesLLMFormatter(dataset_id_2_name=dataset_id_2_name)
        text = formatter.run(candidates)
        inputs['yaml_candidates'] = text
        return inputs

    def _remove_hallucinations(self, inputs: dict) -> DatasetDimQueries:
        candidates = self._get_candidates(inputs)
        candidate_details_by_dataset = (
            IndicatorCandidatesLLMFormatter.get_candidate_details_by_dataset(candidates)
        )
        parsed_response: LLMResponseBase = inputs[self.PARSED_RESPONSE_KEY]
        dataset_queries_pp = {}  # post-processed dataset queries

        for dataset_id, query in parsed_response.get_queries().items():
            if dataset_id not in candidate_details_by_dataset:
                logger.warning(
                    f'!HALLUCINATION! '
                    f'LLM created query for non-existing dataset, "{dataset_id=}"'
                )
                continue

            dataset_queries_pp[dataset_id] = {}

            available_dim_values = {
                dim_id: set(dim_data['values'].keys())
                for dim_id, dim_data in candidate_details_by_dataset[dataset_id].items()
            }

            for dim_id, dim_values in query.items():
                if dim_id not in available_dim_values:
                    logger.warning(
                        f'!HALLUCINATION! '
                        'LLM created query for non-existing dimension '
                        f'"{dim_id=}" in "{dataset_id=}"'
                    )
                    continue

                hallucinations = set(dim_values).difference(available_dim_values[dim_id])

                if hallucinations:
                    logger.warning(
                        f'!HALLUCINATION! '
                        f'{len(hallucinations)} unexpected dimension values found '
                        f'for "{dim_id=}" in "{dataset_id=}": {hallucinations}'
                    )
                    dim_values_pp = list(set(dim_values).difference(hallucinations))
                else:
                    dim_values_pp = dim_values
                # update final response
                dataset_queries_pp[dataset_id][dim_id] = dim_values_pp

        res = DatasetDimQueries(queries=dataset_queries_pp)
        return res

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)


class PackedIndicatorsSelectionV2ChainFactory(PackedIndicatorsSelectionV1ChainFactory):
    """
    Introduce formalized relevancy types to assist LLM to build relevant queries.

    Override prompts and response model.
    """

    # NOTE: is not designed for multiple inicators in the query
    SYSTEM_PROMPT = '''\
You are expert in Economics and SDMX helping user to build a data query for SDMX data source.
You are provided with the user query
and a yaml containing list of available values for indicator dimensions of each available dataset.
YOUR TASK is to build queries for indicator dimensions matching user request in the best possible way.
- It is allowed to use ONLY PROVIDED dimension values to build queries
- Mind that indicators may consist of multiple dimensions
- All dimensions provided are indicator dimensions
- Not all provided dimensions must be filled, fill only those with values explicitly mentioned in the user query.
- Ignore COUNTRY and FREQUENCY dimensions that could be mentioned in the user query


## FLOW
1. reflect on user query and suggest statistical indicators (without providing actual names)
a professional would expect to receive in response.
    - be succinct, do not use lengthy phrases
    - avoid filler phrases in your response
    - focus on indicators, do not include COUNTRY and FREQUENCY dimensions in your response
    - if user query does not contain indicators, suggest nothing
2. build queries, grouped by relevancy type


## QUERY RELEVANCY TYPES

Built queries vary by relevancy to user request.

### exact
query either matches user request exactly,
or is a neccessary formalization with additional specifications.
examples (user request -> query):
- "gdp per capita" -> {{"indicator": ["gdp, per capita"]}}
- "unemployment rate" -> {{"indicator": ["unemployment, total (ILO estimate)", "unemployment, total (national estimate)"]}}
here, "total", "ILO estimate" and "national estimate" are additional specifications neccessary for formalization.
- "trade indicators" -> {{"indicator": ["total exports", "total imports", "trade balance"]}}
"trade indicators" is a broader term, so it is formalized.

### child
query with additional specifications not required by user request or by formalization.
has narrower meaning than what is requested.
examples:
- "gdp per capita" -> {{"indicator": "gdp, per capita, current prices"}}
- "unemployment rate" -> {{"indicator": "Unemployment, male (ILO estimate)"}} ("male" is additional specification)

### other
query with some specificaitons missing but still relevant to user query.
could be either a parent (having a subset of specifications)
or a sibling (meaning it has some common and some different specifications).
examples:
- "gdp per capita" -> {{"indicator": ["gdp"]}} (parent)
- "gdp per capita" -> {{"indicator": ["gdp, current prices"]}} (sibling)

### irrelevant
irrelevant to user query
example: "gdp per capita" -> {{"indicator": ["final consumption expenditure"]}}


## NOTES ON RELEVANCY
Note that relevancy is determined for the whole query, not for each dimension separately.
examples of multidim indicator queries:
"give me GDP per capita for USA in PPP international dollars" ->:
- exact: {{"indicator": ["GDP, per capita"], "unit_of_measure": ["PPP international dollars"]}}
- child: {{"indicator": ["GDP, per capita, constant prices"], "unit_of_measure": ["PPP international dollars"]}}


## RESPONSE FORMAT
you MUST answer with a following JSON:
{{
    "suggested_indicators": [<list of indicator suggestions>],
    "queries": {{
        "exact": {{
            <!-- exact queries. may be empty -->
            <dataset_id>: {{
                <dimension_id>: [<dimension values>]
            }}
        }},
        "child": {{
            <!-- child queries. include ONLY if there were no exact queries found. may be empty -->
            <dataset_id>: {{
                <dimension_id>: [<dimension values>]
            }}
        }}
        <!-- do not include queries with other relevancy types -->
    }}
}}
you MUST use dimension value IDs (and not their names) in this JSON.


## FINAL NOTES
- it is FORBIDDEN to include dimension values absent in the provided list of candidates
- it is allowed not to fill all dimensions
- do not include dimension values that were not explicitly mentioned in the user query.
it is especially true for dimension values like "all", "total", "all maturities", "all currencies". DO NOT INCLUDE THEM UNLESS THEY WERE EXPLICITLY MENTIONED IN THE USER QUERY.
- ignore COUNTRY and FREQUENCY dimensions, focus on indicator dimensions provided
- you must analyze ALL dimension values provided - to ensure 100% recall
- include only "exact" and "child" queries in the response. include "child" only if there are no "exact" queries.
- answer with JSON and nothing more
'''

    # 1. imagine, you don't have access to list of available dimension values.
    # reflect on user query and suggest statistical indicators (in natural language)
    # a professional would expect to receive in response.
    #     - be succinct, do not use lengthy phrases
    #     - avoid filler phrases in your response
    #     - focus on indicators, do not include COUNTRY and FREQUENCY dimensions in your response
    #     - if user query does not contain indicators, suggest nothing
    class LLMResponse(LLMResponseBase):
        suggested_indicators: list[str] = Field(
            default_factory=list
        )  # used for Chain of Thought prompting
        queries: DatasetDimQueriesByRelevancy

        def get_queries(self) -> DatasetDimQueriesType:
            # process relevancy types
            res = self.queries.combine_with_priority().queries
            return res

        async def populate_stage(self, inputs: dict) -> None:
            state = ChainParameters.get_state(inputs)
            data_service = ChainParameters.get_data_service(inputs)
            if not state.get(StateVarsConfig.SHOW_DEBUG_STAGES):
                return

            # TODO: error handling and the rest of stuff?

            logger.info(f'{type(self).__name__}.populate_stage()')

            choice = ChainParameters.get_choice(inputs)
            chain_state = ChainState(**inputs)
            datasets_dict = chain_state.datasets_dict

            lines = []
            exact = self.queries.exact
            child = self.queries.child
            exact_is_valid = exact.is_valid()
            child_is_valid = child.is_valid()

            # disclaimer
            lines.append(
                '**WARNING**: queries below are raw LLM response, '
                'they may have been filtered later in the pipeline during '
                'hallucinations removal or filtration by availability'
            )

            formatter = DatasetDimQueriesSimpleDictFormatter(
                datasets=datasets_dict, auth_context=chain_state.auth_context
            )

            lines.append('### Exact matches:')
            if exact_is_valid:
                line = await formatter.format_multidataset_queries(queries=exact.queries)
                lines.append(line)
            else:
                lines.append('No exact matches found')

            lines.append('### Child matches:')
            if child_is_valid:
                line = await formatter.format_multidataset_queries(queries=child.queries)
                lines.append(line)
            else:
                if exact_is_valid:
                    lines.append('Exact matches were found - no need to search for child matches')
                else:
                    lines.append('No child matches found')

            if exact_is_valid or child_is_valid:
                # add dataset citations
                lines.append('### Dataset References')
                dataset_ids = sorted(set(exact.queries.keys()).union(child.queries.keys()))
                for dataset_id in dataset_ids:
                    dataset = datasets_dict[dataset_id]
                    lines.append(f'* {dataset.name}')
                    lines.append(f'\t* ID: {dataset.source_id}')

                    # todo: check regression after Formatter refactoring

                    formatter = SimpleDatasetFormatter(
                        DatasetFormatterConfig.create_citation_only(
                            locale=data_service.channel_config.locale,
                            citation=CitationFormatterConfig(n_tabs=1, as_md_list=True),
                        ),
                        auth_context=chain_state.auth_context,
                    )
                    if dataset.config.citation:
                        new = await formatter.format(dataset)
                        lines.append(new)

            concat = '\n'.join(lines)

            with choice.create_stage('[DEBUG] Indicators Relevancy Scores') as stage:
                stage.append_content(concat)

    def __init__(
        self,
        candidates_key: str,
        llm_model_config: LLMModelConfig,
        llm_api_base: str | None = None,
    ):
        super().__init__(
            candidates_key=candidates_key,
            llm_model_config=llm_model_config,
            llm_api_base=llm_api_base,
        )

        # override, since we don't use parser and format instructions
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", self.USER_PROMPT),
            ],
        )
