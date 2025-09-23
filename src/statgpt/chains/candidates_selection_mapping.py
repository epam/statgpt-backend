from operator import itemgetter

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from common.config import multiline_logger as logger
from common.schemas import LLMModelConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas import CandidatesRelevancyMapping
from statgpt.services import ScoredCandidate

from .candidates_selection_batched import BatchedSelectionInnerChainFactory


class CandidatesSelectionMappingChainFactory(BatchedSelectionInnerChainFactory):
    def __init__(
        self,
        llm_model_config: LLMModelConfig,
        system_prompt: str,
        user_prompt: str,
        candidates_key: str,
    ):
        super().__init__()
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._candidates_key = candidates_key

    @staticmethod
    def get_output_type():
        return CandidatesRelevancyMapping

    # NOTE: better to use a newer LLMSelectionCandidateBase model
    def _get_candidates(self, inputs: dict) -> list[ScoredCandidate]:
        return inputs[self._candidates_key]

    @staticmethod
    def _complex_indicator_indicator_fix(
        mapping: CandidatesRelevancyMapping,
    ) -> CandidatesRelevancyMapping:
        """
        TODO: this is a fix specific to complex indicators.
        Must remove it once we use a better LLM input format.
        """
        mapping.id_2_relevancy = {
            key.replace('.', '; '): value for key, value in mapping.id_2_relevancy.items()
        }
        return mapping

    def _route_based_on_candidates_presence(
        self, inputs: dict
    ) -> Runnable | CandidatesRelevancyMapping:
        candidates = self._get_candidates(inputs)
        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. ' 'Will return empty mapping dict.'
            )
            return CandidatesRelevancyMapping(id_2_relevancy={})

        auth_context = ChainParameters.get_auth_context(inputs)

        parser = PydanticOutputParser(pydantic_object=CandidatesRelevancyMapping)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", self._user_prompt),
            ],
        ).partial(format_instructions=parser.get_format_instructions())

        chain = (
            RunnablePassthrough.assign(selection_candidates_formatted=self._format_candidates)
            | RunnablePassthrough.assign(
                parsed_response=prompt_template
                | get_chat_model(
                    api_key=auth_context.api_key,
                    model_config=self._llm_model_config,
                )
                | parser
                | self._complex_indicator_indicator_fix
            )
            | self._remove_hallucinations
            | itemgetter("parsed_response")
        )
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )

        return chain

    def _format_candidates(self, inputs: dict) -> str:
        candidates = self._get_candidates(inputs)
        # NOTE: probably need to handle the case when there are no candidates
        text = ScoredCandidate.candidates_to_llm_string(candidates)
        return text

    def _remove_hallucinations(self, inputs: dict):
        candidates = self._get_candidates(inputs)
        parsed_response: CandidatesRelevancyMapping = inputs["parsed_response"]

        candidates_ids = {x.query_id for x in candidates}
        parsed_ids = set(parsed_response.id_2_relevancy.keys())

        hallucinations = parsed_ids.difference(candidates_ids)
        if hallucinations:
            logger.warning(
                f"!HALLUCINATION in Selection chain! "
                f"{len(hallucinations)} unexpected ids found: {hallucinations}"
            )
            parsed_response.id_2_relevancy = {
                k: v for k, v in parsed_response.id_2_relevancy.items() if k in candidates_ids
            }
            inputs["parsed_response"] = parsed_response  # let's be explicit

        missing_candidates = candidates_ids.difference(parsed_ids)
        if missing_candidates:
            logger.warning(
                f"!HALLUCINATION in Selection chain! {len(missing_candidates)} candidate ids "
                f"are missing from the LLM response: {missing_candidates}"
            )
            for ind_id in missing_candidates:
                parsed_response.id_2_relevancy[ind_id] = 0  # set to be non-relevant
            inputs["parsed_response"] = parsed_response  # let's be explicit

        return inputs

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)
