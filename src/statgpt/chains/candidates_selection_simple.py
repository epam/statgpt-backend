from operator import itemgetter

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from common.config import multiline_logger as logger
from common.schemas import LLMModelConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.schemas import LLMSelectionCandidateBase, SelectedCandidates

from .candidates_selection_batched import BatchedSelectionInnerChainFactory


class CandidatesSelectionSimpleChainFactory(BatchedSelectionInnerChainFactory):
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
        return SelectedCandidates

    def _get_candidates(self, inputs: dict) -> list[LLMSelectionCandidateBase]:
        return inputs[self._candidates_key]

    def _route_based_on_candidates_presence(self, inputs: dict) -> Runnable | SelectedCandidates:
        candidates = self._get_candidates(inputs)
        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. '
                'Will return empty list of selected ids.'
            )
            return SelectedCandidates(ids=[])

        auth_context = ChainParameters.get_auth_context(inputs)

        parser = PydanticOutputParser(pydantic_object=SelectedCandidates)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", self._user_prompt),
            ],
        ).partial(format_instructions=parser.get_format_instructions())

        chain = (
            RunnablePassthrough.assign(selection_candidates_formatted=self._format_candidates)
            | self._display_formatted_candidates_in_stage
            | RunnablePassthrough.assign(
                parsed_response=prompt_template
                | get_chat_model(
                    api_key=auth_context.api_key,
                    model_config=self._llm_model_config,
                )
                | parser
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
        if not candidates:
            return ''
        # NOTE: we assume all candidates are of the same type
        text = candidates[0].candidates_to_llm_string(candidates)
        return text

    def _display_formatted_candidates_in_stage(self, inputs: dict):
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES)

        if not show_debug_stages:
            return inputs

        with choice.create_stage(
            name='[DEBUG] Non-Indicator Candidates for LLM selection'
        ) as stage:
            candidates_formatted = inputs['selection_candidates_formatted']
            content = f'```yaml\n{candidates_formatted}\n```'
            stage.append_content(content)

        return inputs

    def _remove_hallucinations(self, inputs: dict):
        candidates = self._get_candidates(inputs)
        parsed_response: SelectedCandidates = inputs["parsed_response"]

        candidates_ids = {x._id for x in candidates}
        parsed_ids = set(parsed_response.ids)

        hallucinations = parsed_ids.difference(candidates_ids)
        if hallucinations:
            logger.warning(
                f"!HALLUCINATION in Selection chain! "
                f"{len(hallucinations)} unexpected ids found: {hallucinations}"
            )
            parsed_response.ids = list(parsed_ids.intersection(candidates_ids))
            inputs["parsed_response"] = parsed_response  # let's be explicit
        return inputs

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)
