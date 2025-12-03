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

    def _get_llm_response(self, inputs: dict) -> SelectedCandidates:
        return inputs["parsed_response"]

    def _route_based_on_candidates_presence(self, inputs: dict) -> Runnable | SelectedCandidates:
        candidates = self._get_candidates(inputs)
        if not candidates:
            logger.warning(
                'No candidates were passed to selection chain. '
                'Will return empty list of selected ids.'
            )
            return SelectedCandidates(ids=[])

        auth_context = ChainParameters.get_auth_context(inputs)

        parser: PydanticOutputParser[SelectedCandidates] = PydanticOutputParser(
            pydantic_object=SelectedCandidates
        )
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
            | self._display_llm_response_in_stage
            | self._get_llm_response
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
        text = candidates[0].candidates_to_llm_string(candidates)  # type: ignore[arg-type]
        return text

    def _display_formatted_candidates_in_stage(self, inputs: dict) -> dict:
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

    def _display_llm_response_in_stage(self, inputs: dict) -> dict:
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES)

        if not show_debug_stages:
            return inputs

        response = self._get_llm_response(inputs)
        response_formatted = response.model_dump_json(indent=2)
        with choice.create_stage(name='[DEBUG] Non-Indicator LLM Selection Response') as stage:
            content = f'```json\n{response_formatted}\n```'
            stage.append_content(content)

        return inputs

    def _remove_hallucinations(self, inputs: dict):
        candidates = self._get_candidates(inputs)
        response = self._get_llm_response(inputs)

        candidates_ids = {x._id for x in candidates}
        selected_ids = set(response.ids)

        hallucinations = selected_ids.difference(candidates_ids)
        if hallucinations:
            logger.warning(
                f"!HALLUCINATION in Selection chain! "
                f"{len(hallucinations)} unexpected ids found: {hallucinations}"
            )
            response.ids = list(selected_ids.intersection(candidates_ids))  # inplace update
            inputs["parsed_response"] = response  # not required, but explicit
        return inputs

    def create_chain(self) -> Runnable:
        return RunnableLambda(self._route_based_on_candidates_presence)
