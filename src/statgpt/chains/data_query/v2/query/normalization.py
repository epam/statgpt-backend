from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from common.config import DialConfig, LLMModelsConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters


class NormalizationChain:

    _system_prompt: str

    def __init__(
        self,
        system_prompt: str,
        llm_api_base: str | None = None,
        llm_model_name: str | None = None,
        llm_temperature: float = 0.0,
    ):
        self._system_prompt = system_prompt
        self._llm_api_base = llm_api_base or DialConfig.get_url()
        self._llm_model_name = llm_model_name or LLMModelsConfig.GPT_4_TURBO_2024_04_09
        self._llm_temperature = llm_temperature

    @classmethod
    def get_normalization_input(cls, inputs: dict) -> str:
        """
        use groups expander output if available.
        otherwise, use the tool input query.
        """
        if groups_expander_output := inputs.get('query_with_expanded_groups'):
            return groups_expander_output

        query = ChainParameters.get_query(inputs)
        return query

    def create_chain(self, inputs: dict) -> Runnable:
        auth_context = ChainParameters.get_auth_context(inputs)
        input_query = self.get_normalization_input(inputs)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("user", input_query),
            ],
        )

        chain = (
            prompt_template
            | get_chat_model(
                api_key=auth_context.api_key,
                model=self._llm_model_name,
                temperature=self._llm_temperature,
                azure_endpoint=self._llm_api_base,
            )
            | StrOutputParser()
        )

        return chain
