from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from common.config import LLMModelsConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters


class RagFallbackFactory:

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt

    async def stream_response(self, inputs: dict) -> str:
        """
        Stream LLM response to `target` content (Choice for Router approach and Stage for Agentic approach)
        and return concatenated response as a string.
        """
        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._system_prompt),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        model = get_chat_model(
            api_key=auth_context.api_key,
            model=LLMModelsConfig.GPT_4_TURBO_2024_04_09,
            temperature=0.0,
        )

        chain = prompt_template | model | StrOutputParser()

        response_chunks = []
        async for chunk in chain.astream(inputs):
            if chunk:
                target.append_content(chunk)
                response_chunks.append(chunk)

        response = "".join(response_chunks)
        return response
