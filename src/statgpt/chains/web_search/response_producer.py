import abc
from typing import Any

from openai import APIError

from common.config import multiline_logger as logger
from common.schemas import StagesConfig
from common.schemas.dial import Attachment
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.utils import OpenAiToDialStreamer, openai
from statgpt.utils.dial_tools import optional_stage, timed_stage


class ResponseProducerABC(abc.ABC):
    def __init__(
        self, deployment_id: str, stages_config: StagesConfig, system_prompt: str | None = None
    ):
        self._deployment_id = deployment_id
        self._stages_config = stages_config
        self._system_prompt = system_prompt

    @abc.abstractmethod
    async def run(self, inputs: dict, query: str) -> str:
        pass

    def _construct_history(self, query: str) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append({'role': 'system', 'content': self._system_prompt})

        messages.append({'role': 'user', 'content': query})
        return messages


class RagResponseProducer(ResponseProducerABC):
    """Returns the response from the DIAL WEB RAG"""

    async def run(self, inputs: dict, query: str) -> str:
        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)

        client = openai.get_async_client(api_key=auth_context.api_key)
        stream = await client.chat.completions.create(
            model=self._deployment_id, stream=True, messages=self._construct_history(query)
        )
        dial_streamer = OpenAiToDialStreamer(
            target,
            choice,
            deployment=self._deployment_id,
            stream_content=False,
            show_debug_stages=state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False),
            stages_config=self._stages_config,
        )

        with dial_streamer:
            try:
                async for chunk in stream:
                    dial_streamer.send_chunk(chunk)
            except APIError as e:
                logger.exception(e)

            return dial_streamer.content_with_attachments_metadata


class UrlOnlyResponseProducer(ResponseProducerABC):
    """Returns only the URLs of the attachments from the DIAL WEB RAG"""

    async def run(self, inputs: dict, query: str) -> str:
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)

        stage_generator = timed_stage(choice=choice, name=f"[DEBUG] Web RAG search: {query}")
        with optional_stage(stage_generator, enabled=show_debug_stages) as debug_stage:
            client = openai.get_async_client(api_key=auth_context.api_key)
            stream = await client.chat.completions.create(
                model=self._deployment_id,
                stream=True,
                messages=self._construct_history(query),
            )
            dial_streamer = OpenAiToDialStreamer(
                debug_stage,
                choice,
                deployment=self._deployment_id,
                stream_content=True,
                show_debug_stages=state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False),
                stages_config=self._stages_config,
            )

            with dial_streamer:
                try:
                    async for chunk in stream:
                        dial_streamer.send_chunk(chunk)
                except APIError as e:
                    logger.exception(e)

                attachments = [
                    Attachment.model_validate({k: v for k, v in attachment.items() if k != 'index'})
                    for attachment in dial_streamer.attachments
                ]
        return self._create_response_from(attachments)

    @staticmethod
    def _create_response_from(attachments: list[Attachment]) -> str:
        if len(attachments) == 0:
            return "No relevant web pages found."

        unique_urls = {attachment.reference_url for attachment in attachments}
        if len(unique_urls) == 1:
            return f"The answer can be found on the following web page: {unique_urls.pop()}"

        attachments_str = "\n".join(f"- {url}" for url in unique_urls)
        return f"The answer can be found on the following web pages:\n{attachments_str}"
