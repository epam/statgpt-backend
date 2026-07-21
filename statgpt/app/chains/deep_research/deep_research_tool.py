import time
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from openai import APIError
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.utils import OpenAiToDialStreamer, openai
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import DeepResearchTool as DeepResearchToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager


class DeepResearchArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description="The natural language question to research in depth."
    )


class DeepResearchTool(StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.DEEP_RESEARCH):
    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @classmethod
    def get_args_schema(cls, tool_config: DeepResearchToolConfig) -> type[DeepResearchArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DeepResearchArgs

    def _construct_history(self, query: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt := self._tool_config.details.system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': query})
        return messages

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)

        details = self._tool_config.details
        deployment_id = details.get_deployment_id()

        client = openai.get_async_client(api_key=auth_context.api_key)

        create_kwargs: dict[str, Any] = dict(
            model=deployment_id,
            stream=True,
            messages=self._construct_history(query),
        )
        if details.configuration:
            create_kwargs["extra_body"] = dict(
                custom_fields=dict(configuration=details.configuration)
            )

        show_debug_stages = (
            state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False) or details.always_show_stages
        )

        time_start = time.monotonic()
        stream = await client.chat.completions.create(**create_kwargs)
        dial_streamer = OpenAiToDialStreamer(
            target,
            choice,
            deployment=deployment_id,
            stream_content=True,
            show_debug_stages=show_debug_stages,
            stages_config=details.stages_config,
        )

        with dial_streamer:
            try:
                async for chunk in stream:
                    dial_streamer.send_chunk(chunk)
            except APIError as e:
                logger.exception(e)

        duration_s = time.monotonic() - time_start
        if (duration_manager := get_llm_call_duration_manager()) is not None:
            duration_manager.add_duration(
                LLMCallDurationItem(deployment=deployment_id, duration_s=duration_s)
            )

        response = dial_streamer.content_with_attachments_metadata
        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))
        return response, artifact
