import re
import time
import typing as t
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, GenerationChunk
from langchain_core.outputs.llm_result import LLMResult

from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas.token_usage import TokenUsageItem
from statgpt.common.utils.token_usage_context import get_token_usage_manager

from .exceptions import InvalidLLMStreamResponse


class LCMessageLoggerAsync(AsyncCallbackHandler):
    """
    Default LangChain logging (when using set_debug(True)) produces looooots of redundant logs.
    Here we define our custom langchain logger.
    """

    RE_B64_IMAGE_IN_HISTORY = re.compile(r"(data:image/(?:\w+);base64,)(.*?)(\'|\"|\n)")

    def langchain_msg_2_role_content(self, msg: BaseMessage):
        res = {'role': msg.type, 'content': msg.content}
        if self._log_tool_calls:
            if tool_calls := getattr(msg, 'tool_calls', None):
                res['tool_calls'] = tool_calls
            if tool_call_id := getattr(msg, 'tool_call_id', None):
                res['tool_call_id'] = tool_call_id
        return res

    def __init__(self, log_raw_llm_response=True, log_token_usage=False, log_tool_calls=True):
        """
        log_token_usage: whether we should log the use of tokens or not
        """

        super().__init__()
        self._log_raw_llm_response = log_raw_llm_response
        self._log_token_usage = log_token_usage
        self._log_tool_calls = log_tool_calls

    async def on_chat_model_start(
        self,
        serialized: dict[str, t.Any],
        messages: list[list[BaseMessage]],
        **kwargs: t.Any,
    ) -> None:
        """Run when Chat Model starts running."""
        if len(messages) != 1:
            raise ValueError(f'expected "messages" to have len 1, got: {len(messages)}')

        if serialized['id'][-1] == 'AzureChatOpenAI':
            try:
                model = serialized['kwargs']['deployment_name']
            except Exception:
                model = '<failed to determine LLM>'
        else:
            model = '<failed to determine LLM>'

        msgs_list = list(map(self.langchain_msg_2_role_content, messages[0]))
        msgs_str = '\n'.join(map(str, msgs_list))
        # remove base64 encoded image from calls to gpt-4-vision.
        msgs_str = self.RE_B64_IMAGE_IN_HISTORY.sub(r'\1<base64_image>\3', msgs_str)

        logger.info(f'call to {model} with {len(msgs_list)} messages:\n{msgs_str}')

    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: t.Any,
    ) -> None:
        """Run when LLM ends running."""
        generations = response.generations
        if len(generations) != 1:
            raise ValueError(f'expected "generations" to have len 1, got: {len(generations)}')
        if len(generations[0]) != 1:
            raise ValueError(f'expected "generations[0]" to have len 1, got: {len(generations[0])}')

        if self._log_raw_llm_response is True:
            gen: ChatGeneration = generations[0][0]  # type: ignore[assignment]
            ai_msg = gen.message
            logger.info(f'raw LLM response: "{ai_msg.content}"')

        if self._log_token_usage:
            llm_output = response.llm_output
            if llm_output:
                token_usage = llm_output.get('token_usage')
                logger.info(f"LLM usage (from LLM response): {token_usage}")
            else:
                logger.warning(
                    "failed to extract extract LLM usage from LLM response: 'llm_output' is None"
                )


class TokenUsageByModelsCallback(AsyncCallbackHandler):
    """Callback to track token usage across different models."""

    def __init__(self) -> None:
        super().__init__()
        self._run_2_deployment: dict[UUID, str] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, t.Any],
        messages: list[list[t.Any]],
        *,
        run_id: UUID,
        **kwargs: t.Any,
    ) -> None:
        try:
            self._run_2_deployment[run_id] = serialized['kwargs']['deployment_name']
        except (KeyError, TypeError):
            pass

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: t.Any,
    ) -> None:
        deployment_id = self._run_2_deployment.pop(run_id, None)

        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                else:
                    usage_metadata = None

                if not deployment_id and generation.generation_info:
                    deployment_id = generation.generation_info.get('model_name')
            except AttributeError:
                usage_metadata = None
        else:
            usage_metadata = None

        if usage_metadata:
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]
        else:
            if response.llm_output is None:
                return None
            if "token_usage" not in response.llm_output:
                return None
            # compute tokens and cost for this request
            token_usage = response.llm_output["token_usage"]
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)

        if not deployment_id and response.llm_output:
            deployment_id = response.llm_output.get('model_name')

        if not deployment_id:
            deployment_id = 'unknown'

        logger.info(
            f"Token usage for model {deployment_id!r}:"
            f" prompt_tokens={prompt_tokens!r}, completion_tokens={completion_tokens!r}"
        )

        token_usage_manager = get_token_usage_manager()
        token_usage_manager.add_usage(
            TokenUsageItem(
                deployment=deployment_id,
                model=deployment_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )


class LLMCallDurationCallback(AsyncCallbackHandler):
    """Callback to track and log LLM call duration."""

    def __init__(self) -> None:
        super().__init__()
        self._start_times: dict[UUID, float] = {}
        self._run_2_deployment: dict[UUID, str] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, t.Any],
        messages: list[list[t.Any]],
        *,
        run_id: UUID,
        **kwargs: t.Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        try:
            self._run_2_deployment[run_id] = serialized['kwargs']['deployment_name']
        except (KeyError, TypeError):
            pass

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: t.Any,
    ) -> None:
        start_time = self._start_times.pop(run_id, None)
        deployment_id = self._run_2_deployment.pop(run_id, None)
        if start_time is None:
            return

        duration_s = time.monotonic() - start_time

        if not deployment_id and response.llm_output:
            deployment_id = response.llm_output.get('model_name')
        if not deployment_id:
            deployment_id = 'unknown'

        logger.info(f"LLM call duration for model {deployment_id!r}: {duration_s:.2f}s")

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: t.Any,
    ) -> None:
        self._start_times.pop(run_id, None)
        self._run_2_deployment.pop(run_id, None)


class BrokenResponseInterceptor(AsyncCallbackHandler):
    raise_error: bool = True

    def __init__(self, regex_pattern: str, max_chunk_number: int = 3):
        if max_chunk_number < 1:
            raise ValueError('max_chunk_number must be >= 1')

        self._regex_pattern = regex_pattern
        self._max_chunk_number = max_chunk_number

        self._regex = re.compile(self._regex_pattern)
        self._chunk_number = 0

    def __repr__(self):
        return f'{type(self).__name__}(regex_pattern={self._regex_pattern}, max_chunk_number={self._max_chunk_number})'

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: t.Any,
    ) -> None:
        if self._regex.fullmatch(token) is None:
            self._chunk_number = 0
            return

        self._chunk_number += 1
        if self._chunk_number >= self._max_chunk_number:
            raise InvalidLLMStreamResponse(
                f"LLM streamed invalid response token {self._chunk_number} times in a row"
            )
