import re
import typing as t

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs.llm_result import LLMResult

from common.config import multiline_logger as logger
from common.schemas.token_usage import TokenUsageItem
from common.utils.token_usage_context import get_token_usage_manager


class LCMessageLoggerAsync(AsyncCallbackHandler):
    # NOTE: According to https://python.langchain.com/docs/modules/callbacks/async_callbacks
    # "If you are planning to use the async API,
    # it is recommended to use AsyncCallbackHandler to avoid blocking the runloop."
    #
    # For sync callback handler, subclass from 'BaseCallbackHandler'

    """
    Default LangChain logging (when using set_debug(True)) produces looooots of redundant logs.
    Here we define our custom langchain logger.
    """

    RE_B64_IMAGE_IN_HISTORY = re.compile(r"(data:image/(?:\w+);base64,)(.*?)(\'|\"|\n)")

    def langchain_msg_2_role_content(self, msg: BaseMessage):
        res = {'role': msg.type, 'content': msg.content}
        if self._log_tool_calls:
            if tool_calls := msg.additional_kwargs.get('tool_calls'):
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

    def on_chat_model_start(
        self, serialized: dict[str, t.Any], messages: list[list[BaseMessage]], **kwargs: t.Any
    ) -> t.Any:
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

    def on_llm_end(self, response: LLMResult, **kwargs: t.Any) -> t.Any:
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

    def on_llm_end(self, response: LLMResult, **kwargs: t.Any) -> None:  # type: ignore[override]
        deployment_id = None

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

                if generation.generation_info:
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


# class LCGPTUsageLoggerAsync(AsyncCallbackHandler):
#     """
#     NOTE: may be obsolete and not-required after langchain v2 release. Need to check!

#     Langchain callback handler counting tokens for any (streaming/non-streaming) chat LLM calls.
#     NOTE: standard langchain callbacks can't handle streaming, this class can.
#     Tracks usage of all calls within a chain.
#     NOTE: It is tied to openai, since it uses it's own tokenizer - tiktoken.
#     NOTE: you need to reinstantiate this handler each time,
#     else token stats will get accumulated from multiple calls.

#     Rationale: see langchain-bug-streaming-llm-usage-stats.ipynb notebook (in old Azure repository)
#     describing bug of counting LLM usage stats in LangChain streaming calls.

#     After the call you can access token stats like: 'handler.usage'.
#     """

#     RE_B64_IMAGE = re.compile("^data:image/jpeg;base64,(.*)$")

#     def __init__(self, verbose=True, chain_name: str = ''):
#         self.verbose = verbose

#         self._tokenizer = None
#         self.usage = MultiLLMUsage()
#         # NOTE: this callback may probably be not thread-safe, since
#         # self.prompt_tokens or self.current_model may get updated
#         # between self.on_chat_model_start() and self.on_llm_end() calls.
#         self.prompt_tokens = 0
#         self.current_model: str = None
#         self.chain_name = chain_name
#         self.name = type(self).__name__

#         logger.info(f'initialized {self.name} for "{self.chain_name or "unnamed"}" chain')

#     def _count_gpt_prompt_tokens_chat_llm(self, messages: list[list[BaseMessage]]):
#         """
#         Count tokens in prompt to GPT chat LLM.

#         Modified code from:
#         https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
#         """

#         # TODO: can delegate to `llm.get_num_tokens_from_messages()` function

#         tokens_per_message = 3
#         # tokens_per_name = 1  # see assumption below. we assume we won't use it.

#         self.prompt_tokens = 0

#         for messages_list in messages:
#             for msg in messages_list:
#                 # each chat message has it's own additional tokens
#                 self.prompt_tokens += tokens_per_message
#                 # now we add number of tokens from the message role.
#                 # it equals 1 (you can check it yourself).
#                 self.prompt_tokens += 1
#                 # now we add number of tokens from the message content
#                 msg_content = msg.content
#                 if isinstance(msg_content, str):
#                     tokenized = self._tokenizer.encode(msg_content)
#                     self.prompt_tokens += len(tokenized)
#                     # NOTE: and we assume no "name" field is present in any of messages used.
#                     # see original example in jupyter notebook from openai docs (above).
#                 elif isinstance(msg_content, list):
#                     # we support only 1 such case: sending request to vision LLM
#                     try:
#                         text = msg_content[0]['text']
#                         tokenized = self._tokenizer.encode(text)
#                         self.prompt_tokens += len(tokenized)

#                         img_url = msg_content[1]['image_url']['url']
#                         match = self.RE_B64_IMAGE.fullmatch(img_url)
#                         b64_image = match.group(1)
#                         img_bytes = base64.b64decode(b64_image)
#                         buf = BytesIO(img_bytes)
#                         img = Image.open(buf)
#                         width, height = img.size
#                         n_img_tokens = VisionModelImageTokensCounter.count(
#                             width=width, height=height, detail='high'
#                         )
#                         logger.info(
#                             'decoded b64 image from call to LLM. size: '
#                             f'{width}x{height} taking {n_img_tokens} tokens'
#                         )
#                         self.prompt_tokens += n_img_tokens
#                     except Exception as e:
#                         logger.error(
#                             'error while handling the case '
#                             f'when message content is a list: {repr(e)}'
#                         )
#                 else:
#                     logger.warning(
#                         f'unsupported mode! message content has {type(msg_content)} type'
#                     )

#         # every reply is primed with <|start|>assistant<|message|>
#         self.prompt_tokens += 3

#     async def on_chat_model_start(
#         self, serialized: dict[str, t.Any], messages: list[list[BaseMessage]], **kwargs
#     ):
#         """
#         On each new call to LLM we need to:
#         - instantiate new tokenizer
#         - set 'current_model'
#         so that we can track usages of calls to different models.
#         """
#         if serialized['id'][-1] == 'AzureChatOpenAI':
#             self.current_model = serialized['kwargs']['azure_deployment']
#             self._tokenizer = tiktoken.encoding_for_model(self.current_model)
#         else:
#             self.current_model = '<model_not_recognized>'
#             # use default tokenizer
#             logger.warning(f'{self.name}. failed to get LLM model name. will use default tokenizer')
#             self._tokenizer = tiktoken.get_encoding('cl100k_base')

#         self._count_gpt_prompt_tokens_chat_llm(messages=messages)

#     async def on_llm_end(self, response: LLMResult, **kwargs: t.Any) -> None:
#         completion_tokens = 0
#         for generation_list in response.generations:
#             for generation in generation_list:
#                 tokenized = self._tokenizer.encode(generation.text)
#                 completion_tokens += len(tokenized)

#         usage_total = LLMUsage.from_tokens_stats(
#             prompt_tokens=self.prompt_tokens,
#             completion_tokens=completion_tokens,
#             model_name=self.current_model,
#         )
#         self.usage += usage_total

#         if self.verbose is True:
#             logger.info(f'LC-GPT LLM usage, last call: {usage_total}')
#             chain_name_formatted = f'"{self.chain_name}" ' if self.chain_name else ''
#             logger.info(f"LC-GPT LLM usage, {chain_name_formatted}chain total: {self.usage}")
