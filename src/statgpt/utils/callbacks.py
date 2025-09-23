import typing as t
from datetime import datetime
from uuid import UUID

from aidial_sdk.chat_completion import Choice, Stage
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig

from common.config import multiline_logger as logger
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.settings.dial_app import dial_app_settings


class StageCallback(AsyncCallbackHandler):
    def __init__(
        self,
        stage_name: str,
        content_appender: t.Callable[[Stage, t.Dict[str, t.Any]], t.Awaitable[None]] | None,
        debug_only: bool = False,
    ):
        self._stage_name = stage_name
        self._content_appender = content_appender
        self._choice_present = False
        self._run_id: UUID | None = None
        self._stage: Stage | None = None
        self._start_time: datetime | None = None
        self._debug_only = debug_only
        self._show_debug_stages = False

    async def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: t.Optional[UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> None:
        if self._run_id is not None:
            return

        if not isinstance(inputs, dict):
            # TODO: find the reason why this happens
            logger.warning(f"Expected 'inputs' to be a dict, got: {type(inputs)}")
            return

        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        self._show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES) or False
        if self._debug_only and not self._show_debug_stages:
            logger.info(f"Skipping debug stage: {self._stage_name}")
            return

        if choice is None:
            logger.warning(f'"choice" is absent in inputs for stage "{self._stage_name}" callback')
            return
        self._choice_present = True

        stage_name = self._stage_name
        if self._debug_only:
            stage_name = '[DEBUG] ' + stage_name

        self._run_id = run_id
        self._stage = choice.create_stage(stage_name)
        self._stage.open()
        self._start_time = datetime.now()

    async def on_chain_end(
        self,
        outputs: t.Dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: t.Optional[UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        **kwargs: t.Any,
    ) -> None:
        if self._debug_only and not self._show_debug_stages:
            return
        if not self._choice_present:
            return

        if run_id != self._run_id:
            return
        if self._run_id is None:
            raise ValueError("Run ID is not set")
        if self._stage is None:
            raise ValueError("Stage is not set")
        if self._start_time is None:
            raise ValueError("Start time is not set")
        try:
            if self._content_appender is not None:
                await self._content_appender(self._stage, outputs)
        except Exception as e:
            logger.exception(f"An error occurred while populating the stage content: {repr(e)}")
            self._stage.append_content('An error occurred while populating the stage content.')
        finally:
            end_time = datetime.now()
            start_str = self._start_time.strftime('%H:%M:%S')
            end_str = end_time.strftime('%H:%M:%S')
            took_seconds: str = (
                f" ({(end_time - self._start_time).total_seconds():.2f} s. "
                f"start: {start_str}, end: {end_str})"
            )
            if dial_app_settings.dial_show_stage_seconds:
                self._stage.append_name(took_seconds)
            self._stage.close()

    @classmethod
    def create_config(
        cls,
        stage_name: str,
        content_appender: t.Callable[[Stage, t.Dict[str, t.Any]], t.Awaitable[None]],
    ) -> RunnableConfig:
        return RunnableConfig(callbacks=[cls(stage_name, content_appender)])


class ChoiceCallback(AsyncCallbackHandler):
    _content_appender: t.Callable[[Choice, t.Dict[str, t.Any]], t.Awaitable[t.NoReturn]]
    _run_id: t.Optional[UUID]

    def __init__(
        self,
        content_appender: t.Callable[[Choice, t.Dict[str, t.Any]], t.Awaitable[t.NoReturn]],
    ):
        self._content_appender = content_appender
        self._choice_present = False
        self._run_id = None

    async def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: t.Optional[UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        if self._run_id is not None:
            return

        if not isinstance(inputs, dict):
            # TODO: find the reason why this happens
            logger.warning(f"Expected 'inputs' to be a dict, got: {type(inputs)}")
            return
        choice: Choice | None = inputs.get("choice")
        if choice is None:
            logger.warning('"choice" is absent in inputs for choice callback')
            return
        self._choice_present = True

        self._run_id = run_id

    async def on_chain_end(
        self,
        outputs: t.Dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: t.Optional[UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        if not self._choice_present:
            return
        if run_id != self._run_id:
            return
        if self._run_id is None:
            raise ValueError("Run ID is not set")
        choice: Choice = outputs["choice"]
        await self._content_appender(choice, outputs)


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
