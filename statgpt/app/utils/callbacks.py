import typing as t
from datetime import datetime
from uuid import UUID

from aidial_sdk.chat_completion import Choice, Stage, Status
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import StateVarsConfig
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.config import multiline_logger as logger


class StageCallback(AsyncCallbackHandler):
    def __init__(
        self,
        stage_name: str,
        content_appender: t.Callable[[Stage, dict[str, t.Any]], t.Awaitable[None]] | None,
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
        serialized: dict[str, t.Any],
        inputs: dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, t.Any] | None = None,
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

    def _append_stage_timing(self) -> None:
        if not dial_app_settings.dial_show_stage_seconds:
            return
        assert self._start_time is not None, "Start time must be set"
        assert self._stage is not None, "Stage must be set"
        end_time = datetime.now()
        start_str = self._start_time.strftime('%H:%M:%S')
        end_str = end_time.strftime('%H:%M:%S')
        took_seconds: str = (
            f" ({(end_time - self._start_time).total_seconds():.2f} s. "
            f"start: {start_str}, end: {end_str})"
        )
        self._stage.append_name(took_seconds)

    def _raise_if_not_initialized(self) -> None:
        if self._run_id is None:
            raise ValueError("Run ID is not set")
        if self._stage is None:
            raise ValueError("Stage is not set")
        if self._start_time is None:
            raise ValueError("Start time is not set")

    async def on_chain_end(
        self,
        outputs: dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: t.Any,
    ) -> None:
        if self._debug_only and not self._show_debug_stages:
            return
        if not self._choice_present:
            return

        if run_id != self._run_id:
            return
        self._raise_if_not_initialized()
        assert self._stage is not None, "Stage must be set after initialization check"
        try:
            if self._content_appender is not None:
                await self._content_appender(self._stage, outputs)
        except Exception as e:
            logger.exception(f"An error occurred while populating the stage content: {repr(e)}")
            self._stage.append_content('An error occurred while populating the stage content.')
        finally:
            self._append_stage_timing()
            self._stage.close()

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: t.Any,
    ) -> None:
        if self._debug_only and not self._show_debug_stages:
            return
        if not self._choice_present:
            return
        if run_id != self._run_id:
            return

        self._raise_if_not_initialized()
        assert self._stage is not None, "Stage must be set after initialization check"
        try:
            self._stage.append_content(f"Error: {repr(error)}")
        finally:
            self._append_stage_timing()
            self._stage.close(status=Status.FAILED)

    @classmethod
    def create_config(
        cls,
        stage_name: str,
        content_appender: t.Callable[[Stage, dict[str, t.Any]], t.Awaitable[None]],
    ) -> RunnableConfig:
        return RunnableConfig(callbacks=[cls(stage_name, content_appender)])


class ChoiceCallback(AsyncCallbackHandler):
    _content_appender: t.Callable[[Choice, dict[str, t.Any]], t.Awaitable[t.NoReturn]]
    _run_id: UUID | None

    def __init__(
        self,
        content_appender: t.Callable[[Choice, dict[str, t.Any]], t.Awaitable[t.NoReturn]],
    ):
        self._content_appender = content_appender
        self._choice_present = False
        self._run_id = None

    async def on_chain_start(
        self,
        serialized: dict[str, t.Any],
        inputs: dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, t.Any] | None = None,
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
        outputs: dict[str, t.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
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
