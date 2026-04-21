import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from aidial_sdk.chat_completion import Attachment, Stage
from aidial_sdk.chat_completion.enums import Status
from aidial_sdk.chat_completion.stage import ChunkQueue, ContentStream

from statgpt.app.settings.dial_app import dial_app_settings

_log = logging.getLogger(__name__)


@runtime_checkable
class ChoiceI(Protocol):
    """Structural protocol for Choice-like objects.

    Both ``aidial_sdk.chat_completion.Choice`` and ``NullChoice`` satisfy this
    protocol, so Pydantic's ``isinstance`` check (used with
    ``arbitrary_types_allowed``) passes for either.
    """

    def create_stage(self, *args: Any, **kwargs: Any) -> Any: ...

    def append_content(self, content: str) -> Any: ...

    def add_attachment(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_state(self, state: dict) -> Any: ...


class StageInterface(ABC):
    """Abstract interface for Stage-like classes."""

    @abstractmethod
    def append_content(self, content: str):
        pass

    @abstractmethod
    def append_name(self, name: str):
        pass

    @abstractmethod
    def add_attachment(self, *args, **kwargs):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self, status: Status = Status.COMPLETED):
        pass

    @property
    @abstractmethod
    def content_stream(self) -> ContentStream:
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DelayedStage(StageInterface):
    """
    A Stage that delays the opening of the stage (and appending the name) until the first content is added.
    """

    def __init__(
        self,
        stage_generator: Callable[[], Stage],
        name: str | None = None,
    ):
        self._stage_generator = stage_generator
        self._name = name
        self._actual_stage: Stage | None = None
        self._is_opened = False
        self._is_closed = False
        self._pending_content: list[str] = []
        self._pending_names: list[str] = []
        self._pending_attachments: list[Attachment] = []

    def _ensure_opened(self):
        if not self._is_opened and not self._is_closed:
            self._actual_stage = self._stage_generator()
            self._actual_stage.open()
            self._is_opened = True

            for name in self._pending_names:
                self._actual_stage.append_name(name)
            self._pending_names.clear()

            for content in self._pending_content:
                self._actual_stage.append_content(content)
            self._pending_content.clear()

            for args, kwargs in self._pending_attachments:
                self._actual_stage.add_attachment(*args, **kwargs)
            self._pending_attachments.clear()

    def append_content(self, content: str):
        if not self._is_opened:
            self._ensure_opened()
        if self._actual_stage:
            self._actual_stage.append_content(content)

    def append_name(self, name: str):
        if not self._is_opened:
            self._pending_names.append(name)
        elif self._actual_stage:
            self._actual_stage.append_name(name)

    def add_attachment(self, *args, **kwargs):
        if not self._is_opened:
            self._pending_attachments.append((args, kwargs))
        elif self._actual_stage:
            self._actual_stage.add_attachment(*args, **kwargs)

    def open(self):
        if not self._is_opened:
            self._ensure_opened()

    def close(self, status: Status = Status.COMPLETED):
        if self._is_opened and self._actual_stage:
            self._actual_stage.close(status)
            self._is_closed = True
        elif not self._is_opened:
            # If stage was never opened, we don't need to close it
            self._is_closed = True

    @property
    def content_stream(self) -> ContentStream:
        if not self._is_opened:
            self._ensure_opened()
        if self._actual_stage:
            return self._actual_stage.content_stream
        # Fallback, though this should not happen if _ensure_opened works
        return ContentStream(self)

    def __enter__(self):
        # Don't open immediately - wait for first content
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            if self._is_opened and not self._is_closed:
                self.close(Status.COMPLETED)
        else:
            if self._is_opened:
                self.close(Status.FAILED)
        return False


class DummyStage(StageInterface):
    """A silent dummy stage that does nothing."""

    def append_content(self, content: str):
        pass

    def append_name(self, name: str):
        pass

    def add_attachment(self, *args, **kwargs):
        pass

    def open(self):
        pass

    def close(self, status: Status = Status.COMPLETED):
        pass

    @property
    def content_stream(self) -> ContentStream:
        # Return an empty ContentStream
        queue: ChunkQueue = asyncio.Queue()
        dummy_stage = Stage(queue, 0, 0, name="Dummy Stage")
        return dummy_stage.content_stream

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __bool__(self):
        return False


class WarningDummyStage(DummyStage):
    """A dummy stage that logs warnings when content is appended."""

    def append_content(self, content: str):
        _log.warning("The content is being appended to a dummy stage and will be ignored.")

    def append_name(self, name: str):
        _log.warning("The name is being appended to a dummy stage and will be ignored.")

    def add_attachment(self, *args, **kwargs):
        _log.warning("The attachment is being added to a dummy stage and will be ignored.")


class NullChoice:
    """A no-op Choice replacement for contexts without DIAL streaming (e.g., MCP)."""

    def create_stage(self, *args, **kwargs):
        return DummyStage()

    def append_content(self, content: str):
        pass

    def add_attachment(self, *args, **kwargs):
        _log.warning("add_attachment() called on NullChoice — ignored in MCP context.")

    def set_state(self, state: dict):
        _log.warning("set_state() called on NullChoice — tools should not call this.")


@contextmanager
def _add_timing_to_stage(stage_generator):
    """Internal context manager that adds timing information to any stage."""
    with stage_generator as stage:
        start_time = datetime.now()
        start = perf_counter()
        try:
            yield stage
        finally:
            end = perf_counter()
            end_time = datetime.now()

            if dial_app_settings.dial_show_stage_seconds:
                start_str = start_time.strftime('%H:%M:%S')
                end_str = end_time.strftime('%H:%M:%S')
                stage.append_name(f" ({end - start:.2f}s, start: {start_str}, end: {end_str})")


@contextmanager
def timed_stage(choice: ChoiceI, *args, **kwargs):
    """Context manager for creating a timed stage."""
    stage_generator = choice.create_stage(*args, **kwargs)
    with _add_timing_to_stage(stage_generator) as stage:
        yield stage


@contextmanager
def delayed_timed_stage(choice: ChoiceI, *args, **kwargs):
    """Context manager for creating a delayed timed stage."""
    stage_generator = DelayedStage(lambda: choice.create_stage(*args, **kwargs))
    with _add_timing_to_stage(stage_generator) as stage:
        yield stage


@contextmanager
def optional_stage(stage_generator: AbstractContextManager[StageInterface], enabled: bool):
    if not enabled:
        # Create a dummy stage that logs warnings
        stage_generator = WarningDummyStage()

    with stage_generator as stage:
        yield stage


@contextmanager
def optional_timed_stage(choice: ChoiceI, *args, enabled: bool, **kwargs):
    """Context manager for creating an optional timed stage."""
    stage_generator = timed_stage(choice, *args, **kwargs)
    with optional_stage(stage_generator, enabled) as stage:
        yield stage


@contextmanager
def optional_delayed_timed_stage(choice: ChoiceI, *args, enabled: bool, **kwargs):
    """Context manager for creating an optional delayed timed stage."""
    stage_generator = delayed_timed_stage(choice, *args, **kwargs)
    with optional_stage(stage_generator, enabled) as stage:
        yield stage
