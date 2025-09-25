import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

from aidial_sdk.chat_completion import Attachment, Choice, Stage
from aidial_sdk.chat_completion.enums import Status
from aidial_sdk.chat_completion.stage import ChunkQueue, ContentStream

from statgpt.settings.dial_app import dial_app_settings

_log = logging.getLogger(__name__)


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
    """A dummy stage that does nothing."""

    def append_content(self, content: str):
        _log.warning("The content is being appended to a dummy stage and will be ignored.")

    def append_name(self, name: str):
        _log.warning("The name is being appended to a dummy stage and will be ignored.")

    def add_attachment(self, *args, **kwargs):
        _log.warning("The attachment is being added to a dummy stage and will be ignored.")

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
def timed_stage(choice: Choice, *args, **kwargs):
    """Context manager for creating a timed stage."""
    stage_generator = choice.create_stage(*args, **kwargs)
    with _add_timing_to_stage(stage_generator) as stage:
        yield stage


@contextmanager
def delayed_timed_stage(choice: Choice, *args, **kwargs):
    """Context manager for creating a delayed timed stage."""
    stage_generator = DelayedStage(lambda: choice.create_stage(*args, **kwargs))
    with _add_timing_to_stage(stage_generator) as stage:
        yield stage


@contextmanager
def optional_stage(stage_generator: Generator, enabled: bool):
    if not enabled:
        # Create a dummy stage that does nothing
        stage_generator = DummyStage()

    with stage_generator as stage:
        yield stage


@contextmanager
def optional_timed_stage(choice: Choice, *args, enabled: bool, **kwargs):
    """Context manager for creating an optional timed stage."""
    stage_generator = timed_stage(choice, *args, **kwargs)
    with optional_stage(stage_generator, enabled) as stage:
        yield stage


@contextmanager
def optional_delayed_timed_stage(choice: Choice, *args, enabled: bool, **kwargs):
    """Context manager for creating an optional delayed timed stage."""
    stage_generator = delayed_timed_stage(choice, *args, **kwargs)
    with optional_stage(stage_generator, enabled) as stage:
        yield stage
