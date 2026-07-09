"""Record/replay proxies for the DIAL choice, used by speculative execution.

While buffering, ``RecordingChoice`` records every choice/stage operation instead
of forwarding it; on resolution the buffer is replayed onto the real choice
(:meth:`RecordingChoice.flush_to`) or dropped (:meth:`RecordingChoice.discard`).
"""

import enum
from collections.abc import Callable
from typing import Any

from aidial_sdk.chat_completion.enums import Status
from aidial_sdk.chat_completion.stage import ContentStream

from statgpt.app.utils.dial_stages import ChoiceI, StageI


class _Mode(enum.Enum):
    BUFFERING = "buffering"
    PASS_THROUGH = "pass_through"
    DISCARDED = "discarded"


class RecordingChoice(ChoiceI):
    """A ``ChoiceI`` proxy that records operations for a later replay.

    Choice-level ops and those of the ``RecordingStage`` proxies it hands out share
    one ordered op list, so a replay reproduces the original write order (including
    nested stages).
    """

    def __init__(self) -> None:
        self._mode = _Mode.BUFFERING
        self._ops: list[Callable[[ChoiceI], Any]] = []
        self._real_choice: ChoiceI | None = None

    @property
    def _real(self) -> ChoiceI:
        assert self._real_choice is not None, "flush_to() must be called before pass-through use"
        return self._real_choice

    def _record(self, op: Callable[[ChoiceI], Any]) -> None:
        self._ops.append(op)

    def _dispatch(self, op: Callable[[ChoiceI], Any]) -> Any:
        if self._mode is _Mode.PASS_THROUGH:
            return op(self._real)
        if self._mode is _Mode.BUFFERING:
            self._record(op)
        return None

    def create_stage(self, *args: Any, **kwargs: Any) -> Any:
        if self._mode is _Mode.PASS_THROUGH:
            return self._real.create_stage(*args, **kwargs)
        stage = RecordingStage(self)
        if self._mode is _Mode.BUFFERING:
            self._record(lambda real: stage._attach(real.create_stage(*args, **kwargs)))
        return stage

    def adopt_stage(self, real_stage: StageI) -> "RecordingStage":
        """Proxy a stage that already exists on the real choice (e.g. the shared
        performance stage) so writes to it are buffered with the rest and replay in
        global recording order. No ``create_stage`` op is recorded.
        """
        stage = RecordingStage(self)
        stage._attach(real_stage)
        return stage

    def append_content(self, content: str) -> None:
        self._dispatch(lambda real: real.append_content(content))

    def add_attachment(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch(lambda real: real.add_attachment(*args, **kwargs))

    def set_state(self, state: dict) -> None:
        self._dispatch(lambda real: real.set_state(state))

    def flush_to(self, real_choice: ChoiceI) -> None:
        """Replay the recorded ops onto ``real_choice`` in order, then switch to
        pass-through. Synchronous on purpose: no ``await`` between replay and the mode
        flip, so a running speculative task can't interleave writes. Idempotent.
        """
        if self._mode is not _Mode.BUFFERING:
            return
        self._real_choice = real_choice
        for op in self._ops:
            op(real_choice)
        self._ops.clear()
        self._mode = _Mode.PASS_THROUGH

    def discard(self) -> None:
        """Drop the buffer; later ops become no-ops and never touch the real choice."""
        if self._mode is not _Mode.BUFFERING:
            return
        self._ops.clear()
        self._mode = _Mode.DISCARDED


class RecordingStage(StageI):
    """Stage proxy handed out by ``RecordingChoice``.

    Records into the parent's shared op list; on replay the parent attaches the real
    stage and calls delegate to it, after discard they are no-ops.
    """

    def __init__(self, parent: RecordingChoice) -> None:
        self._parent = parent
        # Attached lazily during replay (recorded ops resolve it then); pre-set by
        # ``adopt_stage``.
        self._real_stage: StageI | None = None

    def _attach(self, real_stage: StageI) -> None:
        self._real_stage = real_stage

    def _dispatch(self, op: Callable[[Any], Any]) -> Any:
        mode = self._parent._mode
        if mode is _Mode.PASS_THROUGH:
            return op(self._real_stage)
        if mode is _Mode.BUFFERING:
            self._parent._record(lambda _real_choice: op(self._real_stage))
        return None

    def append_content(self, content: str) -> None:
        self._dispatch(lambda stage: stage.append_content(content))

    def append_name(self, name: str) -> None:
        self._dispatch(lambda stage: stage.append_name(name))

    def add_attachment(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch(lambda stage: stage.add_attachment(*args, **kwargs))

    def open(self) -> None:
        self._dispatch(lambda stage: stage.open())

    def close(self, status: Status = Status.COMPLETED) -> None:
        self._dispatch(lambda stage: stage.close(status))

    @property
    def content_stream(self) -> ContentStream:
        # Unavailable while buffering; nothing on the speculative path uses it.
        if self._parent._mode is _Mode.PASS_THROUGH:
            assert self._real_stage is not None, "pass-through implies an attached real stage"
            return self._real_stage.content_stream
        raise NotImplementedError("content_stream is not available on a recording stage")

    def __enter__(self) -> 'RecordingStage':
        self._dispatch(lambda stage: stage.__enter__())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        # Recorded op replays this exc info, so the real stage closes with the same status.
        return self._dispatch(lambda stage: stage.__exit__(exc_type, exc_val, exc_tb))
