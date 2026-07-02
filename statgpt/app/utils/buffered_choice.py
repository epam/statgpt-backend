"""Record/replay proxies for the DIAL choice, used by speculative execution.

``BufferedChoice`` implements the ``ChoiceI`` protocol. While *buffering*, every
operation is recorded instead of being forwarded to the real choice; once the
speculation is resolved, the buffer is either replayed onto the real choice
(:meth:`BufferedChoice.flush_to`) or dropped (:meth:`BufferedChoice.discard`).

Distinct from ``dial_stages.BufferedStage``, which buffers stage *content* only:
the proxies here capture the full choice surface — including stage creation and
lifecycle — so a whole speculative agent run can be replayed verbatim.
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


class BufferedChoice(ChoiceI):
    """A ``ChoiceI`` proxy that records operations for a later replay.

    All recorded operations — both choice-level ones and those of the
    ``BufferedStage`` proxies it hands out — share a single ordered op list, so
    a replay reproduces the exact original write order (including nested
    stages). Timed stages compute their durations at record time, so timings
    shown after a replay stay correct.
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

    def create_stage(self, *args: Any, **kwargs: Any) -> Any:
        if self._mode is _Mode.PASS_THROUGH:
            return self._real.create_stage(*args, **kwargs)
        stage = BufferedStage(self)
        if self._mode is _Mode.BUFFERING:
            self._record(lambda real: stage._attach(real.create_stage(*args, **kwargs)))
        return stage

    def append_content(self, content: str) -> None:
        if self._mode is _Mode.PASS_THROUGH:
            self._real.append_content(content)
        elif self._mode is _Mode.BUFFERING:
            self._record(lambda real: real.append_content(content))

    def add_attachment(self, *args: Any, **kwargs: Any) -> None:
        if self._mode is _Mode.PASS_THROUGH:
            self._real.add_attachment(*args, **kwargs)
        elif self._mode is _Mode.BUFFERING:
            self._record(lambda real: real.add_attachment(*args, **kwargs))

    def set_state(self, state: dict) -> None:
        if self._mode is _Mode.PASS_THROUGH:
            self._real.set_state(state)
        elif self._mode is _Mode.BUFFERING:
            self._record(lambda real: real.set_state(state))

    def flush_to(self, real_choice: ChoiceI) -> None:
        """Replay the recorded operations onto ``real_choice``, in order, then
        switch self (and all issued stages) to pass-through delegation.

        Deliberately synchronous: the replay and the mode flip happen without
        yielding to the event loop, so a still-running speculative task cannot
        interleave writes mid-flush. Idempotent — a second call is a no-op.
        """
        if self._mode is not _Mode.BUFFERING:
            return
        self._real_choice = real_choice
        for op in self._ops:
            op(real_choice)
        self._ops.clear()
        self._mode = _Mode.PASS_THROUGH

    def discard(self) -> None:
        """Drop the buffer; subsequent operations (e.g. a final write from a
        cancelled task) become no-ops and never touch the real choice."""
        if self._mode is not _Mode.BUFFERING:
            return
        self._ops.clear()
        self._mode = _Mode.DISCARDED


class BufferedStage(StageI):
    """Stage proxy handed out by :meth:`BufferedChoice.create_stage`.

    Records its operations into the parent choice's shared op list (preserving
    the global write order). During the replay the parent attaches the real
    stage; afterwards all calls delegate to it directly. After ``discard()`` on
    the parent, every call is a no-op.
    """

    def __init__(self, parent: BufferedChoice) -> None:
        self._parent = parent
        # The real stage created during the replay; ops recorded earlier resolve
        # it lazily, so recording order equals replay order.
        self._real_stage: Any = None

    def _attach(self, real_stage: Any) -> None:
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
        # Streaming from a not-yet-materialized stage is not possible; nothing
        # in the speculative agent path uses `content_stream`.
        if self._parent._mode is _Mode.PASS_THROUGH:
            return self._real_stage.content_stream
        raise NotImplementedError("content_stream is not available on a buffered stage")

    def __enter__(self) -> 'BufferedStage':
        self._dispatch(lambda stage: stage.__enter__())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        # In buffering mode the recorded op replays the original exc info, so
        # the real stage closes with the same status; the record-time exception
        # keeps propagating here (returns None).
        return self._dispatch(lambda stage: stage.__exit__(exc_type, exc_val, exc_tb))
