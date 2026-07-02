from aidial_sdk.chat_completion.enums import Status

from statgpt.app.utils.buffered_choice import BufferedChoice, BufferedStage


class FakeStage:
    """Records every stage operation into the shared owner log."""

    def __init__(self, log: list, name: str | None):
        self._log = log
        self._name = name

    def append_content(self, content: str):
        self._log.append(("stage_content", self._name, content))

    def append_name(self, name: str):
        self._log.append(("stage_name", self._name, name))

    def add_attachment(self, *args, **kwargs):
        self._log.append(("stage_attachment", self._name, args, kwargs))

    def open(self):
        self._log.append(("stage_open", self._name))

    def close(self, status: Status = Status.COMPLETED):
        self._log.append(("stage_close", self._name, status))

    def __enter__(self):
        self._log.append(("stage_enter", self._name))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.append(("stage_exit", self._name, exc_type))
        return False


class FakeChoice:
    """Records every choice operation, in order, into a single log."""

    def __init__(self):
        self.log: list = []

    def create_stage(self, name: str | None = None) -> FakeStage:
        self.log.append(("create_stage", name))
        return FakeStage(self.log, name)

    def append_content(self, content: str):
        self.log.append(("content", content))

    def add_attachment(self, *args, **kwargs):
        self.log.append(("attachment", args, kwargs))

    def set_state(self, state: dict):
        self.log.append(("state", state))


def test_flush_replays_ops_in_recording_order_including_nested_stages():
    buffered = BufferedChoice()

    buffered.append_content("before")
    with buffered.create_stage("outer") as outer:
        outer.append_content("outer-1")
        with buffered.create_stage("inner") as inner:
            inner.append_name(" (0.5s)")
            inner.append_content("inner-1")
        outer.add_attachment(title="table", data="csv")
        outer.append_content("outer-2")
    buffered.add_attachment(title="chart")
    buffered.append_content("after")

    real = FakeChoice()
    assert real.log == []  # nothing reaches the real choice while buffering

    buffered.flush_to(real)

    assert real.log == [
        ("content", "before"),
        ("create_stage", "outer"),
        ("stage_enter", "outer"),
        ("stage_content", "outer", "outer-1"),
        ("create_stage", "inner"),
        ("stage_enter", "inner"),
        ("stage_name", "inner", " (0.5s)"),
        ("stage_content", "inner", "inner-1"),
        ("stage_exit", "inner", None),
        ("stage_attachment", "outer", (), {"title": "table", "data": "csv"}),
        ("stage_content", "outer", "outer-2"),
        ("stage_exit", "outer", None),
        ("attachment", (), {"title": "chart"}),
        ("content", "after"),
    ]


def test_explicit_stage_lifecycle_is_replayed():
    buffered = BufferedChoice()

    stage = buffered.create_stage("manual")
    stage.open()
    stage.append_content("body")
    stage.close(Status.FAILED)

    real = FakeChoice()
    buffered.flush_to(real)

    assert real.log == [
        ("create_stage", "manual"),
        ("stage_open", "manual"),
        ("stage_content", "manual", "body"),
        ("stage_close", "manual", Status.FAILED),
    ]


def test_pass_through_after_flush():
    buffered = BufferedChoice()
    stage = buffered.create_stage("pre-flush")
    stage.append_content("buffered")

    real = FakeChoice()
    buffered.flush_to(real)
    real.log.clear()

    buffered.append_content("live")
    stage.append_content("live-stage")  # a stage issued before the flush delegates too
    post_flush_stage = buffered.create_stage("post-flush")

    assert isinstance(post_flush_stage, FakeStage)  # real stage, not a proxy
    assert real.log == [
        ("content", "live"),
        ("stage_content", "pre-flush", "live-stage"),
        ("create_stage", "post-flush"),
    ]


def test_noop_after_discard():
    buffered = BufferedChoice()
    buffered.append_content("speculative")
    stage = buffered.create_stage("stage")
    stage.append_content("speculative-stage")

    buffered.discard()

    # A cancelled task racing a final write must not touch the real choice.
    buffered.append_content("late")
    stage.append_content("late-stage")
    late_stage = buffered.create_stage("late")
    late_stage.append_content("ignored")

    real = FakeChoice()
    buffered.flush_to(real)  # flush after discard is a no-op

    assert isinstance(late_stage, BufferedStage)
    assert real.log == []


def test_flush_is_idempotent():
    buffered = BufferedChoice()
    buffered.append_content("once")

    real = FakeChoice()
    buffered.flush_to(real)
    buffered.flush_to(real)

    assert real.log == [("content", "once")]
