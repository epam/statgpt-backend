from aidial_sdk.chat_completion.enums import Status

from statgpt.app.utils.recording_choice import RecordingChoice, RecordingStage


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
    recording = RecordingChoice()

    recording.append_content("before")
    with recording.create_stage("outer") as outer:
        outer.append_content("outer-1")
        with recording.create_stage("inner") as inner:
            inner.append_name(" (0.5s)")
            inner.append_content("inner-1")
        outer.add_attachment(title="table", data="csv")
        outer.append_content("outer-2")
    recording.add_attachment(title="chart")
    recording.append_content("after")

    real = FakeChoice()
    assert real.log == []  # nothing reaches the real choice while buffering

    recording.flush_to(real)

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
    recording = RecordingChoice()

    stage = recording.create_stage("manual")
    stage.open()
    stage.append_content("body")
    stage.close(Status.FAILED)

    real = FakeChoice()
    recording.flush_to(real)

    assert real.log == [
        ("create_stage", "manual"),
        ("stage_open", "manual"),
        ("stage_content", "manual", "body"),
        ("stage_close", "manual", Status.FAILED),
    ]


def test_pass_through_after_flush():
    recording = RecordingChoice()
    stage = recording.create_stage("pre-flush")
    stage.append_content("buffered")

    real = FakeChoice()
    recording.flush_to(real)
    real.log.clear()

    recording.append_content("live")
    stage.append_content("live-stage")  # a stage issued before the flush delegates too
    post_flush_stage = recording.create_stage("post-flush")

    assert isinstance(post_flush_stage, FakeStage)  # real stage, not a proxy
    assert real.log == [
        ("content", "live"),
        ("stage_content", "pre-flush", "live-stage"),
        ("create_stage", "post-flush"),
    ]


def test_noop_after_discard():
    recording = RecordingChoice()
    recording.append_content("speculative")
    stage = recording.create_stage("stage")
    stage.append_content("speculative-stage")

    recording.discard()

    # A cancelled task racing a final write must not touch the real choice.
    recording.append_content("late")
    stage.append_content("late-stage")
    late_stage = recording.create_stage("late")
    late_stage.append_content("ignored")

    real = FakeChoice()
    recording.flush_to(real)  # flush after discard is a no-op

    assert isinstance(late_stage, RecordingStage)
    assert real.log == []


def test_flush_is_idempotent():
    recording = RecordingChoice()
    recording.append_content("once")

    real = FakeChoice()
    recording.flush_to(real)
    recording.flush_to(real)

    assert real.log == [("content", "once")]


# ~~~ adopt_stage: proxy a stage that already exists on the real choice ~~~


def test_adopt_stage_buffers_until_flush_with_no_create_stage_op():
    real = FakeChoice()
    # a stage that was created on the real choice before the speculative run
    perf = FakeStage(real.log, "perf")

    recording = RecordingChoice()
    proxy = recording.adopt_stage(perf)

    recording.append_content("before")
    proxy.append_content("perf-row-1")
    recording.append_content("after")
    proxy.append_content("perf-row-2")

    assert real.log == []  # held back while buffering

    recording.flush_to(real)

    # replayed in global recording order; no ("create_stage", ...) op for the adopted stage
    assert real.log == [
        ("content", "before"),
        ("stage_content", "perf", "perf-row-1"),
        ("content", "after"),
        ("stage_content", "perf", "perf-row-2"),
    ]


def test_adopt_stage_passes_through_after_flush():
    real = FakeChoice()
    perf = FakeStage(real.log, "perf")

    recording = RecordingChoice()
    proxy = recording.adopt_stage(perf)
    recording.flush_to(real)
    real.log.clear()

    proxy.append_content("live-row")

    assert real.log == [("stage_content", "perf", "live-row")]


def test_adopt_stage_noop_after_discard():
    real = FakeChoice()
    perf = FakeStage(real.log, "perf")

    recording = RecordingChoice()
    proxy = recording.adopt_stage(perf)
    proxy.append_content("speculative-row")

    recording.discard()

    proxy.append_content("late-row")
    recording.flush_to(real)  # no-op

    assert real.log == []
