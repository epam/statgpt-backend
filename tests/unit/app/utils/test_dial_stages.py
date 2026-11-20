from statgpt.app.utils.dial_stages import DummyStage, delayed_timed_stage


def test_dummy_stage_content_stream_drops_writes():
    print("dropped", file=DummyStage().content_stream)


class _NameRecordingChoice:
    def __init__(self) -> None:
        self.names: list[str | None] = []

    def create_stage(self, name: str | None = None) -> DummyStage:
        self.names.append(name)
        return DummyStage()

    def append_content(self, content: str) -> None:
        pass


def test_delayed_stage_named_by_a_callable_uses_the_value_when_it_opens():
    choice = _NameRecordingChoice()
    current = {"query": "old"}

    with delayed_timed_stage(choice, name=lambda: f"Result: {current['query']}") as stage:
        current["query"] = "new"
        assert choice.names == []  # not opened before the first content
        stage.append_content("content")

    assert choice.names == ["Result: new"]


def test_delayed_stage_never_opened_never_renders_its_name():
    choice = _NameRecordingChoice()

    def fail() -> str:
        raise AssertionError("the name of an unopened stage must not be rendered")

    with delayed_timed_stage(choice, name=fail):
        pass

    assert choice.names == []


def test_delayed_stage_accepts_a_plain_name():
    choice = _NameRecordingChoice()

    with delayed_timed_stage(choice, name="Result") as stage:
        stage.append_content("content")

    assert choice.names == ["Result"]
