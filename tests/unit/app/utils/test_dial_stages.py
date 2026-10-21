from statgpt.app.utils.dial_stages import DummyStage


def test_dummy_stage_content_stream_drops_writes():
    print("dropped", file=DummyStage().content_stream)
