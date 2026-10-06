"""Tests for `DeepResearchQueryChecker`, the LLM check that the user's query suits Deep Research
before a new Deep Research session is started (#732)."""

import asyncio
from unittest.mock import MagicMock

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.deep_research import DeepResearchQueryChecker
from statgpt.app.chains.deep_research import query_checker as query_checker_module
from statgpt.app.chains.deep_research.query_checker import DeepResearchQueryCheckResponse
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.default_prompts import deep_research_default_prompts
from statgpt.app.utils.dial_stages import DummyStage, NullChoice
from statgpt.app.utils.message_history import History
from statgpt.common.schemas import DeepResearchQueryCheckConfig


def _patch_model(
    monkeypatch, *, suits: bool = True, error: Exception | None = None, delay: float = 0
) -> list:
    """Replace the check's LLM; returns the list the prompted messages are appended to."""
    prompts: list = []

    async def _respond(prompt_value):
        prompts.append(prompt_value.to_messages())
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        return DeepResearchQueryCheckResponse(reasoning="because", suits_deep_research=suits)

    class _Model:
        def with_structured_output(self, schema, **kwargs):
            assert schema is DeepResearchQueryCheckResponse
            return RunnableLambda(_respond)

    monkeypatch.setattr(query_checker_module, "get_chat_model", lambda **kwargs: _Model())
    return prompts


class _StageRecordingChoice(NullChoice):
    def __init__(self) -> None:
        self.stage_content: list[str] = []

    def create_stage(self, *args, **kwargs):
        sink = self.stage_content

        class _RecordingStage(DummyStage):
            def append_content(self, content: str) -> None:
                sink.append(content)

            def __bool__(self) -> bool:
                # A real DIAL stage is truthy; the check writes to the stage only when it is.
                return True

        return _RecordingStage()


def _inputs(*, debug: bool = False, choice=None) -> dict:
    return {
        ChainParametersConfig.STATE: {StateVarsConfig.SHOW_DEBUG_STAGES: debug},
        ChainParametersConfig.CHOICE: choice if choice is not None else NullChoice(),
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(api_key="k"),
        ChainParametersConfig.HISTORY: History(
            messages=[
                DialMessage(role=Role.USER, content="Hi"),
                DialMessage(role=Role.ASSISTANT, content="Hello!"),
                DialMessage(role=Role.USER, content="What can you do?"),
            ]
        ),
    }


def _checker(**config: object) -> DeepResearchQueryChecker:
    return DeepResearchQueryChecker(DeepResearchQueryCheckConfig.model_validate(config))


async def test_returns_the_llm_decision(monkeypatch) -> None:
    _patch_model(monkeypatch, suits=False)
    assert await _checker().suits_deep_research(_inputs()) is False

    _patch_model(monkeypatch, suits=True)
    assert await _checker().suits_deep_research(_inputs()) is True


async def test_judges_the_conversation(monkeypatch) -> None:
    """The prompt is followed by the conversation, so the latest message is judged in context."""
    prompts = _patch_model(monkeypatch)

    await _checker().suits_deep_research(_inputs())

    assert [m.content for m in prompts[0][1:]] == ["Hi", "Hello!", "What can you do?"]


async def test_default_excluded_topics_are_used_when_unset(monkeypatch) -> None:
    prompts = _patch_model(monkeypatch)

    await _checker().suits_deep_research(_inputs())

    system_prompt = prompts[0][0].content
    assert "# Topics that do not suit Deep Research" in system_prompt
    for topic in deep_research_default_prompts.default_excluded_topics:
        assert topic in system_prompt


async def test_configured_excluded_topics_replace_the_defaults(monkeypatch) -> None:
    prompts = _patch_model(monkeypatch)

    await _checker(excluded_topics=["Weather {forecasts}"]).suits_deep_research(_inputs())

    system_prompt = prompts[0][0].content
    # Braces in a configured topic are kept verbatim, not treated as template placeholders.
    assert "1. Weather {forecasts}" in system_prompt
    for topic in deep_research_default_prompts.default_excluded_topics:
        assert topic not in system_prompt


async def test_empty_excluded_topics_omit_the_section(monkeypatch) -> None:
    prompts = _patch_model(monkeypatch)

    await _checker(excluded_topics=[]).suits_deep_research(_inputs())

    assert "# Topics that do not suit Deep Research" not in prompts[0][0].content


async def test_fails_open_when_the_check_fails(monkeypatch) -> None:
    """The check only spares an unnecessary run, so a failure honors the user's choice."""
    _patch_model(monkeypatch, error=RuntimeError("LLM unavailable"))

    assert await _checker().suits_deep_research(_inputs()) is True


async def test_fails_open_when_the_check_times_out(monkeypatch) -> None:
    """The deadline bounds the whole check, so a slow or unavailable model cannot hold the turn up."""
    _patch_model(monkeypatch, suits=False, delay=10)
    choice = _StageRecordingChoice()

    suits = await asyncio.wait_for(
        _checker(timeout_seconds=0.01).suits_deep_research(_inputs(debug=True, choice=choice)),
        timeout=1,
    )

    assert suits is True
    assert choice.stage_content == ["The check timed out, so Deep Research is started."]


async def test_debug_stage_shows_the_decision_and_reasoning(monkeypatch) -> None:
    _patch_model(monkeypatch, suits=False)
    choice = _StageRecordingChoice()

    await _checker().suits_deep_research(_inputs(debug=True, choice=choice))

    assert choice.stage_content == ["The query does not suit Deep Research, reasoning: because"]
