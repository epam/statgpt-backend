"""Stage names of the Deep Research tools.

The resume tool is built from the start tool's config but is called with a `message`, so it has its
own `resumeStagesConfig` templates and exposes the session's original question — which lives in the
conversation state, not in the tool arguments.
"""

import pytest

from statgpt.app.chains.deep_research import ResumeDeepResearchTool
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DataQueryTool, DeepResearchTool


def _channel_config(details_overrides: dict | None = None) -> ChannelConfig:
    details: dict = {"deployment_id": "dr-app"} | (details_overrides or {})
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(name="X", domain="d", terminology_domain="t"),
        deep_research=DeepResearchTool(
            name="deep_research", description="DR", enabled=True, details=details
        ),
        data_query=DataQueryTool(name="data_query", description="DQ", enabled=True, details={}),
    )


def _inputs(session: DeepResearchSession | None) -> dict:
    state: dict = {}
    if session is not None:
        state[StateVarsConfig.DEEP_RESEARCH_SESSION] = session.model_dump(mode='json')
    return {ChainParametersConfig.STATE: state}


def _session(*questions: str) -> DeepResearchSession:
    return DeepResearchSession(
        turns=[
            DeepResearchTurn(user_message=q, assistant_content="a", deep_research_state={})
            for q in questions
        ]
    )


def _render(tool: StatGptTool, args: dict, inputs: dict) -> tuple[str, str]:
    return tool.render_stage_name(args, inputs), tool.render_result_stage_name(args, inputs)


def _start_tool(stages_config: dict) -> StatGptTool:
    channel_config = _channel_config({"stagesConfig": stages_config})
    assert channel_config.deep_research is not None
    return StatGptTool.from_config(channel_config.deep_research, channel_config)


def _resume_tool(details_overrides: dict | None = None) -> ResumeDeepResearchTool:
    channel_config = _channel_config(details_overrides)
    assert channel_config.deep_research is not None
    return ResumeDeepResearchTool.build(channel_config.deep_research, channel_config)


class TestResumeStagesConfigDefaults:
    def test_defaults_applied_when_block_is_omitted(self):
        config = _resume_tool()._resume_stages_config
        assert config.tool_call_name == "Researching: {original_question}"
        assert config.tool_result_name == "Research result: {original_question}"
        assert config.missing_question_name == "Continue researching"

    def test_camel_case_overrides_parse(self):
        config = _resume_tool(
            {
                "resumeStagesConfig": {
                    "toolCallName": "Resuming: {message}",
                    "toolResultName": "Resumed: {message}",
                    "missingQuestionName": "Carrying on",
                }
            }
        )._resume_stages_config
        assert config.tool_call_name == "Resuming: {message}"
        assert config.tool_result_name == "Resumed: {message}"
        assert config.missing_question_name == "Carrying on"


class TestResumeStageNames:
    def test_original_question_comes_from_the_session(self):
        tool = _resume_tool()
        call_name, result_name = _render(
            tool, {"message": "yes, go ahead"}, _inputs(_session("GDP of Ukraine", "follow-up"))
        )
        assert call_name == "Researching: GDP of Ukraine"
        assert result_name == "Research result: GDP of Ukraine"

    def test_message_placeholder_renders_the_tool_argument(self):
        tool = _resume_tool({"resumeStagesConfig": {"toolCallName": "Resuming: {message}"}})
        call_name, _ = _render(tool, {"message": "yes, go ahead"}, _inputs(_session("q")))
        assert call_name == "Resuming: yes, go ahead"

    @pytest.mark.parametrize("session", [None, DeepResearchSession()])
    def test_missing_question_falls_back_to_the_configured_name(
        self, session: DeepResearchSession | None
    ):
        tool = _resume_tool({"resumeStagesConfig": {"missingQuestionName": "Carrying on"}})
        call_name, result_name = _render(tool, {"message": "m"}, _inputs(session))
        assert call_name == "Carrying on"
        assert result_name == "Carrying on"

    def test_fallback_leaves_templates_without_the_question_alone(self):
        tool = _resume_tool(
            {
                "resumeStagesConfig": {
                    "toolCallName": "Resuming: {message}",
                    "missingQuestionName": "Carrying on",
                }
            }
        )
        call_name, result_name = _render(tool, {"message": "m"}, _inputs(None))
        assert call_name == "Resuming: m"
        assert result_name == "Carrying on"

    def test_start_tool_keeps_its_own_config(self):
        tool = _start_tool({"toolCallName": "Researching: {query}"})
        call_name, _ = _render(tool, {"query": "GDP of Ukraine"}, _inputs(None))
        assert call_name == "Researching: GDP of Ukraine"


class TestStageNameRendering:
    """The default rendering on `StatGptTool`, exercised through the Deep Research start tool."""

    def test_unknown_placeholder_keeps_the_raw_name(self):
        tool = _start_tool({"toolCallName": "Researching: {query}"})
        call_name, _ = _render(tool, {"message": "m"}, _inputs(None))
        assert call_name == "Researching: {query}"

    def test_malformed_template_keeps_the_raw_name(self):
        tool = _start_tool({"toolCallName": "Researching: {"})
        call_name, _ = _render(tool, {"query": "q"}, _inputs(None))
        assert call_name == "Researching: {"

    def test_lists_are_joined(self):
        tool = _start_tool({"toolCallName": "Terms: {terms}"})
        call_name, _ = _render(tool, {"terms": ["a", "b"]}, _inputs(None))
        assert call_name == "Terms: a, b"

    def test_unconfigured_names_fall_back_to_the_tool_name(self):
        tool = _start_tool({})
        call_name, result_name = _render(tool, {"query": "q"}, _inputs(None))
        assert call_name == "Calling deep research tool"
        assert result_name == "Result from deep research tool"
