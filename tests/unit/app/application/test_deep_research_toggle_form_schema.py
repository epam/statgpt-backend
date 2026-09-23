"""Tests for the per-message Deep Research toggle form schema (#683).

Once Deep Research delivers its final report the session is dropped, but the "Deep research"
toggle in the DIAL Chat UI stays armed because the deployment `configuration` schema is static per
deployment. To disarm it without extra user interaction, every assistant turn emits a
`custom_content.form_schema` whose `deep_research` control carries — via `const` — the value the
toggle should hold on the next request:

- report delivered this turn  -> off (disarm, so the follow-up hits the normal, cheap agent);
- run still in progress        -> on  (clarification / plan turns keep the toggle armed);
- otherwise                    -> mirror the user's current selection (stays re-armable).

These tests pin the schema shape and that deterministic resolution.
"""

from unittest.mock import AsyncMock, MagicMock

from statgpt.app.application.channel_completion import ChannelCompletion
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.services.chat_facade import build_deep_research_form_schema


def _session_state(*turns: DeepResearchTurn) -> dict:
    return {
        StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession(turns=list(turns)).model_dump(
            mode="json"
        )
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ form schema shape ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_form_schema_reuses_configuration_field_name_and_title() -> None:
    """The control must reuse the configuration field's name and title so the frontend updates the
    existing toggle rather than rendering a separate control."""
    schema = build_deep_research_form_schema(True)

    prop = schema["properties"]["deep_research"]
    assert prop["title"] == "Deep research"
    assert prop["type"] == "boolean"
    assert prop["dial:widget"] == "buttons"


def test_form_schema_carries_on_value_via_const() -> None:
    schema = build_deep_research_form_schema(True)

    options = schema["properties"]["deep_research"]["oneOf"]
    assert [opt["const"] for opt in options] == [True]


def test_form_schema_carries_off_value_via_const() -> None:
    schema = build_deep_research_form_schema(False)

    options = schema["properties"]["deep_research"]["oneOf"]
    assert [opt["const"] for opt in options] == [False]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ toggle resolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_report_delivered_this_turn_disarms_even_if_selection_was_on() -> None:
    """The delivery turn disarms the toggle; without this the next message would silently launch a
    fresh, expensive run. The report-delivered flag wins over the (still true) user selection."""
    state = {StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED: True}
    configuration = StatGPTConfiguration(deep_research=True)

    assert ChannelCompletion._resolve_deep_research_toggle(state, configuration) is False


def test_run_in_progress_stays_armed() -> None:
    """A clarification / plan-for-approval turn keeps the session alive, so the toggle stays on."""
    state = _session_state(DeepResearchTurn(user_message="q", assistant_content="which region?"))
    configuration = StatGPTConfiguration(deep_research=True)

    assert ChannelCompletion._resolve_deep_research_toggle(state, configuration) is True


def test_normal_turn_mirrors_user_selection_off() -> None:
    assert (
        ChannelCompletion._resolve_deep_research_toggle(
            {}, StatGPTConfiguration(deep_research=False)
        )
        is False
    )


def test_normal_turn_mirrors_user_selection_on() -> None:
    """A deliberately re-armed toggle with no session yet (e.g. a run that errored before saving a
    session, left as-is for retry) keeps the toggle on."""
    assert (
        ChannelCompletion._resolve_deep_research_toggle(
            {}, StatGPTConfiguration(deep_research=True)
        )
        is True
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ emission + gating ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _service(*, available: bool) -> MagicMock:
    service = MagicMock()
    service.channel_config.is_deep_research_available_for = AsyncMock(return_value=available)
    return service


async def test_emits_form_schema_when_available() -> None:
    choice = MagicMock()
    state = _session_state(DeepResearchTurn(user_message="q", assistant_content="a"))

    await ChannelCompletion._emit_deep_research_form_schema(
        _service(available=True),
        auth_context=MagicMock(),
        state=state,
        configuration=StatGPTConfiguration(deep_research=True),
        choice=choice,
    )

    choice.set_form_schema.assert_called_once()
    emitted = choice.set_form_schema.call_args.args[0]
    assert emitted["properties"]["deep_research"]["oneOf"][0]["const"] is True


async def test_skips_emission_when_deep_research_unavailable() -> None:
    """No toggle exists in the UI when Deep Research is unavailable, so nothing is emitted."""
    choice = MagicMock()

    await ChannelCompletion._emit_deep_research_form_schema(
        _service(available=False),
        auth_context=MagicMock(),
        state={},
        configuration=StatGPTConfiguration(deep_research=False),
        choice=choice,
    )

    choice.set_form_schema.assert_not_called()
