"""Tests for the per-message Deep Research toggle form schema (#683).

Once Deep Research delivers its final report the session is dropped, but the "Deep research"
toggle in the DIAL Chat UI stays armed because the deployment `configuration` schema is static per
deployment. To disarm it without extra user interaction, every assistant turn emits a
`custom_content.form_schema` whose `deep_research` control carries — via `const` — the value the
toggle should hold on the next request:

- report delivered this turn  -> off (disarm, so the follow-up hits the normal, cheap agent);
- query check skipped the run  -> off (the query was answered directly, #745);
- run still in progress        -> on  (clarification / plan turns keep the toggle armed);
- otherwise                    -> mirror the user's current selection (stays re-armable).

These tests pin the schema shape and that deterministic resolution.
"""

from unittest.mock import AsyncMock, MagicMock

from statgpt.app.application.channel_completion import ChannelCompletion
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.services.chat_facade import build_message_form_schema


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
    schema = build_message_form_schema(deep_research=True, continue_button=False)

    prop = schema["properties"]["deep_research"]
    assert prop["title"] == "Deep research"
    assert prop["type"] == "boolean"
    assert prop["dial:widget"] == "buttons"


def test_form_schema_carries_on_value_via_const() -> None:
    schema = build_message_form_schema(deep_research=True, continue_button=False)

    options = schema["properties"]["deep_research"]["oneOf"]
    assert [opt["const"] for opt in options] == [True]


def test_form_schema_carries_off_value_via_const() -> None:
    schema = build_message_form_schema(deep_research=False, continue_button=False)

    options = schema["properties"]["deep_research"]["oneOf"]
    assert [opt["const"] for opt in options] == [False]


def test_toggle_only_schema_is_unchanged() -> None:
    """The toggle-only schema stays exactly what the frontend has been getting (#683)."""
    schema = build_message_form_schema(deep_research=True, continue_button=False)

    assert schema == {
        "additionalProperties": False,
        "properties": {
            "deep_research": {
                "dial:widget": "buttons",
                "oneOf": [
                    {
                        "const": True,
                        "dial:widgetOptions": {
                            "confirmationMessage": None,
                            "populateText": None,
                            "submit": False,
                        },
                        "title": "Deep research",
                    }
                ],
                "title": "Deep research",
                "type": "boolean",
            }
        },
        "required": ["deep_research"],
        "title": "MessageForm",
        "type": "object",
    }


def _continue_option(schema: dict) -> dict:
    prop = schema["properties"]["button"]
    assert prop["type"] == "string"
    assert prop["dial:widget"] == "buttons"
    assert "description" not in prop
    [option] = prop["oneOf"]
    return option


def test_button_only_schema() -> None:
    """DIAL Chat renders buttons only from a `button` or `starter` property (#831)."""
    schema = build_message_form_schema(deep_research=None, continue_button=True)

    assert list(schema["properties"]) == ["button"]
    assert _continue_option(schema) == {
        "const": "continue",
        "title": "Continue",
        "dial:widgetOptions": {
            "confirmationMessage": None,
            "populateText": "Continue",
            "submit": True,
        },
    }


def test_toggle_and_button_share_one_schema() -> None:
    schema = build_message_form_schema(deep_research=False, continue_button=True)

    assert list(schema["properties"]) == ["deep_research", "button"]
    assert schema["properties"]["deep_research"]["oneOf"][0]["const"] is False
    assert _continue_option(schema)["const"] == "continue"


def test_no_schema_without_toggle_or_button() -> None:
    assert build_message_form_schema(deep_research=None, continue_button=False) is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ toggle resolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_report_delivered_this_turn_disarms_even_if_selection_was_on() -> None:
    """The delivery turn disarms the toggle; without this the next message would silently launch a
    fresh, expensive run. The report-delivered flag wins over the (still true) user selection."""
    state = {StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED: True}
    configuration = StatGPTConfiguration(deep_research=True)

    assert ChannelCompletion._resolve_deep_research_toggle(state, configuration) is False


def test_query_answered_directly_disarms_even_if_selection_was_on() -> None:
    """When the query check answers the query directly instead of starting a run, the toggle is
    disarmed as after a delivered report, so the follow-up does not go to Deep Research again."""
    state = {StateVarsConfig.DEEP_RESEARCH_AUTO_DEACTIVATED: True}
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

    await ChannelCompletion._emit_form_schema(
        _service(available=True),
        auth_context=MagicMock(),
        state=state,
        configuration=StatGPTConfiguration(deep_research=True),
        choice=choice,
        continue_button=False,
    )

    choice.set_form_schema.assert_called_once()
    emitted = choice.set_form_schema.call_args.args[0]
    assert emitted["properties"]["deep_research"]["oneOf"][0]["const"] is True


async def test_skips_emission_when_deep_research_unavailable() -> None:
    """No toggle exists in the UI when Deep Research is unavailable, so without the "Continue"
    button nothing is emitted."""
    choice = MagicMock()

    await ChannelCompletion._emit_form_schema(
        _service(available=False),
        auth_context=MagicMock(),
        state={},
        configuration=StatGPTConfiguration(deep_research=False),
        choice=choice,
        continue_button=False,
    )

    choice.set_form_schema.assert_not_called()
