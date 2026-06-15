import pytest

from statgpt.app.default_prompts import hybrid_search_default_prompts

EXPECTED_VARS = {"entities", "period", "forbidden", "input"}


def test_normalization_prompt_input_variables():
    """The YAML placeholders must stay in sync with the keys passed by _normalize_input."""
    template = hybrid_search_default_prompts.normalization_prompt.get_template()
    assert set(template.input_variables) == EXPECTED_VARS


def test_normalization_prompt_split_system_holds_instructions_user_holds_only_input():
    template = hybrid_search_default_prompts.normalization_prompt.get_template()
    messages = template.format_messages(
        entities="Named Entities:\n - Germany (Country/Reference area) (REMOVE)\n",
        period="Time Period:\nfrom 2010-01-01 to 2020-12-31",
        forbidden="Forbidden to remove words:\nbalance of payments\n",
        input="gdp for germany",
    )
    assert len(messages) == 2
    system_msg, human_msg = messages[0].content, messages[1].content
    assert isinstance(system_msg, str) and isinstance(human_msg, str)

    # Instructions + JSON output contract live in the system message.
    assert "Instructions" in system_msg
    assert "cleaned_input" in system_msg
    assert "JSON" in system_msg
    # The runtime query context is rendered into the system message, not the user message.
    assert "Germany" in system_msg
    assert "balance of payments" in system_msg
    # The user message carries ONLY the raw query.
    assert human_msg.strip() == "gdp for germany"
    assert "Instructions" not in human_msg


@pytest.mark.parametrize(
    "entities, period, forbidden",
    [
        (
            "Named Entities:\n - Germany (Country/Reference area) (REMOVE)\n",
            "Time Period:\nfrom 2010-01-01 to 2020-12-31",
            "Forbidden to remove words:\nGDP\n",
        ),
        ("Named Entities:\n - Germany (Country/Reference area) (REMOVE)\n", "", ""),
        ("", "Time Period:\nfrom 2010-01-01 to 2020-12-31", ""),
        ("", "", ""),
    ],
)
def test_normalization_prompt_renders_all_conditional_cases(entities, period, forbidden):
    """Every combination of present/absent context sections renders without error."""
    template = hybrid_search_default_prompts.normalization_prompt.get_template()
    messages = template.format_messages(
        entities=entities, period=period, forbidden=forbidden, input="some query"
    )
    assert len(messages) == 2
    human_msg = messages[1].content
    assert isinstance(human_msg, str)
    assert human_msg.strip() == "some query"
