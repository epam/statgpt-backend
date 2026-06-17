from statgpt.app.default_prompts import hybrid_search_default_prompts

EXPECTED_VARS = {"statement", "items"}


def test_relevancy_prompt_input_variables():
    """The YAML placeholders must stay in sync with the keys passed by _relevance_candidates."""
    template = hybrid_search_default_prompts.relevancy_prompt.get_template()
    assert set(template.input_variables) == EXPECTED_VARS


def test_relevancy_prompt_split_system_holds_instructions_user_holds_only_data():
    template = hybrid_search_default_prompts.relevancy_prompt.get_template()
    messages = template.format_messages(
        statement="gdp growth rate",
        items="- National accounts\n    - (1) Gross Domestic Product growth rate",
    )
    assert len(messages) == 2
    system_msg, human_msg = messages[0].content, messages[1].content
    assert isinstance(system_msg, str) and isinstance(human_msg, str)

    # Instructions + JSON output contract live in the system message.
    assert "Instructions" in system_msg
    assert "relevance" in system_msg
    assert "JSON" in system_msg
    # The per-call payload is rendered into the user message, not the system message.
    assert "gdp growth rate" not in system_msg
    assert "gdp growth rate" in human_msg
    assert "(1) Gross Domestic Product growth rate" in human_msg
    # The user message carries ONLY the statement and items, no instructions.
    assert human_msg.lstrip().startswith("Statement:")
    assert "Instructions" not in human_msg
