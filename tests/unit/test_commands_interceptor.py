import pytest

from statgpt.utils.message_history import InterceptableCommand


@pytest.fixture
def commands():
    cmd1 = InterceptableCommand(command='one', state_var='state_var_1')
    cmd2 = InterceptableCommand(command='two', state_var='state_var_2')
    return [cmd1, cmd2]


@pytest.mark.parametrize(
    "query, expected_query, expected_state",
    [
        ('', '', {}),
        ('normal query without commands', 'normal query without commands', {}),
        ('!one query', 'query', {'state_var_1': True}),
        ('!two query', 'query', {'state_var_2': True}),
        ('!one !two query', 'query', {'state_var_1': True, 'state_var_2': True}),
        ('!two !one query', 'query', {'state_var_1': True, 'state_var_2': True}),
        ('!three query', '!three query', {}),
        ('!three !two query', '!three query', {'state_var_2': True}),
        ('!three !two !one query', '!three query', {'state_var_1': True, 'state_var_2': True}),
        ('!two !three !one query', '!three query', {'state_var_1': True, 'state_var_2': True}),
    ],
)
def test_process_query(commands, query, expected_query, expected_state):
    state = {}
    for cmd in commands:
        query = cmd.process_query(query, state=state)
    assert query == expected_query
    assert state == expected_state
