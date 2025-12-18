import pytest

from statgpt.app.schemas.state import State
from statgpt.app.utils.message_interceptors.commands_interceptor import InterceptableCommand


@pytest.fixture
def commands():
    cmd1 = InterceptableCommand(command='rag_prefilter_only', state_var='cmd_rag_prefilter_only')
    cmd2 = InterceptableCommand(command='out_of_scope_only', state_var='cmd_out_of_scope_only')
    return [cmd1, cmd2]


@pytest.mark.parametrize(
    "query, expected_query, expected_state",
    [
        ('', '', State()),
        ('normal query without commands', 'normal query without commands', State()),
        ('!out_of_scope_only query', 'query', State(cmd_out_of_scope_only=True)),
        (
            '!out_of_scope_only !rag_prefilter_only query',
            'query',
            State(cmd_out_of_scope_only=True, cmd_rag_prefilter_only=True),
        ),
        (
            '!rag_prefilter_only !out_of_scope_only query',
            'query',
            State(cmd_out_of_scope_only=True, cmd_rag_prefilter_only=True),
        ),
        ('!three query', '!three query', State()),
        ('!three !out_of_scope_only query', '!three query', State(cmd_out_of_scope_only=True)),
        (
            '!three !out_of_scope_only !rag_prefilter_only query',
            '!three query',
            State(cmd_out_of_scope_only=True, cmd_rag_prefilter_only=True),
        ),
        (
            '!out_of_scope_only !three !rag_prefilter_only query',
            '!three query',
            State(cmd_out_of_scope_only=True, cmd_rag_prefilter_only=True),
        ),
    ],
)
def test_process_query(commands, query, expected_query, expected_state: State):
    state = State()
    for cmd in commands:
        query = cmd.process_query(query, state=state)
    assert query == expected_query
    assert state == expected_state
