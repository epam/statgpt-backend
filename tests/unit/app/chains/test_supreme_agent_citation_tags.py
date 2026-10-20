from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from statgpt.app.chains.supreme_agent import SupremeAgent
from statgpt.common.utils import InvalidLLMStreamResponse

_ID = '9f81df22e8b442e8a491d163198b7d4a'


class _Chain:
    """Streams `answer`, then raises `error` if set."""

    def __init__(self, answer: list[str], error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error

    async def astream(self, inputs):
        for content in self._answer:
            yield AIMessageChunk(content=content)
        if self._error is not None:
            raise self._error


def _sent(choice: MagicMock) -> str:
    return ''.join(call.args[0] for call in choice.append_content.call_args_list)


async def test_run_repairs_a_mistyped_citation_tag_split_across_chunks():
    choice = MagicMock()
    chain = _Chain(['mid-2024.” <cit data-id="9f81df22', f'{_ID[8:]}></cit', '>\n\nSource'])
    agent = SupremeAgent(choice, chain)  # type: ignore[arg-type]

    response = await agent.run(MagicMock(), MagicMock())

    assert response.finished
    assert _sent(choice) == f'mid-2024.” <cit data-id="{_ID}"></cit>\n\nSource'


async def test_run_flushes_text_held_back_at_the_end_of_the_stream():
    choice = MagicMock()
    agent = SupremeAgent(choice, _Chain(['a <', 'ci']))  # type: ignore[arg-type]

    await agent.run(MagicMock(), MagicMock())

    assert _sent(choice) == 'a <ci'


async def test_run_deletes_the_text_it_sent_on_an_invalid_stream():
    choice = MagicMock()
    chain = _Chain(
        [f'<cit data-id="{_ID}></cit> Japan', ' <cit data-id="'],
        error=InvalidLLMStreamResponse('too many spaces'),
    )
    agent = SupremeAgent(choice, chain)  # type: ignore[arg-type]

    response = await agent.run(MagicMock(), MagicMock())

    sent = f'<cit data-id="{_ID}"></cit> Japan '
    assert not response.finished
    assert choice.append_content.call_args_list[-1].args[0] == (
        f'\n\n<!-- delete_chars({len(sent) + 2}) -->\n\n'
    )
    assert _sent(choice) == f'{sent}\n\n<!-- delete_chars({len(sent) + 2}) -->\n\n'
