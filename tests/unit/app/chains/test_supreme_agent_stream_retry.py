from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from statgpt.app.chains.supreme_agent import SupremeAgent
from statgpt.common.utils import llm_stream_retry


class _StallingThenAnsweringChain:
    """The first stream stalls before any chunk; the next one answers."""

    def __init__(self, answer: list[AIMessageChunk]) -> None:
        self._answer = answer
        self.calls = 0

    async def astream(self, inputs):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ReadTimeout("")
        for chunk in self._answer:
            yield chunk


async def test_run_retries_a_stream_that_stalls_before_any_content(
    monkeypatch: pytest.MonkeyPatch,
):
    async def no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(llm_stream_retry.asyncio, "sleep", no_sleep)
    chain = _StallingThenAnsweringChain(
        [AIMessageChunk(content=""), AIMessageChunk(content="Hello"), AIMessageChunk(content="!")]
    )
    choice = MagicMock()
    agent = SupremeAgent(choice, chain)  # type: ignore[arg-type]

    response = await agent.run(MagicMock(), MagicMock())

    assert chain.calls == 2
    assert response.resp.content == "Hello!"
    assert response.finished
    assert [call.args[0] for call in choice.append_content.call_args_list] == ["Hello", "!"]
