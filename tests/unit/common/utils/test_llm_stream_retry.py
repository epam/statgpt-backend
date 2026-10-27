"""Unit tests for the retry of streaming LLM calls."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_openai import StreamChunkTimeoutError

from statgpt.common.utils import llm_stream_retry
from statgpt.common.utils.llm_stream_retry import astream_with_retry

_REQUEST = httpx.Request("POST", "http://dial.test")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class _ScriptedRunnable:
    """Each `astream` call plays the next attempt: yields its chunks, stalls, then raises its error."""

    def __init__(
        self,
        clock: _Clock,
        *attempts: tuple[list[AIMessageChunk], Exception | None],
        stall_seconds: float = 0,
    ) -> None:
        self._clock = clock
        self._attempts = attempts
        self._stall_seconds = stall_seconds
        self.calls = 0

    async def astream(self, inputs):
        chunks, error = self._attempts[min(self.calls, len(self._attempts) - 1)]
        self.calls += 1
        for chunk in chunks:
            yield chunk
        if error is not None:
            self._clock.now += self._stall_seconds
            raise error


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(llm_stream_retry, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> list[float]:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.now += delay

    monkeypatch.setattr(llm_stream_retry.asyncio, "sleep", fake_sleep)
    return sleeps


async def _stream(runnable: _ScriptedRunnable, collected: list[AIMessageChunk]) -> None:
    async for chunk in astream_with_retry(runnable, {}, name="Test"):  # type: ignore[arg-type]
        collected.append(chunk)


def _tool_call_chunk() -> AIMessageChunk:
    return AIMessageChunk(
        content="", tool_call_chunks=[{"name": "tool", "args": "{", "id": "call-1", "index": 0}]
    )


class TestAstreamWithRetry:
    async def test_retries_a_stall_before_any_content(self, clock, sleeps):
        answer = [AIMessageChunk(content="Hello"), AIMessageChunk(content=" world")]
        runnable = _ScriptedRunnable(clock, ([], httpx.ReadTimeout("")), (answer, None))
        collected: list[AIMessageChunk] = []

        await _stream(runnable, collected)

        assert collected == answer
        assert runnable.calls == 2
        assert sleeps == [2.0]

    async def test_drops_the_chunks_held_back_from_a_failed_attempt(self, clock, sleeps):
        answer = [_tool_call_chunk()]
        runnable = _ScriptedRunnable(
            clock,
            ([AIMessageChunk(content=""), _tool_call_chunk()], httpx.ReadError("")),
            (answer, None),
        )
        collected: list[AIMessageChunk] = []

        await _stream(runnable, collected)

        assert collected == answer

    async def test_yields_the_held_back_chunks_of_a_stream_without_content(self, clock, sleeps):
        answer = [AIMessageChunk(content=""), _tool_call_chunk()]
        runnable = _ScriptedRunnable(clock, (answer, None))
        collected: list[AIMessageChunk] = []

        await _stream(runnable, collected)

        assert collected == answer
        assert sleeps == []

    async def test_does_not_retry_after_content(self, clock, sleeps):
        runnable = _ScriptedRunnable(
            clock,
            ([AIMessageChunk(content=""), AIMessageChunk(content="Hel")], httpx.ReadError("")),
        )
        collected: list[AIMessageChunk] = []

        with pytest.raises(httpx.ReadError):
            await _stream(runnable, collected)

        assert collected == [AIMessageChunk(content=""), AIMessageChunk(content="Hel")]
        assert runnable.calls == 1

    @pytest.mark.parametrize(
        "error",
        [
            StreamChunkTimeoutError(60),
            httpx.ReadTimeout(""),
            httpx.ReadError(""),
            httpx.RemoteProtocolError(""),
            openai.APIError("boom", _REQUEST, body={"type": "server_error", "code": None}),
            openai.APIError("slow down", _REQUEST, body={"code": "rate_limit_exceeded"}),
        ],
        ids=lambda error: type(error).__name__,
    )
    async def test_retries_transient_errors(self, clock, sleeps, error):
        answer = [AIMessageChunk(content="Hello")]
        runnable = _ScriptedRunnable(clock, ([], error), (answer, None))
        collected: list[AIMessageChunk] = []

        await _stream(runnable, collected)

        assert collected == answer
        assert runnable.calls == 2

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("bug"),
            openai.APIError("filtered", _REQUEST, body={"code": "content_filter"}),
            # Raised before the stream starts, and already retried by the openai client.
            openai.RateLimitError(
                "slow down",
                response=httpx.Response(429, request=_REQUEST),
                body={"code": "rate_limit_exceeded"},
            ),
        ],
        ids=["ValueError", "APIError-content_filter", "RateLimitError"],
    )
    async def test_does_not_retry_other_errors(self, clock, sleeps, error):
        runnable = _ScriptedRunnable(clock, ([], error))

        with pytest.raises(type(error)):
            await _stream(runnable, [])

        assert runnable.calls == 1
        assert sleeps == []

    async def test_backs_off_exponentially_until_the_budget_is_spent(self, clock, sleeps):
        runnable = _ScriptedRunnable(clock, ([], httpx.ReadError("")))

        with pytest.raises(httpx.ReadError):
            await _stream(runnable, [])

        assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
        assert runnable.calls == 6

    @pytest.mark.parametrize(
        "error, stall_seconds, expected_sleeps",
        [
            # The httpx read timeout: no bytes for 60s.
            (httpx.ReadTimeout(""), 60, [2.0, 4.0]),
            # langchain-openai's chunk timeout: keepalive bytes, but no chunk for 120s.
            (StreamChunkTimeoutError(120), 120, [2.0]),
        ],
        ids=["60s-stall", "120s-stall"],
    )
    async def test_the_budget_starts_at_the_first_failure(
        self, clock, sleeps, error, stall_seconds, expected_sleeps
    ):
        runnable = _ScriptedRunnable(clock, ([], error), stall_seconds=stall_seconds)

        with pytest.raises(type(error)):
            await _stream(runnable, [])

        assert sleeps == expected_sleeps
        assert runnable.calls == len(expected_sleeps) + 1
