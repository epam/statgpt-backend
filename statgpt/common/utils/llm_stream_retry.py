import asyncio
import itertools
import time
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import httpx
import openai
from langchain_core.messages import BaseMessageChunk
from langchain_core.runnables import Runnable

from statgpt.common.config import multiline_logger as logger

_ChunkT = TypeVar('_ChunkT', bound=BaseMessageChunk)

# A stream that stalls or drops after the response headers arrived fails with one of these. The
# openai client's `max_retries` covers only failures before the response starts, so a stream needs
# its own retry. `TimeoutError` covers langchain-openai's `StreamChunkTimeoutError`.
_TRANSIENT_STREAM_ERRORS: tuple[type[Exception], ...] = (
    TimeoutError,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)

# Codes of an error event sent inside the stream that are worth retrying.
_TRANSIENT_STREAM_ERROR_CODES = frozenset({'server_error', 'rate_limit_exceeded'})

# Exponential backoff (2s, 4s, 8s, ...). No retry starts once its backoff would end past the budget,
# counted from the first failure, so a call that stalls for a long time before failing still gets
# retried. A mid-stream 429 carries no `Retry-After`, so the budget alone has to be long enough to
# outlast a rate-limit episode.
_RETRY_BUDGET_SECONDS = 120.0
_INITIAL_DELAY_SECONDS = 2.0


async def astream_with_retry(
    runnable: Runnable[Any, _ChunkT], inputs: Any, *, name: str
) -> AsyncIterator[_ChunkT]:
    """Stream `runnable`, retrying a transient failure that happens before any text content.

    Chunks without text content (e.g. tool call deltas) are held back until the first chunk with
    text content, so a failed attempt leaves nothing behind for the caller. Once text content has
    been yielded - and may have been shown to the user - a failure is raised as is.
    """
    first_failure: float | None = None
    for attempt in itertools.count(1):
        held_back: list[_ChunkT] = []
        content_started = False
        try:
            async for chunk in runnable.astream(inputs):
                if content_started:
                    yield chunk
                elif isinstance(chunk.content, str) and chunk.content:
                    content_started = True
                    for held_chunk in held_back:
                        yield held_chunk
                    yield chunk
                else:
                    held_back.append(chunk)
        except Exception as e:
            if content_started or not _is_transient(e):
                raise
            now = time.monotonic()
            if first_failure is None:
                first_failure = now
            delay = _INITIAL_DELAY_SECONDS * 2 ** (attempt - 1)
            if now - first_failure + delay > _RETRY_BUDGET_SECONDS:
                logger.error(
                    f"{name} LLM stream failed before any content, giving up after"
                    f" {attempt} attempt(s): {e!r}"
                )
                raise
            logger.warning(
                f"{name} LLM stream failed before any content (attempt {attempt}),"
                f" retrying in {delay:.1f}s: {e!r}"
            )
            await asyncio.sleep(delay)
        else:
            # The stream ended without text content, e.g. a response with tool calls only.
            for held_chunk in held_back:
                yield held_chunk
            return


def _is_transient(error: Exception) -> bool:
    if isinstance(error, _TRANSIENT_STREAM_ERRORS):
        return True
    # The openai client raises a plain `APIError` for an error event inside the stream. Its
    # subclasses are raised before the stream starts and are already retried by the client.
    return type(error) is openai.APIError and bool(
        {error.code, error.type} & _TRANSIENT_STREAM_ERROR_CODES
    )
