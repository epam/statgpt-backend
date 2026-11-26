import asyncio
from collections.abc import AsyncIterator, MutableMapping
from typing import Any
from unittest.mock import AsyncMock

import pytest

from statgpt.admin.responses import ClosingStreamingResponse


def _response(chunks: AsyncIterator[bytes], close: AsyncMock) -> ClosingStreamingResponse:
    return ClosingStreamingResponse(chunks, close, "text/csv", {"Cache-Control": "no-store"})


def _scope(version: str) -> dict[str, Any]:
    return {"type": "http", "asgi": {"spec_version": version}}


async def _receive() -> MutableMapping[str, Any]:
    await asyncio.Event().wait()
    return {"type": "http.disconnect"}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
async def test_completion_closes_the_opened_stream(version: str) -> None:
    async def chunks():
        yield b"first"
        yield b"second"

    close = AsyncMock()
    send = AsyncMock()
    await asyncio.wait_for(_response(chunks(), close)(_scope(version), _receive, send), timeout=2)
    close.assert_awaited_once()
    assert [
        call.args[0]["body"]
        for call in send.await_args_list
        if call.args[0]["type"] == "http.response.body"
    ] == [b"first", b"second", b""]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
async def test_send_failure_before_body_iteration_still_closes(version: str) -> None:
    iterated = False

    async def chunks():
        nonlocal iterated
        iterated = True
        yield b"never sent"

    close = AsyncMock()
    send = AsyncMock(side_effect=RuntimeError("response-start failed"))
    with pytest.raises(RuntimeError, match="response-start failed"):
        await _response(chunks(), close)(_scope(version), _receive, send)
    assert not iterated
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
async def test_storage_failure_mid_stream_closes_without_json_replacement(version: str) -> None:
    async def chunks():
        yield b"partial csv"
        raise RuntimeError("storage iterator failed")

    close = AsyncMock()
    send = AsyncMock()
    with pytest.raises(RuntimeError, match="storage iterator failed"):
        await _response(chunks(), close)(_scope(version), _receive, send)
    close.assert_awaited_once()
    messages = [call.args[0] for call in send.await_args_list]
    assert [
        message["status"] for message in messages if message["type"] == "http.response.start"
    ] == [200]
    assert [message["body"] for message in messages if message["type"] == "http.response.body"] == [
        b"partial csv"
    ]


@pytest.mark.asyncio
async def test_disconnect_closes_even_while_body_iterator_is_waiting() -> None:
    sent = asyncio.Event()

    async def chunks():
        yield b"partial csv"
        await asyncio.Event().wait()

    async def receive():
        await sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            sent.set()

    close = AsyncMock()
    await asyncio.wait_for(_response(chunks(), close)(_scope("2.3"), receive, send), timeout=2)
    assert sent.is_set()
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_transport_disconnect_on_new_asgi_path_closes_stream() -> None:
    async def chunks():
        yield b"csv"

    close = AsyncMock()
    send = AsyncMock(side_effect=OSError("client disconnected"))
    with pytest.raises(Exception) as caught:
        await _response(chunks(), close)(_scope("2.4"), _receive, send)
    assert type(caught.value).__name__ == "ClientDisconnect"
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
async def test_request_cancellation_awaits_stream_close(version: str) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        started.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    async def finish_close():
        await asyncio.sleep(0)
        closed.set()

    close = AsyncMock(side_effect=finish_close)
    response_task = asyncio.create_task(
        _response(chunks(), close)(_scope(version), _receive, AsyncMock())
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    response_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(response_task, timeout=2)
    assert closed.is_set()
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.3", "2.4"])
async def test_repeated_cancellation_during_close_waits_until_resources_close(version: str) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        yield b"csv"

    async def finish_close():
        started.set()
        await release.wait()
        closed.set()

    close = AsyncMock(side_effect=finish_close)
    response_task = asyncio.create_task(
        _response(chunks(), close)(_scope(version), _receive, AsyncMock())
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    response_task.cancel()
    await asyncio.sleep(0)
    response_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(response_task, timeout=2)
    assert closed.is_set()
    close.assert_awaited_once()
