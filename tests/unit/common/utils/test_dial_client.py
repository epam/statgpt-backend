import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from aidial_client import AsyncDial

from statgpt.common.utils.dial import client as client_module


@pytest.mark.parametrize("repeat_cancellation", [False, True])
async def test_cancelled_stream_opening_finishes_client_cleanup(
    monkeypatch: pytest.MonkeyPatch, repeat_cancellation: bool
) -> None:
    opening = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def stream_download(url: str) -> AsyncIterator[None]:
        assert url == "files/bucket/snapshot.csv"
        opening.set()
        await asyncio.Event().wait()
        yield None

    async def close_dial(*_: object) -> None:
        closing.set()
        await release_close.wait()
        closed.set()

    dial = MagicMock(spec=AsyncDial)
    dial.__aenter__.return_value = dial
    dial.__aexit__.side_effect = close_dial
    dial.files = MagicMock()
    dial.files.stream_download.side_effect = stream_download
    monkeypatch.setattr(client_module, "AsyncDial", MagicMock(return_value=dial))

    task = asyncio.create_task(
        client_module.open_file_stream("http://storage", "test-key", "files/bucket/snapshot.csv")
    )
    try:
        await asyncio.wait_for(opening.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=2)
        if repeat_cancellation:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert closed.is_set()
        dial.__aexit__.assert_awaited_once()
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
