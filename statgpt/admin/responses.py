import logging
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Any

from fastapi.responses import StreamingResponse

from statgpt.common.utils.async_utils import await_cancellation_safe

_log = logging.getLogger(__name__)


class ClosingStreamingResponse(StreamingResponse):
    """Close an opened stream even when sending fails before body iteration begins."""

    def __init__(
        self,
        content: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]],
        media_type: str,
        headers: dict[str, str],
    ) -> None:
        super().__init__(content=content, media_type=media_type, headers=headers)
        self._close_callback = close

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await await_cancellation_safe(self._close_stream())

    async def _close_stream(self) -> None:
        try:
            await self._close_callback()
        except Exception:
            _log.exception("Failed to close prepared-download storage resources.")
