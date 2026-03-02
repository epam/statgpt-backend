import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import HTTPException, Request

_log = logging.getLogger(__name__)


async def cancel_on_disconnect(request: Request) -> AsyncGenerator[None, None]:
    """A function developed to be a FastAPI dependency that cancels processing of a user request if the client disconnects.

    Use it to save resources when the client closes the connection on a READ request.
    DO NOT use for other operations (e.g. CREATE, UPDATE, DELETE) as this may cause an unexpected app state.

    Notes:
        1. This dependency properly works with other dependencies. For example, a database dependency
           will still close the session if the request is canceled.
        2. Context managers also work as expected.
        3. Coroutines scheduled by `asyncio.gather` or `asyncio.TaskGroup()` will be closed (canceled) immediately,
           without waiting for completion.
        4. WARNING: Tasks in separate threads started by `asyncio.to_thread` will not be canceled and will be finished completely.
        5. WARNING: This dependency does not work properly with `fastapi.BackgroundTasks`.
           A normal (non-canceled) request will fail after sending a response to the user,
           and background tasks will be canceled almost immediately.
    """

    handler_task = asyncio.current_task()
    if handler_task is None:
        yield
        return

    watch_task = asyncio.create_task(_watcher(request, handler_task))
    try:
        yield
    except asyncio.CancelledError:
        # Optional: map to 499 like nginx "Client Closed Request"
        _log.warning("Request was cancelled due to client disconnect")
        raise HTTPException(status_code=499, detail="Client closed request") from None
    finally:
        watch_task.cancel()


async def _watcher(request: Request, handler_task: asyncio.Task, interval: float = 2.0) -> None:
    _log.info(f"Watching {request.url}")

    while True:
        if await request.is_disconnected():
            _log.info(f"Cancelling handler task for {request.url}")
            handler_task.cancel()
            return
        else:
            _log.debug(f"Request {request.url} still connected")
        await asyncio.sleep(interval)
