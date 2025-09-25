import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from admin_portal.settings.background_tasks import BackgroundTasksSettings

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

_SETTINGS = BackgroundTasksSettings()
_MAX_BACKGROUND_TASKS_SEMAPHORE = asyncio.Semaphore(_SETTINGS.max_concurrent)


def background_task(
    func: Callable[Param, Awaitable[RetType]]
) -> Callable[Param, Awaitable[RetType]]:
    """Limit the amount of background tasks to `MAX_BACKGROUND_TASKS`."""

    @functools.wraps(func)
    async def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        async with _MAX_BACKGROUND_TASKS_SEMAPHORE:
            return await func(*args, **kwargs)

    return wrapper
