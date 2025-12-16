import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from statgpt.admin.settings.background_tasks import BackgroundTasksSettings

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

_log = logging.getLogger(__name__)

_SETTINGS = BackgroundTasksSettings()
_MAX_BACKGROUND_TASKS_SEMAPHORE = asyncio.Semaphore(_SETTINGS.max_concurrent)
_task_counter = 0


def background_task(
    func: Callable[Param, Awaitable[RetType]]
) -> Callable[Param, Awaitable[RetType]]:
    """Limit the amount of background tasks to `MAX_BACKGROUND_TASKS`."""

    @functools.wraps(func)
    async def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        global _task_counter
        _task_counter += 1
        task_id = f"{func.__name__}_{_task_counter}"

        # Log semaphore state before acquisition
        available_before = _MAX_BACKGROUND_TASKS_SEMAPHORE._value
        _log.debug(
            f"[{task_id}] Attempting to acquire semaphore. "
            f"Available slots: {available_before}/{_SETTINGS.max_concurrent}"
        )

        try:
            async with _MAX_BACKGROUND_TASKS_SEMAPHORE:
                available_after_acquire = _MAX_BACKGROUND_TASKS_SEMAPHORE._value
                _log.info(
                    f"[{task_id}] Acquired semaphore. "
                    f"Available slots: {available_after_acquire}/{_SETTINGS.max_concurrent}"
                )

                try:
                    result = await func(*args, **kwargs)
                    _log.info(f"[{task_id}] Completed successfully")
                    return result
                except asyncio.CancelledError:
                    _log.warning(f"[{task_id}] Was cancelled")
                    raise
                except Exception as e:
                    _log.error(f"[{task_id}] Failed with exception: {e}")
                    raise
        finally:
            available_final = _MAX_BACKGROUND_TASKS_SEMAPHORE._value
            _log.info(
                f"[{task_id}] Released semaphore. "
                f"Available slots: {available_final}/{_SETTINGS.max_concurrent}"
            )

    return wrapper
