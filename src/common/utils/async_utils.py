import asyncio
import functools
import logging
from typing import Callable, TypeVar

T = TypeVar('T')


def catch_and_log_async(logger: logging.Logger | None = None):
    """
    Decorator for async methods/functions that catches exceptions,
    logs them, and returns None instead of raising.

    Args:
        logger: Optional logger instance. If not provided, uses the module's logger.

    Example:
        @catch_and_log_async()
        async def risky_operation():
            # code that might raise an exception
            pass

        @catch_and_log_async(custom_logger)
        async def another_operation():
            # code with custom logger
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T | None:
            _logger = logger or logging.getLogger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                _logger.error(
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": args,
                        "kwargs": kwargs,
                    },
                )
                return None

        return wrapper

    return decorator


def gather_with_concurrency(n: int, *coros) -> asyncio.Future:
    """Run coroutines with a limit on the number of concurrent tasks.

    Args:
        n: Maximum number of coroutines to run concurrently (must be > 0)
        *coros: Variable number of coroutine objects to execute

    Returns:
        A future aggregating results from all coroutines

    Raises:
        ValueError: If n <= 0
    """
    if n <= 0:
        raise ValueError(f"Concurrency limit must be positive, got {n}")

    if n >= len(coros):
        return asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return asyncio.gather(*(sem_coro(coro) for coro in coros))
