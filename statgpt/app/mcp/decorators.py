import functools
import inspect
import logging
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from statgpt.app.mcp.exceptions import MissingDeploymentIdError
from statgpt.app.security.exceptions import AuthenticationError, AuthorizationError

P = ParamSpec("P")
T = TypeVar("T")

AsyncHandler = Callable[P, Coroutine[Any, Any, T]]


def guard_channel_resolution(
    *, default: Any, log_prefix: str, detail_arg: str | None = None
) -> Callable[[AsyncHandler[P, T]], AsyncHandler[P, T]]:
    """Degrade gracefully when an MCP handler cannot resolve its channel context.

    Logs the failure and returns ``default`` instead of propagating. ``log_prefix``
    is the action label (e.g. ``"tools/list"``); when ``detail_arg`` names a parameter
    of the wrapped handler, its runtime value is appended as ``"<log_prefix> (<value>)"``
    (e.g. ``"tools/call (my_tool)"``).
    """

    def decorator(func: AsyncHandler[P, T]) -> AsyncHandler[P, T]:
        logger = logging.getLogger(func.__module__)
        signature = inspect.signature(func)

        def _label(args: tuple, kwargs: dict) -> str:
            if detail_arg is None:
                return log_prefix
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return f"{log_prefix} ({bound.arguments.get(detail_arg)})"

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except (AuthenticationError, AuthorizationError) as e:
                logger.warning(
                    "Auth error resolving channel context for %s: %s", _label(args, kwargs), e
                )
            except MissingDeploymentIdError as e:
                logger.warning(
                    "Configuration error resolving channel context for %s: %s",
                    _label(args, kwargs),
                    e,
                )
            except Exception:
                logger.exception("Could not resolve channel context for %s", _label(args, kwargs))
            return default

        return wrapper

    return decorator
