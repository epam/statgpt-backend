import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, Self, TypeVar

ValueType = TypeVar("ValueType")


class ValueUpdater(Generic[ValueType]):
    def __init__(self, async_get_value: Callable[[], Awaitable[ValueType]], timeout: int = 3600):
        """
        ValueManager is a class that updates a value at a specified interval.

        Args:
            async_get_value: An awaitable that returns the value to be managed.
            timeout: The time in seconds after which the value will be refreshed.
        """
        self._async_get_value = async_get_value
        self._timeout = timeout

        self._value: ValueType | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    @property
    def value(self) -> ValueType:
        if self._value is None:
            raise RuntimeError("Value has not been initialized yet.")
        return self._value

    @property
    def is_initialized(self) -> bool:
        """Check if the refresh loop is running."""
        return self._refresh_task is not None

    async def initialize(self) -> None:
        """Initialize the value and start the refresh loop."""
        if self._value is None:
            await self.refresh_value()
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        """Stop the refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._timeout)
            await self.refresh_value()

    async def refresh_value(self) -> None:
        async with self._lock:
            self._value = await self._async_get_value()
