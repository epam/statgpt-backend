import asyncio

import pytest

from statgpt.admin.services import background_tasks
from statgpt.admin.services.background_tasks import background_task


class TestBackgroundTaskTimeoutIsolation:
    """Tests for the ``background_task`` decorator.

    Regression guard: a per-task timeout must be contained at the decorator
    boundary and must not propagate out, otherwise a single timed-out job aborts
    the surrounding ``asyncio.gather`` (auto-update batch) or Starlette's
    sequential ``BackgroundTasks`` runner, taking down sibling tasks.
    """

    @pytest.mark.asyncio
    async def test_timeout_is_contained_and_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 0.01)

        @background_task
        async def slow() -> str:
            await asyncio.sleep(1)
            return "should-not-reach"

        # The timeout must be swallowed (logged), not propagated.
        result = await slow()
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_does_not_abort_sibling_tasks(self, monkeypatch) -> None:
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 0.01)

        @background_task
        async def slow() -> str:
            await asyncio.sleep(1)
            return "slow"

        @background_task
        async def fast() -> str:
            return "fast"

        # gather uses the default return_exceptions=False; if the decorator let the
        # TimeoutError escape, this gather would raise and lose the fast result.
        slow_result, fast_result = await asyncio.gather(slow(), fast())

        assert slow_result is None
        assert fast_result == "fast"

    @pytest.mark.asyncio
    async def test_sequential_runner_continues_after_timeout(self, monkeypatch) -> None:
        # Mimics Starlette's ``BackgroundTasks.__call__`` (``for task in tasks: await task()``):
        # a timed-out task must not break the loop, so the next task still starts.
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 0.01)
        ran: list[str] = []

        @background_task
        async def slow() -> None:
            await asyncio.sleep(1)
            ran.append("slow")

        @background_task
        async def fast() -> None:
            ran.append("fast")

        for coro in (slow(), fast()):
            await coro

        assert ran == ["fast"]  # slow timed out before appending; fast still ran

    @pytest.mark.asyncio
    async def test_timeout_releases_slot_for_tasks_over_concurrency_limit(
        self, monkeypatch
    ) -> None:
        # With more tasks than the concurrency limit, the timed-out task must release
        # its semaphore slot so the queued tasks can acquire it and run.
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 0.05)
        monkeypatch.setattr(
            background_tasks, "_MAX_BACKGROUND_TASKS_SEMAPHORE", asyncio.Semaphore(1)
        )
        completed: list[str] = []

        @background_task
        async def slow() -> None:
            await asyncio.sleep(1)
            completed.append("slow")

        @background_task
        async def fast(name: str) -> None:
            completed.append(name)

        # slow() acquires the only slot first and holds it until it times out;
        # the queued fast tasks must still run once the slot is released.
        await asyncio.gather(slow(), fast("a"), fast("b"))

        assert "slow" not in completed
        assert set(completed) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_genuine_cancellation_still_propagates(self, monkeypatch) -> None:
        # Disable the timeout so asyncio.timeout is a no-op and an external cancel
        # surfaces as a plain CancelledError (not converted to TimeoutError).
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", None)

        started = asyncio.Event()

        @background_task
        async def long_running() -> None:
            started.set()
            await asyncio.sleep(10)

        task = asyncio.ensure_future(long_running())
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_normal_completion_returns_value(self, monkeypatch) -> None:
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 5)

        @background_task
        async def ok() -> str:
            return "done"

        assert await ok() == "done"

    @pytest.mark.asyncio
    async def test_ordinary_exception_still_propagates(self, monkeypatch) -> None:
        # The `except Exception` branch is intentionally left unchanged: ordinary
        # errors still propagate (the *_in_background_task wrappers swallow them).
        monkeypatch.setattr(background_tasks._SETTINGS, "task_timeout", 5)

        @background_task
        async def boom() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await boom()
