"""Unit tests for preload_data_task_context: warm-up, periodic refresh, and shutdown."""

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest

from statgpt.common.services import data_preloader


async def _drive_preload_loop(
    fake_preload: Callable[..., Awaitable[None]],
    done: asyncio.Event,
    *,
    refresh_interval_seconds: int = 60,
) -> None:
    """Run preload_data_task_context with preload_data and asyncio.sleep patched.

    `fake_preload` replaces preload_data; the interval sleep is collapsed to a bare
    event-loop yield so the refresh loop iterates without real delay. Returns once
    `done` is set (or raises TimeoutError if the loop never gets there).
    """
    orig_sleep = asyncio.sleep

    async def instant_sleep(_seconds: float) -> None:
        await orig_sleep(0)

    with (
        patch.object(data_preloader, "preload_data", new=fake_preload),
        patch.object(data_preloader.asyncio, "sleep", new=instant_sleep),
    ):
        async with data_preloader.preload_data_task_context(
            allow_cached_datasets=True,
            use_resolved_config=True,
            refresh_interval_seconds=refresh_interval_seconds,
        ):
            await asyncio.wait_for(done.wait(), timeout=5)


class TestPreloadDataTaskContext:

    @pytest.mark.asyncio
    async def test_single_warmup_when_interval_zero(self) -> None:
        calls: list[bool] = []
        warmed = asyncio.Event()

        async def fake_preload(*, force_refresh: bool = False, **_kwargs) -> None:
            calls.append(force_refresh)
            warmed.set()

        with patch.object(data_preloader, "preload_data", new=fake_preload):
            async with data_preloader.preload_data_task_context(
                allow_cached_datasets=False, use_resolved_config=False
            ):
                await asyncio.wait_for(warmed.wait(), timeout=5)
                await asyncio.sleep(0)  # let the task return after the one-shot warm-up

        # Exactly one warm-up, without force_refresh; no periodic refreshes.
        assert calls == [False]

    @pytest.mark.asyncio
    async def test_running_preload_cancelled_on_exit(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_preload(**_kwargs) -> None:
            started.set()
            try:
                await asyncio.sleep(3600)  # never completes on its own
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with patch.object(data_preloader, "preload_data", new=fake_preload):
            async with data_preloader.preload_data_task_context(
                allow_cached_datasets=True, use_resolved_config=True
            ):
                await asyncio.wait_for(started.wait(), timeout=5)

        # Leaving the context cancels and awaits a still-running preload, so pools it
        # uses can be closed safely afterwards.
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_periodic_refresh_uses_force_refresh(self) -> None:
        calls: list[bool] = []
        done = asyncio.Event()

        async def fake_preload(*, force_refresh: bool = False, **_kwargs) -> None:
            calls.append(force_refresh)
            if len(calls) >= 3:  # warm-up + 2 refreshes
                done.set()

        await _drive_preload_loop(fake_preload, done)

        assert calls[0] is False  # initial warm-up
        assert all(force is True for force in calls[1:])  # periodic refreshes force-replace

    @pytest.mark.asyncio
    async def test_warmup_failure_does_not_stop_refresh_loop(self) -> None:
        calls: list[bool] = []
        done = asyncio.Event()

        async def fake_preload(*, force_refresh: bool = False, **_kwargs) -> None:
            calls.append(force_refresh)
            if len(calls) == 1:
                raise RuntimeError("db down")  # transient failure during warm-up
            if len(calls) >= 3:
                done.set()

        await _drive_preload_loop(fake_preload, done)

        assert calls[0] is False  # warm-up attempted, then raised
        assert calls[1] is True  # loop survived the failure and kept refreshing
