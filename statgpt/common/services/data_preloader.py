import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.models.database import get_session_context_manager
from statgpt.common.services import DataSetService
from statgpt.common.settings.dial import dial_settings

_log = logging.getLogger(__name__)


class _DataPreloaderAuthContext(AuthContext):
    """This AuthContext is created only to load datasets when applications start."""

    @property
    def is_system(self) -> bool:
        return True

    @property
    def dial_access_token(self) -> None:
        return None

    @property
    def api_key(self) -> str:
        return dial_settings.api_key.get_secret_value()


async def preload_data(
    allow_cached_datasets: bool,
    use_resolved_config: bool,
    force_refresh: bool = False,
) -> None:
    """Preload all datasets into cache.

    Args:
        allow_cached_datasets: Whether to put dataset classes into cache.
        use_resolved_config: If True, use resolved_config from completed versions
            (with concrete URN values). If False, use original dataset config
            (which may contain dynamic values like 'latest').
        force_refresh: Reload datasets and replace cached entries even if they
            are still live (used by the periodic cache refresher).
    """
    _log.info('~~~ Data preload ~~~')

    _log.info("Loading dataset cache...")
    async with get_session_context_manager() as session:
        try:
            count = await DataSetService(session).preload_datasets(
                auth_context=_DataPreloaderAuthContext(),
                allow_cached_datasets=allow_cached_datasets,
                use_resolved_config=use_resolved_config,
                force_refresh=force_refresh,
            )
            _log.info(f'{count} datasets preloaded')
        except Exception:
            _log.exception("Error happened while loading dataset cache")

    _log.info('~~~ Data preload finished ~~~')


async def _run_dataset_preload(
    allow_cached_datasets: bool,
    use_resolved_config: bool,
    refresh_interval_seconds: int,
) -> None:
    """Warm up dataset caches, then (if an interval is set) keep them warm.

    Runs an initial warm-up and, when ``refresh_interval_seconds > 0``, force-refreshes
    the caches on that interval so entries are replaced before they expire. Each pass is
    guarded so a transient failure (e.g. a DB hiccup) neither kills the task before the
    refresh loop starts nor stops later refreshes.
    """
    # The first pass warms up without evicting live entries; later passes force-replace.
    force_refresh = False
    while True:
        try:
            await preload_data(
                allow_cached_datasets=allow_cached_datasets,
                use_resolved_config=use_resolved_config,
                force_refresh=force_refresh,
            )
        except Exception:
            _log.exception("%s dataset preload failed", "Periodic" if force_refresh else "Startup")
        if refresh_interval_seconds <= 0:
            return
        await asyncio.sleep(refresh_interval_seconds)
        force_refresh = True


@asynccontextmanager
async def preload_data_task_context(
    allow_cached_datasets: bool,
    use_resolved_config: bool,
    refresh_interval_seconds: int = 0,
) -> AsyncIterator[None]:
    """Run dataset preloading as a background task for the duration of the context.

    With ``refresh_interval_seconds > 0`` the task keeps the caches warm by
    force-refreshing entries on that interval after the initial warm-up; with ``0``
    (the default) it performs a single warm-up and exits.

    The held reference protects the task from GC; on exit a still-running preload is
    cancelled and awaited, so resources its calls use (e.g. shared HTTP pools) can be
    closed safely afterwards.
    """
    task = asyncio.create_task(
        _run_dataset_preload(
            allow_cached_datasets=allow_cached_datasets,
            use_resolved_config=use_resolved_config,
            refresh_interval_seconds=refresh_interval_seconds,
        )
    )
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            _log.exception("Data preload background task failed")
