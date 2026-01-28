import logging

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.models.database import get_session_contex_manager
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


async def preload_data(allow_cached_datasets: bool, use_resolved_config: bool) -> None:
    """Preload all datasets into cache.

    Args:
        allow_cached_datasets: Whether to put dataset classes into cache.
        use_resolved_config: If True, use resolved_config from completed versions
            (with concrete URN values). If False, use original dataset config
            (which may contain dynamic values like 'latest').
    """
    _log.info('~~~ Data preload ~~~')

    _log.info("Loading dataset cache...")
    async with get_session_contex_manager() as session:
        try:
            count = await DataSetService(session).preload_datasets(
                auth_context=_DataPreloaderAuthContext(),
                allow_cached_datasets=allow_cached_datasets,
                use_resolved_config=use_resolved_config,
            )
            _log.info(f'{count} datasets preloaded')
        except Exception:
            _log.exception("Error happened while loading dataset cache")

    _log.info('~~~ Data preload finished ~~~')
