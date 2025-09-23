from common.auth.auth_context import AuthContext
from common.config.logging import logger
from common.models.database import get_session_contex_manager
from common.services import DataSetService
from common.settings.dial import dial_settings


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


async def preload_data(allow_cached_datasets: bool) -> None:
    logger.info('~~~ Data preload ~~~')

    logger.info("Loading dataset cache...")
    async with get_session_contex_manager() as session:
        try:
            datasets = await DataSetService(session).get_datasets_schemas(
                limit=None,
                offset=0,
                auth_context=_DataPreloaderAuthContext(),
                allow_cached_datasets=allow_cached_datasets,
            )
            logger.info(f'{len(datasets)} datasets loaded')
        except Exception:
            logger.exception("Error happened while loading dataset cache")

    logger.info('~~~ Data preload finished ~~~')
