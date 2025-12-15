import asyncio

from sqlalchemy import text

from statgpt.common.config import Versions
from statgpt.common.config import multiline_logger as logger
from statgpt.common.settings.database import PostgresSettings

from .database import get_session_contex_manager


class AlembicTableNotFoundError(RuntimeError):
    pass


class WrongAlembicVersionError(RuntimeError):
    pass


class DatabaseHealthChecker:

    def __init__(self):
        self._postgres_settings = PostgresSettings()

    @classmethod
    async def check_alembic_version(cls) -> None:
        async with get_session_contex_manager() as session:
            res = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'alembic_version');"
                )
            )
            table_exist: bool = res.scalar()  # type: ignore[assignment]
            if not table_exist:
                raise AlembicTableNotFoundError("Alembic table doesn't exist")

            res = await session.execute(text("SELECT version_num FROM alembic_version;"))
            alembic_version: str = res.scalar()  # type: ignore[assignment]

            if alembic_version != Versions.ALEMBIC_TARGET_VERSION:
                logger.error(
                    f"{alembic_version=}, TARGET_VERSION={Versions.ALEMBIC_TARGET_VERSION!r}"
                )
                raise WrongAlembicVersionError("Alembic version is not correct")

    async def check(self) -> None:
        for attempt in range(self._postgres_settings.alembic_max_retries):
            try:
                await self.check_alembic_version()
            except (AlembicTableNotFoundError, WrongAlembicVersionError) as e:
                logger.error(
                    f"Failed to check alembic version on attempt {attempt + 1}"
                    f"/{self._postgres_settings.alembic_max_retries}: {str(e)}"
                )

                if attempt < self._postgres_settings.alembic_max_retries - 1:
                    sleep_time = self._postgres_settings.alembic_retry_interval * (2**attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise
