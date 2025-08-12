from sqlalchemy import text

from common.config import Versions
from common.config import multiline_logger as logger

from .database import get_session_contex_manager


class DatabaseConnectionError(RuntimeError):
    pass


class AlembicTableNotFoundError(RuntimeError):
    pass


class WrongAlembicVersionError(RuntimeError):
    pass


class DatabaseHealthChecker:
    @classmethod
    async def check_connection(cls):
        try:
            async with get_session_contex_manager() as session:
                await session.execute(text("SELECT 1;"))
        except Exception as e:
            msg = "Connection to database failed. Check if the database is running and the connection string is correct."
            logger.error(msg)
            logger.error(e)
            raise DatabaseConnectionError(msg)

    @classmethod
    async def check_alembic_version(cls):
        async with get_session_contex_manager() as session:
            res = await session.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'alembic_version');"
                )
            )
            table_exist: bool = res.scalar()
            if not table_exist:
                raise AlembicTableNotFoundError("Alembic table doesn't exist")

            res = await session.execute(text("SELECT version_num FROM alembic_version;"))
            alembic_version: str = res.scalar()

            if alembic_version != Versions.ALEMBIC_TARGET_VERSION:
                logger.error(
                    f"{alembic_version=}, TARGET_VERSION={Versions.ALEMBIC_TARGET_VERSION!r}"
                )
                raise WrongAlembicVersionError("Alembic version is not correct")

    @classmethod
    async def check(cls):
        await cls.check_connection()
        await cls.check_alembic_version()
