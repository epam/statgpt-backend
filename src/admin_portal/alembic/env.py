import os
import sys
from pathlib import Path

import dotenv

module_path = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(module_path))
dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

import common.models as models
from common.models.database import SessionMaker, optional_msi_token_manager_context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = models.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    async with optional_msi_token_manager_context():
        engine = await SessionMaker().create_engine()
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)

        await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
