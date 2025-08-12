import pytest  # noqa: F401
import pytest_asyncio
from sqlalchemy import text

from common import models
from common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from common.data.quanthub.v21.sdmx_extensions import __apply_sdmx_extensions

from .mocks import AsyncSdmxClientMock


async def _truncate_table(session, table_name):
    await session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
    await session.commit()
    return


@pytest_asyncio.fixture
async def clear_channels(session):
    """Clear the channels table before test."""

    await _truncate_table(session, models.Channel.__tablename__)
    return


@pytest_asyncio.fixture
async def clear_data_sources(session):
    """Use this fixture to clear the data_sources table before running a test."""

    await _truncate_table(session, models.DataSource.__tablename__)
    return


@pytest_asyncio.fixture
async def clear_datasets(session):
    """Clear the datasets table before test."""

    await _truncate_table(session, models.DataSet.__tablename__)
    return


@pytest_asyncio.fixture
async def clear_all(session):
    """Clear all tables before test."""

    await _truncate_table(session, models.ChannelDataset.__tablename__)
    await _truncate_table(session, models.DataSet.__tablename__)
    await _truncate_table(session, models.DataSource.__tablename__)
    await _truncate_table(session, models.Channel.__tablename__)
    return


@pytest_asyncio.fixture
def sdmx_clint_mock(monkeypatch):
    def create_async_mock(*args, **kwargs):
        __apply_sdmx_extensions()
        return AsyncSdmxClientMock()

    monkeypatch.setattr(AsyncQuanthubClient, "from_config", create_async_mock)
