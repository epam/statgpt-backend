import pytest  # noqa: F401
import pytest_asyncio
from sqlalchemy import text

from statgpt.admin.audit import decorators as audit_decorators
from statgpt.admin.audit.context import AuditContext
from statgpt.common import models
from statgpt.common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from statgpt.common.data.quanthub.v21.sdmx_extensions import __apply_sdmx_extensions

from .mocks import AsyncSdmxClientMock


def get_integration_test_audit_context() -> AuditContext:
    return AuditContext(
        performed_by="integration-tests",
        performed_by_name="Integration Tests",
        trace_id="00000000000000000000000000000001",
    )


@pytest.fixture(autouse=True)
def integration_test_audit_context(monkeypatch):
    monkeypatch.setattr(
        audit_decorators,
        "get_audit_context",
        lambda: get_integration_test_audit_context(),
    )


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
