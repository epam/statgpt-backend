import os
import sys
from pathlib import Path

import dotenv
import pytest_asyncio

src_path = Path(__file__).parent.parent.absolute() / 'src'
sys.path.append(str(src_path))
dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

from common.models.database import SessionMaker


@pytest_asyncio.fixture
async def async_session_factory():
    # todo: use SessionMakerSingleton and "session" fixture scope
    #   but it doesn't work: maybe because of "function" scope for base event_loop fixture
    #   maybe try to change event_loop scope to module, but keep in mind that integration tests
    #   shouldn't never be started at the same time if they are writing to the SQL table

    session_maker = await SessionMaker(engine_config={}).create()
    return session_maker


@pytest_asyncio.fixture
async def session(async_session_factory):
    async with async_session_factory() as session:
        yield session


# @pytest_asyncio.fixture
# async def mock_async_session_factory(monkeypatch, async_session_factory):
#     from common.models import database
#
#     @asynccontextmanager
#     async def get_session_contex_manager_mock():
#         async with async_session_factory() as session:
#             yield session
#
#     monkeypatch.setattr(database, "get_session_contex_manager", get_session_contex_manager_mock)
