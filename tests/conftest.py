import os

import pytest_asyncio

# AI DIAL SDK must run in pydantic v2 mode; set before `statgpt` imports it.
os.environ.setdefault("PYDANTIC_V2", "True")

from statgpt.common.models.database import SessionMaker


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
#     from statgpt.common.models import database
#
#     @asynccontextmanager
#     async def get_session_context_manager_mock():
#         async with async_session_factory() as session:
#             yield session
#
#     monkeypatch.setattr(database, "get_session_context_manager", get_session_context_manager_mock)
