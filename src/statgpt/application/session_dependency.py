from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import get_readonly_session


async def store_session_in_request(
    request: Request, session: AsyncSession = Depends(get_readonly_session)
) -> AsyncSession:
    """
    Dependency that gets a database session and stores it in the request state
    so it can be accessed from AIDIAL SDK handlers.
    """
    request.state.db_session = session
    return session


def get_session_from_request(request: Request) -> AsyncSession:
    """
    Retrieve the database session from request state.
    Should be called from within AIDIAL SDK handlers.
    """
    if not hasattr(request.state, 'db_session'):
        raise RuntimeError(
            "Database session not found in request state. "
            "Make sure 'store_session_in_request' dependency is configured."
        )
    return request.state.db_session
