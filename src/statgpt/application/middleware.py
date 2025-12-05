import re

from aidial_sdk import logger
from starlette.middleware.base import BaseHTTPMiddleware, Response
from starlette.requests import Request
from starlette.types import ASGIApp


class DebugRequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log raw request bodies for matching endpoints."""

    def __init__(self, app: ASGIApp, patterns: list[str]):
        """
        Args:
            app: The ASGI application.
            patterns: List of regex patterns to match request paths.
        """
        super().__init__(app)
        self._patterns = [re.compile(p) for p in patterns]

    async def dispatch(self, request: Request, call_next) -> Response:
        if any(p.search(request.url.path) for p in self._patterns):
            body = await request.body()
            logger.debug(f"Request [{request.url.path}]: {body.decode('utf-8')}")

        return await call_next(request)
