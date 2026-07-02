import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncContextManager

import dotenv
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from fastapi import FastAPI, status

module_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(module_path))

dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

from statgpt.admin.routers import router
from statgpt.admin.settings.app import APP_SETTINGS
from statgpt.common.config import multiline_logger as logger
from statgpt.common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from statgpt.common.services.data_preloader import preload_data
from statgpt.common.utils.elastic import elasticsearch_client_context
from statgpt.common.utils.http_pool import close_shared_http_clients

Lifespan = Callable[[FastAPI], AsyncContextManager[None]]


@asynccontextmanager
async def app_lifespan(app_: FastAPI):
    async with optional_msi_token_manager_context(), elasticsearch_client_context():
        # Check resources' availability:
        await DatabaseHealthChecker().check()

        # Start data preloading in the background
        asyncio.create_task(preload_data(allow_cached_datasets=False, use_resolved_config=False))

        try:
            yield
        finally:
            # Clean up
            await close_shared_http_clients()


lifespan: Lifespan = app_lifespan
mcp_app = None

if APP_SETTINGS.beta_mcp_enabled:
    logger.info("Beta MCP is enabled. Initializing MCP app...")

    try:
        from statgpt.admin.mcp.app import mcp
    except ImportError as e:
        logger.warning(f"MCP is enabled, but optional beta-mcp dependencies are not installed: {e}")
        sys.exit(1)

    mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)

    @asynccontextmanager
    async def combined_lifespan(app_: FastAPI):
        async with app_lifespan(app_):
            async with mcp_app.lifespan(app_):  # type: ignore[union-attr]
                yield

    lifespan = combined_lifespan

else:
    logger.info("Beta MCP is disabled.")

app = FastAPI(
    lifespan=lifespan,
    docs_url="/admin/api/docs",
    redoc_url="/admin/api/redoc",
    openapi_url="/admin/api/openapi.json",
)

if mcp_app:
    logger.info("Mounting MCP app at /mcp")
    app.mount("/mcp", mcp_app)

init_telemetry(
    app=app,
    config=TelemetryConfig(
        service_name=APP_SETTINGS.otel_service_name,
        tracing=TracingConfig(),
        metrics=MetricsConfig(),
    ),
)

app.include_router(router)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, log_config=None)
