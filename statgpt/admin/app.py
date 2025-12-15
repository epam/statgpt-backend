import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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
from statgpt.common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from statgpt.common.services.data_preloader import preload_data


@asynccontextmanager
async def lifespan(app_: FastAPI):
    async with optional_msi_token_manager_context():
        # Check resources' availability:
        await DatabaseHealthChecker().check()

        # Start data preloading in the background
        asyncio.create_task(preload_data(allow_cached_datasets=False))

        yield
        # Clean up


app = FastAPI(
    lifespan=lifespan,
    docs_url="/admin/api/docs",
    redoc_url="/admin/api/redoc",
    openapi_url="/admin/api/openapi.json",
)

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
