import asyncio
from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig

from common.config import DialConfig
from common.config.logging import multiline_logger as logger
from common.indexer.cache_factory import CacheFactory
from common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from common.services.data_preloader import preload_data
from statgpt.config import DialAppConfig
from statgpt.utils import HierarchiesLoader

from .channel_completion import ChannelCompletion
from .service_endpoints import router as service_router


@asynccontextmanager
async def lifespan(app: "StatGPTApp"):
    async with optional_msi_token_manager_context():
        # Check resources' availability:
        await DatabaseHealthChecker.check()

        # Load in cache:
        await HierarchiesLoader.get_hierarchy("country_groups")

        # Start data preloading in the background
        asyncio.create_task(preload_data())

        CacheFactory.get_instance()

        yield
        # Clean up


class StatGPTApp(DIALApp):
    def __init__(self, **kwargs):
        super().__init__(
            dial_url=DialConfig.get_url(),
            add_healthcheck=True,
            lifespan=lifespan,
            **kwargs,
        )


class DialAppFactory:
    def create_app(self) -> DIALApp:
        logger.info("Creating DIAL app")
        app = StatGPTApp(
            telemetry_config=TelemetryConfig(
                service_name=DialAppConfig.APP_NAME,
                tracing=TracingConfig(),
                metrics=MetricsConfig(),
            ),
        )

        app.add_chat_completion("{deployment_id}", ChannelCompletion())
        app.include_router(service_router)

        return app
