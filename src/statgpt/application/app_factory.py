from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from fastapi import Depends

from common.config.logging import multiline_logger as logger
from statgpt.settings.application import application_settings
from statgpt.settings.dial_app import dial_app_settings

from .application import StatGPTApp
from .channel_completion import ChannelCompletion
from .service_endpoints import router as service_router
from .session_dependency import store_session_in_request


class DialAppFactory:
    def create_app(self) -> DIALApp:
        logger.info("Creating DIAL app")
        app = StatGPTApp(
            telemetry_config=TelemetryConfig(
                service_name=dial_app_settings.dial_app_name,
                tracing=TracingConfig(),
                metrics=MetricsConfig(),
            ),
        )

        app.add_chat_completion_with_dependencies(
            "{deployment_id}",
            ChannelCompletion(),
            heartbeat_interval=10,
            chat_completion_dependencies=[Depends(store_session_in_request)],
            configuration_dependencies=[Depends(store_session_in_request)],
        )
        app.include_router(service_router)

        # Add memory debug endpoints (only in development)
        if application_settings.memory_debug:
            from common.routers.memory_debug import router as memory_debug_router

            app.include_router(memory_debug_router)

        return app
