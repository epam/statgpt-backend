import logging

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig

from common.settings.application import application_settings
from statgpt.settings.dial_app import dial_app_settings

from .application import StatGPTApp
from .channel_completion import ChannelCompletion
from .service_endpoints import router as service_router

_log = logging.getLogger(__name__)


class DialAppFactory:
    def create_app(self) -> DIALApp:
        _log.info("Creating DIAL app name=%s", dial_app_settings.dial_app_name)
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
        )
        app.include_router(service_router)

        # Add memory debug endpoints (only in development)
        if application_settings.memory_debug:
            from common.routers.memory_debug import router as memory_debug_router

            app.include_router(memory_debug_router)

        return app
