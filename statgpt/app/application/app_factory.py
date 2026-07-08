import logging
from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk import logger as dial_logger
from aidial_sdk.chat_completion import ChatCompletion, ConfigurationRequest, Request, Response
from aidial_sdk.deployment.configuration import ConfigurationResponse
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from fastapi import Request as FastAPIRequest

from statgpt.app.mcp.app import mcp
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.settings.application import application_settings
from statgpt.common.settings.dial import dial_settings

from .application import StatGPTApp
from .application import lifespan as base_lifespan
from .channel_completion import ChannelCompletion
from .channel_onboarding_completion import ChannelOnboardingCompletion
from .middleware import DebugRequestLoggingMiddleware
from .service_endpoints import router as service_router

_log = logging.getLogger(__name__)


class AppChatCompletion(ChatCompletion):

    _ONBOARDING_SUFFIX = "-onboarding"
    _ONBOARDING_SUFFIX_LEN = len(_ONBOARDING_SUFFIX)

    def _get_completion_impl(self, original_request: FastAPIRequest) -> ChatCompletion:
        deployment_id = original_request.path_params["deployment_id"]
        if deployment_id.endswith(self._ONBOARDING_SUFFIX):
            deployment_id = deployment_id[: -self._ONBOARDING_SUFFIX_LEN]
            return ChannelOnboardingCompletion(deployment_id)
        return ChannelCompletion(deployment_id)

    async def chat_completion(self, request: Request, response: Response) -> None:
        impl = self._get_completion_impl(request.original_request)
        await impl.chat_completion(request, response)

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:
        impl = self._get_completion_impl(request.original_request)
        return await impl.configuration(request)


class DialAppFactory:
    def create_app(self) -> DIALApp:
        _log.info("Creating DIAL app name=%s", dial_app_settings.dial_app_name)

        mcp_app = mcp.http_app(
            path="/", transport="streamable-http", stateless_http=True, allowed_hosts=["*"]
        )

        @asynccontextmanager
        async def lifespan(app_: StatGPTApp):  # noqa: E306
            async with base_lifespan(app_):
                async with mcp_app.lifespan(app_):
                    yield

        app = StatGPTApp(
            dial_url=dial_settings.url,
            add_healthcheck=True,
            lifespan=lifespan,
            telemetry_config=TelemetryConfig(
                service_name=dial_app_settings.dial_app_name,
                tracing=TracingConfig(),
                metrics=MetricsConfig(),
            ),
        )

        if dial_logger.isEnabledFor(logging.DEBUG):
            app.add_middleware(
                DebugRequestLoggingMiddleware,
                patterns=[r"/chat/completions$"],
            )

        # dependencies = [Depends(cancel_on_disconnect)]
        dependencies: list = []

        app.add_chat_completion_with_dependencies(
            "{deployment_id}",
            AppChatCompletion(),
            heartbeat_interval=5,
            chat_completion_dependencies=dependencies,
            configuration_dependencies=dependencies,
        )
        app.include_router(service_router)

        mcp_path = "/api/v1/{deployment_id}/mcp"
        _log.info("Mounting MCP app at %s", mcp_path)
        app.mount(mcp_path, mcp_app)

        # Add memory debug endpoints (only in development)
        if application_settings.memory_debug:
            from statgpt.common.routers.memory_debug import router as memory_debug_router

            app.include_router(memory_debug_router)

        return app
