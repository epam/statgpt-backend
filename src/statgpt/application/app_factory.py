import logging

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, ConfigurationRequest, Request, Response
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from fastapi import Request as FastAPIRequest

from common.settings.application import application_settings
from statgpt.settings.dial_app import dial_app_settings

from .application import StatGPTApp
from .channel_completion import ChannelCompletion
from .channel_onboarding_completion import ChannelOnboardingCompletion
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

    async def configuration(self, request: ConfigurationRequest) -> dict:
        impl = self._get_completion_impl(request.original_request)
        return await impl.configuration(request)


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

        # Add memory debug endpoints (only in development)
        if application_settings.memory_debug:
            from common.routers.memory_debug import router as memory_debug_router

            app.include_router(memory_debug_router)

        return app
