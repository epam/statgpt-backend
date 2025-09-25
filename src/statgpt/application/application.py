import asyncio
from collections.abc import Sequence
from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion
from aidial_sdk.deployment.configuration import ConfigurationRequest
from aidial_sdk.deployment.tokenize import TokenizeRequest
from aidial_sdk.deployment.truncate_prompt import TruncatePromptRequest
from aidial_sdk.utils._reflection import get_method_implementation
from fastapi import params as fastapi_params

from common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from common.services.data_preloader import preload_data
from common.settings.dial import dial_settings


@asynccontextmanager
async def lifespan(app: "StatGPTApp"):
    async with optional_msi_token_manager_context():
        # Check resources' availability:
        await DatabaseHealthChecker.check()

        # Start data preloading in the background
        asyncio.create_task(preload_data(allow_cached_datasets=True))

        yield
        # Clean up


class StatGPTApp(DIALApp):
    def __init__(self, **kwargs):
        super().__init__(
            dial_url=dial_settings.url,
            add_healthcheck=True,
            lifespan=lifespan,
            **kwargs,
        )

    def add_chat_completion_with_dependencies(
        self,
        deployment_name: str,
        impl: ChatCompletion,
        *,
        heartbeat_interval: float | None = None,
        chat_completion_dependencies: Sequence[fastapi_params.Depends] | None = None,
        rate_dependencies: Sequence[fastapi_params.Depends] | None = None,
        tokenize_dependencies: Sequence[fastapi_params.Depends] | None = None,
        truncate_prompt_dependencies: Sequence[fastapi_params.Depends] | None = None,
        configuration_dependencies: Sequence[fastapi_params.Depends] | None = None,
    ) -> "StatGPTApp":

        self.add_api_route(
            f"/openai/deployments/{deployment_name}/chat/completions",
            self._chat_completion(
                deployment_name,
                impl,
                heartbeat_interval=heartbeat_interval,
            ),
            methods=["POST"],
            dependencies=chat_completion_dependencies,
        )

        self.add_api_route(
            f"/openai/deployments/{deployment_name}/rate",
            self._rate_response(deployment_name, impl),
            methods=["POST"],
            dependencies=rate_dependencies,
        )

        if endpoint_impl := get_method_implementation(impl, "tokenize"):
            self.add_api_route(
                f"/openai/deployments/{deployment_name}/tokenize",
                self._endpoint_factory(deployment_name, endpoint_impl, "tokenize", TokenizeRequest),
                methods=["POST"],
                dependencies=tokenize_dependencies,
            )

        if endpoint_impl := get_method_implementation(impl, "truncate_prompt"):
            self.add_api_route(
                f"/openai/deployments/{deployment_name}/truncate_prompt",
                self._endpoint_factory(
                    deployment_name,
                    endpoint_impl,
                    "truncate_prompt",
                    TruncatePromptRequest,
                ),
                methods=["POST"],
                dependencies=truncate_prompt_dependencies,
            )

        if endpoint_impl := get_method_implementation(impl, "configuration"):
            self.add_api_route(
                f"/openai/deployments/{deployment_name}/configuration",
                self._endpoint_factory(
                    deployment_name,
                    endpoint_impl,
                    "configuration",
                    ConfigurationRequest,
                ),
                methods=["GET"],
                dependencies=configuration_dependencies,
            )

        return self
