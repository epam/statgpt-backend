import asyncio
import logging
from collections.abc import Sequence
from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion
from aidial_sdk.deployment.configuration import ConfigurationRequest
from aidial_sdk.deployment.tokenize import TokenizeRequest
from aidial_sdk.deployment.truncate_prompt import TruncatePromptRequest
from aidial_sdk.utils._reflection import get_method_implementation
from fastapi import params as fastapi_params

from statgpt.app.chains.sdmx_query_app_tool import sdmx_query_app_http_client
from statgpt.app.mcp.widget_resource import widget_http_client
from statgpt.common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from statgpt.common.services.data_preloader import preload_data
from statgpt.common.settings.data_preloader import data_preloader_settings
from statgpt.common.utils.elastic import elasticsearch_client_context

_log = logging.getLogger(__name__)


async def _preload_data_periodically() -> None:
    """Warm up dataset caches at startup, then refresh them before entries expire."""
    # One-shot warm-up: populate the caches without evicting live entries.
    # Guarded so a transient failure (e.g. DB connect) does not kill the task
    # before the refresh loop starts.
    try:
        await preload_data(allow_cached_datasets=True, use_resolved_config=True)
    except Exception:
        _log.exception("Startup dataset preload failed")

    interval = data_preloader_settings.refresh_interval_seconds
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            await preload_data(
                allow_cached_datasets=True, use_resolved_config=True, force_refresh=True
            )
        except Exception:
            _log.exception("Periodic dataset preload failed")


def _log_preload_task_exit(task: asyncio.Task) -> None:
    # The shutdown `gather(..., return_exceptions=True)` swallows exceptions,
    # so log any non-cancellation crash here to make a dead warm-keeper visible.
    if task.cancelled():
        return
    if (exc := task.exception()) is not None:
        _log.error("Dataset preload task terminated unexpectedly", exc_info=exc)


@asynccontextmanager
async def lifespan(app: "StatGPTApp"):
    async with (
        optional_msi_token_manager_context(),
        elasticsearch_client_context(),
        sdmx_query_app_http_client,
        widget_http_client,
    ):
        # Check resources' availability:
        await DatabaseHealthChecker().check()

        # Start data preloading in the background.
        # Keep the task reference so it is not garbage-collected mid-run.
        preload_task = asyncio.create_task(_preload_data_periodically())
        preload_task.add_done_callback(_log_preload_task_exit)

        yield

        # Clean up
        preload_task.cancel()
        await asyncio.gather(preload_task, return_exceptions=True)


class StatGPTApp(DIALApp):
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
