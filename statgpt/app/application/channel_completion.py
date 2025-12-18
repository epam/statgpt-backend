import logging
from collections.abc import Iterable
from datetime import datetime

import openai
from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response
from aidial_sdk.deployment.configuration import ConfigurationRequest, ConfigurationResponse
from aidial_sdk.exceptions import HTTPException as DIALException

from statgpt.app.chains import MainChainFactory
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig as ParamsConfig
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.schemas.state import State
from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.app.utils.dial_stages import optional_timed_stage
from statgpt.common.models.database import get_readonly_session_contex_manager
from statgpt.common.schemas.dial import Pricing
from statgpt.common.schemas.token_usage import TokenUsagePricedItem
from statgpt.common.settings.application import application_settings
from statgpt.common.settings.dial import dial_settings
from statgpt.common.settings.langchain import langchain_settings
from statgpt.common.utils import dial_core_factory
from statgpt.common.utils.callbacks import LCMessageLoggerAsync, TokenUsageByModelsCallback
from statgpt.common.utils.dial.model_pricing import ModelPricingAuthContext, ModelPricingGetter
from statgpt.common.utils.token_usage_context import TokenUsageManager, token_usage_context
from statgpt.common.utils.token_usage_utils import TokenUsageCostCalculator, TokenUsageDisplayer

_log = logging.getLogger(__name__)


class ChannelCompletion(ChatCompletion):

    def __init__(self, deployment_id: str) -> None:
        self._deployment_id = deployment_id

    async def chat_completion(self, request: Request, response: Response) -> None:
        start_time = datetime.now()
        _log.info(f"{self._deployment_id=}")

        if request.custom_fields and request.custom_fields.configuration:
            configuration = StatGPTConfiguration.model_validate(request.custom_fields.configuration)
        else:
            configuration = StatGPTConfiguration()

        _log.debug(f"StatGPTConfiguration: {configuration}")

        # Take memory snapshot at request start (only in debug mode)
        if application_settings.memory_debug:
            from statgpt.common.utils.memory_profiler import memory_profiler

            memory_profiler.take_snapshot(
                f"request_start_{self._deployment_id}_{start_time.isoformat()}"
            )

        async with get_readonly_session_contex_manager() as db_session:
            try:
                service = await ChannelServiceFacade.get_channel(db_session, self._deployment_id)
            except Exception as e:
                _log.error(e)
                raise DIALException(
                    status_code=404,
                    code="deployment_not_found",
                    message="The API deployment for this resource does not exist.",
                )
            await self._channel_completion(request, response, service, start_time, configuration)

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:

        async with get_readonly_session_contex_manager() as db_session:
            try:
                service = await ChannelServiceFacade.get_channel(db_session, self._deployment_id)
            except Exception as e:
                _log.error(e)
                raise DIALException(
                    status_code=404,
                    code="deployment_not_found",
                    message="The API deployment for this resource does not exist.",
                )
            return service.dial_channel_configuration

    @classmethod
    async def _channel_completion(
        cls,
        request: Request,
        response: Response,
        service: ChannelServiceFacade,
        start_time: datetime,
        configuration: StatGPTConfiguration,
    ) -> None:
        main_chain_factory = MainChainFactory(service.channel_config)
        chain = await main_chain_factory.create_chain()
        with response.create_choice() as choice:
            try:
                auth_context = await create_auth_context(request)
            except Exception:
                _log.exception("Failed to create auth context")
                choice.append_content("401 Unauthorized: Invalid or missing authentication.")
                return

            inputs = {
                ParamsConfig.REQUEST: request,
                ParamsConfig.AUTH_CONTEXT: auth_context,
                ParamsConfig.CHOICE: choice,
                ParamsConfig.DATA_SERVICE: service,
                ParamsConfig.STATE: State.init_state(request),
                ParamsConfig.SKIP_OUT_OF_SCOPE_CHECK: dial_app_settings.skip_out_of_scope_check,
                ParamsConfig.START_OF_REQUEST: start_time,
                ParamsConfig.CONFIGURATION: configuration,
            }

            callbacks: list = []
            if langchain_settings.use_custom_logger_callback:
                callbacks.append(LCMessageLoggerAsync())

            with token_usage_context() as token_usage_manager:
                callbacks.append(TokenUsageByModelsCallback())
                state = ChainParameters.get_state(inputs)  # default in case of error
                try:
                    name = '[DEBUG] Performance of request'
                    debug = state.show_debug_stages
                    with optional_timed_stage(choice=choice, name=name, enabled=debug) as stage:
                        inputs[ParamsConfig.PERFORMANCE_STAGE] = stage
                        chains_response: dict = await chain.ainvoke(
                            inputs, config={'callbacks': callbacks}  # type: ignore[typeddict-item]
                        )
                    state = ChainParameters.get_state(chains_response)
                    state.error = None
                except openai.ContentFilterFinishReasonError as e:
                    _log.exception(e)
                    choice.append_content(
                        "The query was blocked by the LLM provider content filter for violating safety guidelines."
                    )
                    state.error = str(e)
                except Exception as e:
                    _log.exception(e)
                    choice.append_content("An error occurred while processing your request.")
                    state.error = str(e)

                priced_usage = await cls._calc_token_usage_costs(token_usage_manager)

            token_usage_config = service.channel_config.token_usage
            show_cost_stage = not token_usage_config.debug_only or state.show_debug_stages
            with optional_timed_stage(
                choice=choice, name=token_usage_config.stage_name, enabled=show_cost_stage
            ) as stage:
                if stage:
                    table = TokenUsageDisplayer.as_markdown_table(priced_usage)
                    stage.append_content(table)

            cls._add_usage_per_model(priced_usage, response)
            cls.set_dial_state(state, choice)

    @classmethod
    async def _calc_token_usage_costs(
        cls, token_usage_manager: TokenUsageManager
    ) -> list[TokenUsagePricedItem]:
        usage = token_usage_manager.get_usage()

        models = {item.model for item in usage}
        models_pricing = await cls._load_pricing(models)

        priced_usage = TokenUsageCostCalculator(models_pricing).get_token_usage_with_costs(usage)
        return priced_usage

    @staticmethod
    async def _load_pricing(models: Iterable[str]) -> dict[str, Pricing]:
        async with dial_core_factory(
            base_url=dial_settings.url, api_key=ModelPricingAuthContext().api_key
        ) as dial_core:
            getter = ModelPricingGetter(dial_core)

            res = {}
            for model in models:
                if pricing := await getter.get_model_pricing(model):
                    res[model] = pricing
        return res

    @staticmethod
    def _add_usage_per_model(priced_usage: list[TokenUsagePricedItem], response: Response) -> None:
        for item in priced_usage:
            response.add_usage_per_model(
                model=item.model,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
            )

    @staticmethod
    def set_dial_state(state: State, choice: Choice) -> None:
        _log.info(f"setting following DIAL state: {state}")
        choice.set_state(state.model_dump(exclude_none=True))
