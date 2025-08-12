from collections.abc import Iterable

import openai
from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response
from aidial_sdk.deployment.configuration import ConfigurationRequest, ConfigurationResponse
from aidial_sdk.exceptions import HTTPException as DIALException

from common.config import DialConfig, LangChainConfig
from common.config.logging import multiline_logger as logger
from common.models import get_session_contex_manager
from common.schemas.dial import Pricing
from common.schemas.token_usage import TokenUsagePricedItem
from common.utils import dial_core_factory
from common.utils.callbacks import LCMessageLoggerAsync, TokenUsageByModelsCallback
from common.utils.dial.model_pricing import ModelPricingAuthContext, ModelPricingGetter
from common.utils.token_usage_context import TokenUsageManager, token_usage_context
from common.utils.token_usage_utils import TokenUsageCostCalculator, TokenUsageDisplayer
from statgpt.chains import MainChainFactory
from statgpt.chains.parameters import ChainParameters
from statgpt.config import ChainParametersConfig as ParamsConfig
from statgpt.config import DialAppConfig, StateVarsConfig
from statgpt.security import UserAuthContext
from statgpt.services import ChannelServiceFacade
from statgpt.utils.dial_tools import continuous_choice


class ChannelCompletion(ChatCompletion):
    async def chat_completion(self, request: Request, response: Response) -> None:
        deployment_id = request.original_request.path_params["deployment_id"]
        logger.info(f"{deployment_id=}")

        async with get_session_contex_manager() as db_session:
            try:
                service = await ChannelServiceFacade.get_channel(db_session, deployment_id)
            except Exception as e:
                logger.error(e)
                raise DIALException(
                    status_code=404,
                    code="deployment_not_found",
                    message="The API deployment for this resource does not exist.",
                )
            await self._channel_completion(request, response, service)

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:
        deployment_id = request.original_request.path_params["deployment_id"]
        async with get_session_contex_manager() as db_session:
            try:
                service = await ChannelServiceFacade.get_channel(db_session, deployment_id)
            except Exception as e:
                logger.error(e)
                raise DIALException(
                    status_code=404,
                    code="deployment_not_found",
                    message="The API deployment for this resource does not exist.",
                )
        return service.dial_channel_configuration

    @classmethod
    async def _channel_completion(
        cls, request: Request, response: Response, service: ChannelServiceFacade
    ) -> None:
        main_chain_factory = MainChainFactory(service.channel_config)
        chain = await main_chain_factory.create_chain()
        with response.create_choice() as choice:
            with continuous_choice(choice):
                inputs = {
                    ParamsConfig.REQUEST: request,
                    ParamsConfig.AUTH_CONTEXT: UserAuthContext(request),
                    ParamsConfig.CHOICE: choice,
                    ParamsConfig.DATA_SERVICE: service,
                    ParamsConfig.STATE: cls.init_state(request),
                    ParamsConfig.SKIP_OUT_OF_SCOPE_CHECK: DialAppConfig.SKIP_OUT_OF_SCOPE_CHECK,
                }

                callbacks = []
                if LangChainConfig.USE_CUSTOM_LOGGER_CALLBACK:
                    callbacks.append(LCMessageLoggerAsync())

                with token_usage_context() as token_usage_manager:
                    callbacks.append(TokenUsageByModelsCallback())
                    state = ChainParameters.get_state(inputs)  # default in case of error
                    try:
                        chains_response: dict = await chain.ainvoke(
                            inputs, config={'callbacks': callbacks}
                        )
                        state = ChainParameters.get_state(chains_response)
                        state[StateVarsConfig.ERROR] = None
                    except openai.ContentFilterFinishReasonError as e:
                        logger.exception(e)
                        choice.append_content(
                            "The query was blocked by the LLM provider content filter for violating safety guidelines."
                        )
                        state[StateVarsConfig.ERROR] = str(e)
                    except Exception as e:
                        logger.exception(e)
                        choice.append_content("An error occurred while processing your request.")
                        state[StateVarsConfig.ERROR] = str(e)

                    priced_usage = await cls._calc_token_usage_costs(token_usage_manager)

                token_usage_config = service.channel_config.token_usage
                if not token_usage_config.debug_only or state[StateVarsConfig.SHOW_DEBUG_STAGES]:
                    with choice.create_stage(token_usage_config.stage_name) as stage:
                        table = TokenUsageDisplayer.as_markdown_table(priced_usage)
                        stage.append_content(table)

                cls._add_usage_per_model(priced_usage, response)
                cls.set_dial_state(state, choice)

    @staticmethod
    def init_state(request: Request) -> dict:
        defaults = {
            # set default commands values from env vars
            StateVarsConfig.SHOW_DEBUG_STAGES: DialAppConfig.DIAL_SHOW_DEBUG_STAGES,
            StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY: DialAppConfig.CMD_OUT_OF_SCOPE_ONLY,
            StateVarsConfig.CMD_RAG_PREFILTER_ONLY: DialAppConfig.CMD_RAG_PREFILTER_ONLY,
        }

        if len(request.messages) < 2:
            return defaults

        # for some commands, we set their values
        # to be the same as in previous assistant response (if available)
        last_response = request.messages[-2]
        if custom_content := last_response.custom_content:
            if state := custom_content.state:
                if (show_debug_stages := state.get(StateVarsConfig.SHOW_DEBUG_STAGES)) is not None:
                    return {**defaults, StateVarsConfig.SHOW_DEBUG_STAGES: show_debug_stages}

        return defaults

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
            base_url=DialConfig.get_url(), api_key=ModelPricingAuthContext().api_key
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
    def set_dial_state(state: dict, choice: Choice) -> None:
        logger.info(f"setting following DIAL state: {state}")
        choice.set_state(state)
