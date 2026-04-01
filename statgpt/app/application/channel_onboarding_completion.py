import logging

from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response, Role
from aidial_sdk.deployment.configuration import ConfigurationRequest, ConfigurationResponse
from aidial_sdk.exceptions import HTTPException as DIALException
from aidial_sdk.exceptions import InvalidRequestError

from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.services.onboarding import OnboardingService, OnboardingState
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.onboarding import OnboardingConfig

_log = logging.getLogger(__name__)


class ChannelOnboardingCompletion(ChatCompletion):

    def __init__(self, deployment_id: str) -> None:
        self._deployment_id = deployment_id

    async def chat_completion(self, request: Request, response: Response) -> None:
        try:
            service = await ChannelServiceFacade.get_channel(self._deployment_id)
        except Exception as e:
            _log.error(e)
            raise DIALException(
                status_code=404,
                code="deployment_not_found",
                message="The API deployment for this resource does not exist.",
            )

        onboarding_config = service.channel_config.onboarding
        if onboarding_config is None:
            raise DIALException(
                status_code=404,
                code="onboarding_not_found",
                message="Onboarding configuration not found for this deployment.",
            )

        auth_context = await create_auth_context(
            request,
            bearer_token_required=service.channel_config.bearer_token_required,
        )

        with response.create_choice() as choice:
            await self._run_onboarding(
                request=request,
                choice=choice,
                onboarding_config=onboarding_config,
                channel_service=service,
                auth_context=auth_context,
            )

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:
        try:
            service = await ChannelServiceFacade.get_channel(self._deployment_id)
        except Exception as e:
            _log.error(e)
            raise DIALException(
                status_code=404,
                code="deployment_not_found",
                message="The API deployment for this resource does not exist.",
            )
        if onboarding_config := service.channel_config.onboarding:
            onboarding_service = OnboardingService(onboarding_config)
            return onboarding_service.get_initial_form_schema()
        raise DIALException(
            status_code=404,
            code="onboarding_not_found",
            message="Onboarding configuration not found for this deployment.",
        )

    @classmethod
    def _extract_state(cls, request: Request) -> OnboardingState:
        """Extract onboarding state from the last assistant message."""
        messages = request.messages
        # ignore ASSISTANT message if it is the first message
        if len(messages) <= 1:
            return OnboardingState()
        if messages[0].role == Role.ASSISTANT:
            messages = messages[1:]
        # Find last assistant message with onboarding state
        for message in reversed(messages):
            if message.role == Role.ASSISTANT:
                if message.custom_content is None or message.custom_content.state is None:
                    raise InvalidRequestError(
                        "Onboarding state not found in the last assistant message."
                    )
                try:
                    state = OnboardingState.model_validate(message.custom_content.state)
                    return state
                except Exception as e:
                    raise InvalidRequestError(f"Invalid onboarding state: {e}")

        # If no assistant messages found, return initial state
        return OnboardingState()

    @classmethod
    def _extract_button_clicked(cls, request: Request) -> str | None:
        """Extract the button ID clicked by the user from the last user message."""
        if not request.messages:
            return None

        last_message = request.messages[-1]
        if last_message.role != Role.USER:
            return None

        if not last_message.custom_content or not last_message.custom_content.form_value:
            # for first user message the value is stored in configuration field
            configuration = request.custom_fields.configuration if request.custom_fields else None
            if configuration is not None and isinstance(configuration, dict):
                return configuration.get("choice") or configuration.get("completion")
            return None

        form_value = last_message.custom_content.form_value
        # The form field is named "choice" or "completion" depending on the step
        # For navigation forms, it's "choice"
        # For completion form, it's "completion"
        if isinstance(form_value, dict):
            return form_value.get("choice") or form_value.get("completion")

        return None

    async def _run_onboarding(
        self,
        request: Request,
        choice: Choice,
        onboarding_config: OnboardingConfig,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        onboarding_service = OnboardingService(onboarding_config)
        state = self._extract_state(request)
        button_clicked = self._extract_button_clicked(request)

        await onboarding_service.set_content(
            state=state,
            button_clicked=button_clicked,
            choice=choice,
            channel_service=channel_service,
            auth_context=auth_context,
        )
