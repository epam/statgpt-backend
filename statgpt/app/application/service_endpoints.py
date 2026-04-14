from aidial_sdk.exceptions import HTTPException as DIALException
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi import status

from statgpt.app.schemas import (
    ChannelDatasetsMetadataResponse,
    ChannelMetadataResponse,
    GeneratePythonCodeRequest,
    GeneratePythonCodeResponse,
    SettingsResponse,
)
from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.services.python_code_generator import generate_merged_python_code
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import Versions
from statgpt.common.models.database import get_readonly_session_context_manager
from statgpt.common.services.dataset import DataSetService

router = APIRouter()


class _HeaderOnlyDialRequest:
    """Minimal adapter for auth checks in non-DIAL FastAPI endpoints.

    `create_auth_context()` expects `aidial_sdk.chat_completion.Request`, but for metadata GET routes
    we only have headers. Auth logic uses only `.bearer_token` and `.api_key`.
    """

    def __init__(self, api_key: str | None, bearer_token: str | None):
        self._api_key = api_key
        self._bearer_token = bearer_token

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def bearer_token(self) -> str | None:
        return self._bearer_token

    @classmethod
    def from_request(cls, request: FastAPIRequest) -> "_HeaderOnlyDialRequest":
        api_key = request.headers.get("api-key") or request.headers.get("x-api-key")
        token = request.headers.get("authorization")
        bearer_token = token[7:] if token is not None and token.startswith("Bearer ") else None
        return cls(api_key=api_key, bearer_token=bearer_token)


async def _get_auth_context(request: FastAPIRequest) -> AuthContext:
    try:
        return await create_auth_context(_HeaderOnlyDialRequest.from_request(request))
    except ValueError as e:
        raise DIALException(
            status_code=401,
            code="unauthorized",
            message=f"Unauthorized: {e}",
        )


def _get_deployment_id(request: FastAPIRequest) -> str:
    deployment_id = request.headers.get("x-dial-application-id")
    if deployment_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dial Application ID is required from header.",
        )
    return deployment_id


@router.get("/settings")
async def settings() -> SettingsResponse:
    """Returns settings for the channel.

    NOTE:
        Access this endpoint through the Dial Core API at
        "/v1/deployments/{deployment_id}/route/settings".
    """
    return SettingsResponse(
        enable_dev_commands=dial_app_settings.enable_dev_commands,
        enable_direct_tool_calls=dial_app_settings.enable_direct_tool_calls,
        git_commit=Versions.GIT_COMMIT,
    )


@router.get("/metadata/channel")
async def channel_metadata(
    deployment_id: str = Depends(_get_deployment_id),
    auth_context: AuthContext = Depends(_get_auth_context),
) -> ChannelMetadataResponse:
    """Returns channel metadata.

    NOTE:
        Access this endpoint through the Dial Core API at
        "/v1/deployments/{deployment_id}/route/metadata/channel".
    """

    try:
        service = await ChannelServiceFacade.get_channel(deployment_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The API deployment for this resource does not exist.",
        )

    ch = service.channel
    return ChannelMetadataResponse(
        deployment_id=ch.deployment_id,
        title=ch.title,
        description=ch.description or "",
        locale=service.channel_config.locale,
        country_named_entity_type=service.channel_config.country_named_entity_type,
        tools=service.channel_config.tools,
    )


@router.get("/metadata/datasets")
async def channel_datasets_metadata(
    deployment_id: str = Depends(_get_deployment_id),
    auth_context: AuthContext = Depends(_get_auth_context),
) -> ChannelDatasetsMetadataResponse:
    """Returns datasets metadata for the channel.

    NOTE:
        Access this endpoint through the Dial Core API at
        "/v1/deployments/{deployment_id}/route/metadata/datasets".
    """

    try:
        service = await ChannelServiceFacade.get_channel(deployment_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The API deployment for this resource does not exist.",
        )

    async with get_readonly_session_context_manager() as session:
        datasets = await DataSetService(session).get_channel_dataset_schemas(
            limit=None,
            offset=0,
            channel_id=service.channel.id,
            auth_context=auth_context,
        )

    for ds in datasets:
        resolved = ds.last_completed_version and ds.last_completed_version.resolved_config
        if resolved:
            ds.dataset = ds.dataset.model_copy(update={"details": resolved})

    return ChannelDatasetsMetadataResponse(
        deployment_id=service.channel.deployment_id,
        title=service.channel.title,
        n_datasets=len(datasets),
        datasets=datasets,
    )


@router.post("/python-attachment")
async def generate_python_attachment(
    body: GeneratePythonCodeRequest,
    auth_context: AuthContext = Depends(_get_auth_context),
) -> GeneratePythonCodeResponse:
    """Generate Python attachment from JSON query objects.

    NOTE:
        Access this endpoint through the Dial Core API at
        "/v1/deployments/{deployment_id}/route/python-attachment".
    """
    return GeneratePythonCodeResponse(
        python_code=generate_merged_python_code(body.queries),
    )
