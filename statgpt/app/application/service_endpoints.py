from aidial_sdk.exceptions import HTTPException as DIALException
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.app.schemas import (
    ChannelDatasetsMetadataResponse,
    ChannelMetadataResponse,
    GitVersionResponse,
    SettingsResponse,
)
from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import Versions
from statgpt.common.models.database import get_readonly_session
from statgpt.common.services.dataset import DataSetService

router = APIRouter()


class _HeaderOnlyDialRequest:
    """Minimal adapter for auth checks in non-DIAL FastAPI endpoints.

    `create_auth_context()` expects `aidial_sdk.chat_completion.Request`, but for metadata GET routes
    we only have headers. Auth logic uses only `.bearer_token` and `.api_key`.
    """

    api_key: str | None
    bearer_token: str | None

    def __init__(self, request: FastAPIRequest):
        self.api_key = request.headers.get("api-key") or request.headers.get("x-api-key")
        token = request.headers.get("authorization")
        self.bearer_token = token[7:] if token is not None and token.startswith("Bearer ") else None


async def _get_auth_context(request: FastAPIRequest) -> AuthContext:
    try:
        return await create_auth_context(_HeaderOnlyDialRequest(request))
    except ValueError as e:
        raise DIALException(
            status_code=401,
            code="unauthorized",
            message=f"Unauthorized: {e}",
        )


@router.get("/version")
async def version() -> GitVersionResponse:
    return GitVersionResponse(git_commit=Versions.GIT_COMMIT)


@router.get("/settings")
async def settings() -> SettingsResponse:
    return SettingsResponse(
        enable_dev_commands=dial_app_settings.enable_dev_commands,
        enable_direct_tool_calls=dial_app_settings.enable_direct_tool_calls,
        git_commit=Versions.GIT_COMMIT,
    )


@router.get("/statgpt/openai/deployments/{deployment_id}/metadata/channel")
async def channel_metadata(
    deployment_id: str,
    auth_context: AuthContext = Depends(_get_auth_context),
    session: AsyncSession = Depends(get_readonly_session),
) -> ChannelMetadataResponse:

    try:
        service = await ChannelServiceFacade.get_channel(session, deployment_id)
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
        llm_model=ch.llm_model,
        tools=service.channel_config.tools,
    )


@router.get("/statgpt/openai/deployments/{deployment_id}/metadata/datasets")
async def channel_datasets_metadata(
    deployment_id: str,
    auth_context: AuthContext = Depends(_get_auth_context),
    session: AsyncSession = Depends(get_readonly_session),
) -> ChannelDatasetsMetadataResponse:

    try:
        service = await ChannelServiceFacade.get_channel(session, deployment_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The API deployment for this resource does not exist.",
        )

    datasets = await DataSetService(session).get_channel_dataset_schemas(
        limit=None,
        offset=0,
        channel_id=service.channel.id,
        auth_context=auth_context,
    )

    return ChannelDatasetsMetadataResponse(
        deployment_id=service.channel.deployment_id,
        title=service.channel.title,
        n_datasets=len(datasets),
        datasets=datasets,
    )
