from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request as FastAPIRequest, status
from sqlalchemy.ext.asyncio import AsyncSession

from aidial_sdk.chat_completion import Request
from aidial_sdk.exceptions import HTTPException as DIALException

from statgpt.app.schemas import (
    ChannelDatasetsMetadataResponse,
    ChannelMetadataResponse,
    GitVersionResponse,
    SettingsResponse,
)
from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.config import Versions
from statgpt.common.models.database import get_readonly_session
from statgpt.common.services.dataset import DataSetService

router = APIRouter()


class _HeaderOnlyDialRequest:
    """Minimal adapter for auth checks in non-DIAL FastAPI endpoints.

    `create_auth_context()` expects `aidial_sdk.chat_completion.Request`, but for metadata GET routes
    we only have headers. Auth logic uses only `.jwt` and `.api_key`.
    """

    def __init__(self, *, jwt: str | None, api_key: str | None):
        self.jwt = jwt
        self.api_key = api_key


def _dial_request_from_fastapi_request(request: FastAPIRequest) -> Request:
    jwt = request.headers.get("authorization")
    api_key = request.headers.get("api-key") or request.headers.get("x-api-key") or request.headers.get("Api-Key")
    return cast(Request, _HeaderOnlyDialRequest(jwt=jwt, api_key=api_key))


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
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_readonly_session),
) -> ChannelMetadataResponse:

    try:
        await create_auth_context(_dial_request_from_fastapi_request(request))
    except ValueError as e:
        raise DIALException(
            status_code=401,
            code="unauthorized",
            message=f"Unauthorized: {e}",
        )

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
    request: FastAPIRequest,
    session: AsyncSession = Depends(get_readonly_session),
) -> ChannelDatasetsMetadataResponse:

    try:
        auth_context = await create_auth_context(_dial_request_from_fastapi_request(request))
    except ValueError as e:
        raise DIALException(
            status_code=401,
            code="unauthorized",
            message=f"Unauthorized: {e}",
        )

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
