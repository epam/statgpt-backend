from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.app.schemas import (
    ChannelDatasetsMetadataResponse,
    ChannelMetadataResponse,
    GitVersionResponse,
    SettingsResponse,
)
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import Versions
from statgpt.common.models.database import get_readonly_session
from statgpt.common.services.dataset import DataSetService
from statgpt.common.settings.dial import dial_settings

router = APIRouter()


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


class _SystemApiKeyAuthContext(AuthContext):
    """Used for service endpoints that need to call external data sources."""

    @property
    def is_system(self) -> bool:
        return True

    @property
    def dial_access_token(self) -> None:
        return None

    @property
    def api_key(self) -> str:
        return dial_settings.api_key.get_secret_value()


@router.get("/openai/deployments/{deployment_id}/metadata/channel")
async def channel_metadata(
    deployment_id: str,
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


@router.get("/openai/deployments/{deployment_id}/metadata/datasets")
async def channel_datasets_metadata(
    deployment_id: str,
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
        auth_context=_SystemApiKeyAuthContext(),
    )

    return ChannelDatasetsMetadataResponse(
        deployment_id=service.channel.deployment_id,
        title=service.channel.title,
        n_datasets=len(datasets),
        datasets=datasets,
    )
