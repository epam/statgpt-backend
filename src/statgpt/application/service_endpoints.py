from fastapi import APIRouter

from common.config import Versions
from statgpt.schemas import GitVersionResponse, SettingsResponse
from statgpt.settings.dial_app import dial_app_settings

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
