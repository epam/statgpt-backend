from fastapi import APIRouter

from common.config import Versions
from statgpt.config import DialAppConfig
from statgpt.schemas import GitVersionResponse, SettingsResponse

router = APIRouter()


@router.get("/version")
async def version() -> GitVersionResponse:
    return GitVersionResponse(git_commit=Versions.GIT_COMMIT)


@router.get("/settings")
async def settings() -> SettingsResponse:
    return SettingsResponse(
        enable_dev_commands=DialAppConfig.ENABLE_DEV_COMMANDS,
        enable_direct_tool_calls=DialAppConfig.ENABLE_DIRECT_TOOL_CALLS,
        git_commit=Versions.GIT_COMMIT,
    )
