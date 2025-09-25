from pydantic import BaseModel, Field


class GitVersionResponse(BaseModel):
    git_commit: str = Field()


class SettingsResponse(BaseModel):
    enable_dev_commands: bool = Field()
    enable_direct_tool_calls: bool = Field()
    git_commit: str = Field()
