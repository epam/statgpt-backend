from enum import StrEnum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(StrEnum):
    LOCAL = "LOCAL"
    DIAL = "DIAL"


class DialAuthMode(StrEnum):
    USER_TOKEN = "user_token"
    """User token is passed to the DIAL API calls"""
    API_KEY = "api_key"
    """Configured API key is passed to the DIAL API calls"""


class DialAppSettings(BaseSettings):
    """
    DIAL application configuration settings
    """

    model_config = SettingsConfigDict(env_prefix="")

    mode: AppMode = Field(
        default=AppMode.DIAL,
        alias="APP_MODE",
        description="Application mode (LOCAL or DIAL)",
    )

    dial_app_name: str = Field(
        default="StatGPT",
        alias="DIAL_APP_NAME",
        description="Name of the DIAL application",
    )

    dial_auth_mode: DialAuthMode = Field(
        default=DialAuthMode.USER_TOKEN,
        alias="DIAL_AUTH_MODE",
        description="Authentication mode for DIAL API calls",
    )

    dial_log_level: str = Field(
        default="INFO", alias="DIAL_LOG_LEVEL", description="Log level for DIAL application"
    )

    dial_show_stage_seconds: bool = Field(
        default=False,
        alias="DIAL_SHOW_STAGE_SECONDS",
        description="Show stage execution time in seconds",
    )

    dial_show_debug_stages: bool = Field(
        default=False, alias="DIAL_SHOW_DEBUG_STAGES", description="Show debug stages information"
    )

    enable_dev_commands: bool = Field(
        default=False, alias="ENABLE_DEV_COMMANDS", description="Enable development commands"
    )

    enable_direct_tool_calls: bool = Field(
        default=False, alias="ENABLE_DIRECT_TOOL_CALLS", description="Enable direct tool calls"
    )

    official_dataset_label: str = Field(
        default="⭐", alias="OFFICIAL_DATASET_LABEL", description="Label for official datasets"
    )

    skip_out_of_scope_check: bool = Field(
        default=False,
        alias="SKIP_OUT_OF_SCOPE_CHECK",
        description="Skip out-of-scope check for queries",
    )

    cmd_out_of_scope_only: bool = Field(
        default=False,
        alias="CMD_OUT_OF_SCOPE_ONLY",
        description="Only check if query is out of scope",
    )

    cmd_rag_prefilter_only: bool = Field(
        default=False, alias="CMD_RAG_PREFILTER_ONLY", description="Only apply RAG prefilter"
    )

    eval_dial_role: Optional[str] = Field(
        default=None, alias="EVAL_DIAL_ROLE", description="DIAL role for evaluation"
    )


# Create singleton instance
dial_app_settings = DialAppSettings()
