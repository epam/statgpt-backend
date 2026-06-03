from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DialAppSettings(BaseSettings):
    """
    DIAL application configuration settings
    """

    model_config = SettingsConfigDict(env_prefix="")

    dial_app_name: str = Field(
        default="StatGPT",
        alias="DIAL_APP_NAME",
        description="Name of the DIAL application",
    )

    dial_show_stage_seconds: bool = Field(
        default=False,
        alias="DIAL_SHOW_STAGE_SECONDS",
        description="Show stage execution time in seconds",
    )

    dial_show_debug_stages: bool = Field(
        default=False, alias="DIAL_SHOW_DEBUG_STAGES", description="Show debug stages information"
    )

    dial_show_debug_attachments: bool = Field(
        default=False,
        alias="DIAL_SHOW_DEBUG_ATTACHMENTS",
        description="Show debug attachments in chat completion responses",
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

    cmd_skip_data_query_summarization: bool = Field(
        default=False,
        alias="CMD_SKIP_DATA_QUERY_SUMMARIZATION",
        description="Skip data query summarization step",
    )

    cmd_skip_tools_execution: bool = Field(
        default=False,
        alias="CMD_SKIP_TOOLS_EXECUTION",
        description="Skip tools execution step",
    )

    track_llm_call_durations: bool = Field(
        default=False,
        alias="TRACK_LLM_CALL_DURATIONS",
        description="Track and report LLM call durations per model in debug performance stage and DIAL state",
    )

    indicators_total_cache_ttl: int = Field(
        default=60,
        alias="INDICATORS_TOTAL_CACHE_TTL",
        description=(
            "TTL in seconds for the in-process cache of the per-channel indicators total. "
            "The figure is non-transactional; staleness within this window is acceptable."
        ),
    )

    dial_system_user_context_roles: str | None = Field(
        default=None,
        alias="DIAL_SYSTEM_USER_CONTEXT_ROLES",
        description=(
            "Comma-separated list of DIAL roles that can receive system user context "
            "when no JWT is present."
        ),
    )

    @property
    def system_user_context_roles_set(self) -> set[str]:
        """Parse comma-separated roles into a set."""
        if not self.dial_system_user_context_roles:
            return set()
        return {
            role.strip() for role in self.dial_system_user_context_roles.split(",") if role.strip()
        }


# Create singleton instance
dial_app_settings = DialAppSettings()
