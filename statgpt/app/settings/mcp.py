from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpSettings(BaseSettings):
    """Settings for the MCP server's cross-cutting guards: per-caller rate limits on
    model-facing tools and a payload budget that keeps tool results within the host's limits.

    Both guards apply only to model-facing tools. App-only tools (``mcp_visibility`` without
    ``"model"``) are for internal application use, are not loaded into an agent's context, and
    are always exempt.
    """

    model_config = SettingsConfigDict(env_prefix="")

    # ~~~~~~~~~~~~~ rate limiting (#600) ~~~~~~~~~~~~~

    mcp_rate_limit_enabled: bool = Field(
        default=True,
        alias="MCP_RATE_LIMIT_ENABLED",
        description="Enforce per-caller rate limits on model-facing MCP tools.",
    )

    mcp_rate_limit_window_seconds: float = Field(
        default=60.0,
        alias="MCP_RATE_LIMIT_WINDOW_SECONDS",
        gt=0,
        description=(
            "Sustained window, in seconds, over which the per-window allowances below are"
            " replenished (token-bucket refill period)."
        ),
    )

    # Per cost class: `*_per_window` is the sustained allowance replenished over the window;
    # `*_burst` is the bucket capacity, i.e. how many calls can be spent back-to-back.
    mcp_rate_limit_cheap_per_window: int = Field(
        default=120, alias="MCP_RATE_LIMIT_CHEAP_PER_WINDOW", ge=1
    )
    mcp_rate_limit_cheap_burst: int = Field(default=40, alias="MCP_RATE_LIMIT_CHEAP_BURST", ge=1)
    mcp_rate_limit_moderate_per_window: int = Field(
        default=60, alias="MCP_RATE_LIMIT_MODERATE_PER_WINDOW", ge=1
    )
    mcp_rate_limit_moderate_burst: int = Field(
        default=20, alias="MCP_RATE_LIMIT_MODERATE_BURST", ge=1
    )
    mcp_rate_limit_expensive_per_window: int = Field(
        default=20, alias="MCP_RATE_LIMIT_EXPENSIVE_PER_WINDOW", ge=1
    )
    mcp_rate_limit_expensive_burst: int = Field(
        default=8, alias="MCP_RATE_LIMIT_EXPENSIVE_BURST", ge=1
    )

    # ~~~~~~~~~~~~~ payload budget (#601) ~~~~~~~~~~~~~

    mcp_payload_budget_enabled: bool = Field(
        default=True,
        alias="MCP_PAYLOAD_BUDGET_ENABLED",
        description="Enforce a serialized-size budget on model-facing MCP tool results.",
    )

    mcp_payload_max_chars: int = Field(
        default=120_000,
        alias="MCP_PAYLOAD_MAX_CHARS",
        ge=1,
        description=(
            "Maximum serialized size, in characters, of a model-facing tool result. Kept below"
            " the host caps (Claude.ai/Desktop truncate at ~150,000 characters). A result over"
            " this budget has its heavy embedded resources dropped with a note; if it still does"
            " not fit, the tool returns an actionable error asking the caller to narrow the query."
        ),
    )


mcp_settings = McpSettings()
