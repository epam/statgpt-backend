class StateVarsConfig:
    """Keys to access artifacts stored in DIAL message state."""

    # TODO: move to using pydantic model for statgpt state

    V2_QUERY_BUILDER_AGENT_STATE = "v2_query_builder_agent_state"  # todo: delete along with history

    SHOW_DEBUG_STAGES = "show_debug_stages"
    # "cmd_" prefix indicates the command
    CMD_OUT_OF_SCOPE_ONLY = "cmd_out_of_scope_only"
    CMD_RAG_PREFILTER_ONLY = "cmd_rag_prefilter_only"
    ERROR = 'error'

    # values used in Agentic approach
    DIRECT_TOOL_CALLS = "direct_tool_calls"
    OUT_OF_SCOPE = "out_of_scope"
    OUT_OF_SCOPE_REASONING = "out_of_scope_reasoning"
    TOOL_MESSAGES = "tool_messages"
