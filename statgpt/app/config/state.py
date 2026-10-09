class StateVarsConfig:
    """Keys to access artifacts stored in DIAL message state."""

    # TODO: move to using pydantic model for statgpt state

    V2_QUERY_BUILDER_AGENT_STATE = "v2_query_builder_agent_state"  # todo: delete along with history

    SHOW_DEBUG_STAGES = "show_debug_stages"
    # "cmd_" prefix indicates the command
    CMD_OUT_OF_SCOPE_ONLY = "cmd_out_of_scope_only"
    CMD_RAG_PREFILTER_ONLY = "cmd_rag_prefilter_only"
    CMD_SKIP_DATA_QUERY_SUMMARIZATION = "cmd_skip_data_query_summarization"
    CMD_SKIP_TOOLS_EXECUTION = "cmd_skip_tools_execution"
    ERROR = 'error'
    LLM_CALL_DURATIONS = "llm_call_durations"
    RECEIVED_MESSAGES = "received_messages"

    # values used in Agentic approach
    DIRECT_TOOL_CALLS = "direct_tool_calls"
    OUT_OF_SCOPE = "out_of_scope"
    OUT_OF_SCOPE_REASONING = "out_of_scope_reasoning"
    TOOL_MESSAGES = "tool_messages"

    # Supreme Agent <-> Deep Research clarification session, carried across turns.
    DEEP_RESEARCH_SESSION = "deep_research_session"
    # Set for exactly one turn: the turn on which Deep Research delivered its final report. Read
    # when emitting the per-message Deep Research toggle form schema so the toggle disarms itself.
    # One-shot by construction: `init_state` does not carry it forward to the next turn.
    DEEP_RESEARCH_REPORT_DELIVERED = "deep_research_report_delivered"
    # Set for exactly one turn: the turn on which the user had Deep Research mode on, but the query
    # check decided that the query does not suit Deep Research and deactivated the mode, so the
    # query was answered as a normal turn. Disarms the toggle and is one-shot, like
    # `DEEP_RESEARCH_REPORT_DELIVERED`.
    DEEP_RESEARCH_AUTO_DEACTIVATED = "deep_research_auto_deactivated"
