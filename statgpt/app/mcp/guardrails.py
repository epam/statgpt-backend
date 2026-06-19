import logging
from typing import Any

from fastmcp.exceptions import ToolError
from langchain_core.messages import HumanMessage

from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import ChannelConfig

_log = logging.getLogger(__name__)


async def enforce_input_guardrail(
    tool: StatGptTool,
    arguments: dict[str, Any],
    channel_config: ChannelConfig,
    auth_context: AuthContext,
) -> None:
    """Screen a free-text MCP tool argument with the out-of-scope guardrail.

    Mirrors the out-of-scope check the chat completion endpoint runs before the
    Supreme Agent, applied here to a single stateless MCP tool call. Raises
    ``ToolError`` when the request is out of scope or — fail-closed — when the
    guardrail check itself cannot be completed. Does nothing for tools that take
    no arbitrary natural-language input, or when guardrails are disabled (globally
    via ``SKIP_OUT_OF_SCOPE_CHECK`` or per-channel when ``out_of_scope`` is unset).
    """
    if dial_app_settings.skip_out_of_scope_check:
        return
    if channel_config.out_of_scope is None:
        return

    query = tool.get_guardrail_input(arguments)
    if not query:
        return

    checker = OutOfScopeChecker(channel_config)
    messages = [HumanMessage(content=query)]
    try:
        checker_chain = checker.build_checker_chain(messages, auth_context)
        decision = await checker_chain.ainvoke({})
    except Exception:
        # Fail-closed: if the guardrail check cannot run, block the tool call
        # rather than letting an unscreened request through.
        _log.exception("Out-of-scope guardrail failed for MCP tool %s", tool.name)
        raise ToolError("Request blocked: the safety check could not be completed.") from None

    if decision.out_of_scope:
        try:
            response_chain = checker.build_response_chain(
                messages, decision.reasoning, auth_context
            )
            result = await response_chain.ainvoke({})
            message = result.content if isinstance(result.content, str) else str(result.content)
        except Exception:
            # Still block the request; only the polished message could not be produced.
            _log.exception("Out-of-scope response generation failed for MCP tool %s", tool.name)
            message = f"This request is out of scope. Reason: {decision.reasoning}"
        raise ToolError(message)
