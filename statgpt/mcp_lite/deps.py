from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request

from statgpt.app.services.chat_facade import ChannelServiceFacade


async def get_channel_facade() -> ChannelServiceFacade:
    """Resolve the channel facade from the {deployment_id} path parameter.

    Each MCP connection is bound to one channel by URL; this dependency
    extracts that channel and returns its scoped facade.
    """
    request = get_http_request()
    deployment_id = request.path_params.get("deployment_id")
    if not deployment_id:
        raise ToolError("Missing channel (deployment_id) in URL")
    try:
        return await ChannelServiceFacade.get_channel(deployment_id)
    except Exception as e:
        raise ToolError(f"Unknown channel: {deployment_id} ({e!r})")
