class McpRequestError(Exception):
    """Base class for errors resolving the channel context of an MCP request."""


class MissingDeploymentIdError(McpRequestError):
    """Raised when the request path does not carry a ``deployment_id``."""

    def __init__(self, message: str = "Missing deployment_id in path"):
        super().__init__(message)
