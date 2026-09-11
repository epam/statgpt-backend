from statgpt.app.chains.tools import StatGptTool
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config.utils import replace_envs
from statgpt.common.schemas import PlainContentTool as PlainContentToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import MediaTypes
from statgpt.common.utils.dial import dial_client_factory, download_file_by_path


class _PlainContentToolAuthContext(AuthContext):

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return None

    @property
    def api_key(self) -> str:
        return dial_settings.api_key.get_secret_value()


class PlainContentTool(StatGptTool[PlainContentToolConfig], tool_type=ToolTypes.PLAIN_CONTENT):
    """
    Tool for displaying plain content (text, json, yaml) in Markdown format.
    """

    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        # it's assumed that file is stored under app's API key
        async with dial_client_factory(
            dial_settings.url, _PlainContentToolAuthContext().api_key
        ) as dial:
            content, content_type = await download_file_by_path(
                dial, self._tool_config.details.file_path
            )
        text = content.decode('utf-8')
        if self._tool_config.details.replace_envs:
            text = replace_envs(text, prefix="TTYD_TOOL_PLAIN_CONTENT_")
        if content_type == MediaTypes.YAML:
            response = f"```yaml\n{text}\n```"
        elif content_type == MediaTypes.JSON:
            response = f"```json\n{text}\n```"
        elif content_type == MediaTypes.PLAIN_TEXT or content_type == MediaTypes.MARKDOWN:
            response = text
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))
