from common.auth.auth_context import AuthContext
from common.config.utils import replace_envs
from common.schemas import PlainContentTool as PlainContentToolConfig
from common.schemas import ToolTypes
from common.settings.dial import dial_settings
from common.utils import MediaTypes
from common.utils.dial import dial_core_factory
from statgpt.chains.tools import StatGptTool
from statgpt.schemas import ToolArtifact, ToolMessageState


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
        async with dial_core_factory(
            dial_settings.url, _PlainContentToolAuthContext().api_key
        ) as dial_core:
            content, content_type = await dial_core.get_file_by_path(
                self._tool_config.details.file_path
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
