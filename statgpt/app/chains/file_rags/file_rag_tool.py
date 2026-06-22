import typing as t

from langchain_core.runnables import Runnable
from langchain_core.tools import InjectedToolArg
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas import BaseFileRagArtifact
from statgpt.app.schemas.file_rags.dial_rag import RagFilterDial
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import FileRagTool as FileRagToolConfig
from statgpt.common.schemas import RAGVersion, ToolTypes

from .base import BaseRAGFactory
from .dial_rag import DialRagAgentFactory

_RAG_IMPLEMENTATIONS: dict[RAGVersion, type[BaseRAGFactory]] = {
    RAGVersion.DIAL: DialRagAgentFactory,
}


class FileRagArgs(ToolArgs):
    query: t.Annotated[str, GuardrailInput] = Field(description='''\
The query to search an answer for.
- Formulate the query as natural sounding question
- Keep edits to the user query to a minimum
- If user mentions any publication date or type filters, make sure to include them in the query.
- If user query includes phrasing like "according to publications from ..." make sure to include that phrasing in the
  query.
- Keep query concise and to the point, any politeness or greetings should be omitted
''')
    target_prefilter_json: t.Annotated[str | None, InjectedToolArg] = Field(
        default=None,
        description='prefilter to be used in RAG, instead of constructing it from scratch. '
        'used in RAG eval to avoid dependency on prefilter construction in RAG tool. '
        'since RagFilterDial is not JSON-serializable, '
        'it must be passed as a JSON serialized string. ',
    )


class FileRagTool(StatGptTool[FileRagToolConfig], tool_type=ToolTypes.FILE_RAG):
    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_args_schema(cls, tool_config: FileRagToolConfig) -> type[FileRagArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return FileRagArgs

    async def _arun(
        self, inputs: dict, query: str, target_prefilter_json: str | None = None
    ) -> tuple[str, BaseFileRagArtifact]:
        version = self._tool_config.details.version
        implementation = _RAG_IMPLEMENTATIONS[version](self._tool_config, self._channel_config)

        ChainParameters.get_auth_context(inputs)
        chain: Runnable = await implementation.create_chain()

        target_prefilter = (
            RagFilterDial.model_validate_json(target_prefilter_json)
            if target_prefilter_json
            else None
        )
        inputs[ChainParametersConfig.QUERY] = query
        inputs[ChainParametersConfig.TARGET_PREFILTER] = target_prefilter
        res: dict = await chain.ainvoke(inputs)
        logger.info(f"FileRagTool result: {res!r}")

        return res[BaseRAGFactory.FIELD_RESPONSE], res[BaseRAGFactory.FIELD_ARTIFACT]
