import typing as t

from langchain_core.runnables import Runnable
from langchain_core.tools import InjectedToolArg
from pydantic import Field

from common.config import multiline_logger as logger
from common.schemas import FileRagTool as FileRagToolConfig
from common.schemas import RAGVersion, ToolTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.config import ChainParametersConfig
from statgpt.schemas import BaseFileRagArtifact
from statgpt.schemas.file_rags.dial_rag import RagFilterDial

from .base import BaseRAGFactory
from .dial_rag import DialRagAgentFactory

_RAG_IMPLEMENTATIONS: dict[RAGVersion, type[BaseRAGFactory]] = {
    RAGVersion.DIAL: DialRagAgentFactory,
}


class FileRagArgs(ToolArgs):
    query: str = Field(
        description='''\
The query to search an answer for.
- Formulate the query as natural sounding question
- Keep edits to the user query to a minimum
- If user mentions any publication date or type filters, make sure to include them in the query.
- If user query includes phrasing like "according to publications from ..." make sure to include that phrasing in the
  query.
- Keep query concise and to the point, any politeness or greetings should be omitted
'''
    )
    target_prefilter_json: t.Annotated[str | None, InjectedToolArg] = Field(
        default=None,
        description='prefilter to be used in RAG, instead of constructing it from scratch. '
        'used in RAG eval to avoid dependency on prefilter construction in RAG tool. '
        'since RagFilterDial is not JSON-serializable, '
        'it must be passed as a JSON serialized string. ',
    )


class FileRagTool(StatGptTool[FileRagToolConfig], tool_type=ToolTypes.FILE_RAG):
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
