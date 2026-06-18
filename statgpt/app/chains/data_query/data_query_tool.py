from typing import Any

from langchain_core.runnables import Runnable
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas.enums import ToolTypes

from .parameters import DataQueryParameters
from .query_builder.factory import QueryBuilderFactory


class DataQueryArgs(ToolArgs):
    query: str = Field(
        description="An indicator with all of its filters in plain text. "
        "Specify all countries, dates, frequencies, datasets the user requested. "
        "The query must reflect only what the user asked for — do not add, infer, or expand any filters."
    )


class DataQueryTool(StatGptTool[DataQueryToolConfig], tool_type=ToolTypes.DATA_QUERY):

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[DataQueryArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DataQueryArgs

    def get_guardrail_input(self, arguments: dict[str, Any]) -> str | None:
        return arguments.get("query")

    async def _arun(self, inputs: dict, query: str) -> tuple[str, DataQueryArtifact]:
        factory = QueryBuilderFactory(self._tool_config.details)

        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        chain: Runnable = await factory.create_chain(inputs)

        res: dict = await chain.ainvoke(inputs)
        logger.info(f"DataQueryTool result: {res!r}")

        response_str: str = res[DataQueryParameters.RESPONSE_FIELD]
        data_responses: dict[str, DataResponse] = {
            k: v
            for k, v in res.get(ChainParametersConfig.DATA_RESPONSES, {}).items()
            if v is not None
        }
        state: QueryBuilderAgentState = res.get(DataQueryParameters.STATE, QueryBuilderAgentState())
        eval_attachment: DataQueryEvalAttachment = res.get(
            DataQueryParameters.EVAL_ATTACHMENT, DataQueryEvalAttachment()
        )

        return response_str, DataQueryArtifact(
            data_responses=data_responses, state=state, eval_attachment=eval_attachment
        )
