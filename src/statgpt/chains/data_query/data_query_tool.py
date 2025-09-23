from aidial_sdk.chat_completion import Message, Request, Role
from langchain_core.runnables import Runnable
from pydantic import Field

from common.config import multiline_logger as logger
from common.data.base import DataResponse
from common.schemas import DataQueryTool as DataQueryToolConfig
from common.schemas.enums import DataQueryVersion, ToolTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.config import ChainParametersConfig
from statgpt.schemas.query_builder import QueryBuilderAgentState
from statgpt.schemas.tool_artifact import DataQueryArtifact

from .base import BaseDataQueryFactory
from .parameters import DataQueryParameters
from .v2 import QueryBuilderFactoryV2

_IMPLEMENTATIONS: dict[DataQueryVersion, type[BaseDataQueryFactory]] = {
    # DataQueryVersion.v1: QueryBuilderFactory,
    DataQueryVersion.v2: QueryBuilderFactoryV2,
}


class DataQueryArgs(ToolArgs):
    query: str = Field(
        description="Concise data query that includes as detailed as possible information on indicators, time frame, "
        "countries, regions and other dimensions. \n\n* Tool works best for single indicator query (e.g. "
        "GDP, inflation), so try to send one query per indicator\n* At the same time tool works very well "
        "with query that includes multiple values for countries, regions and other dimensions (e.g. France "
        "and UK, Baltic countries and Poland)"
    )


class DataQueryTool(StatGptTool[DataQueryToolConfig], tool_type=ToolTypes.DATA_QUERY):
    @classmethod
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[DataQueryArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DataQueryArgs

    @staticmethod
    def _dummy_request(request: Request, query: str) -> Request:
        """
        Create a dummy request
        consisting only of Agent query for the Data Query tool as a User message
        """

        res = request.copy(exclude={"messages"})
        res.messages = [Message(role=Role.USER, content=query)]
        return res

    async def _arun(self, inputs: dict, query: str) -> tuple[str, DataQueryArtifact]:
        version = self._tool_config.details.version
        implementation = _IMPLEMENTATIONS[version](self._tool_config.details, self._channel_config)

        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        # currently, this request seems not to be used anywhere.
        # all sub-chains of Data Query tool should use QUERY tool input parameter to access history.
        inputs[ChainParametersConfig.REQUEST] = self._dummy_request(
            ChainParameters.get_request(inputs), query
        )

        chain: Runnable = await implementation.create_chain(inputs)

        res: dict = await chain.ainvoke(inputs)
        logger.info(f"DataQueryTool result: {res!r}")

        response_str: str = res[DataQueryParameters.RESPONSE_FIELD]
        data_responses: dict[str, DataResponse] = {
            k: v
            for k, v in res.get(ChainParametersConfig.DATA_RESPONSES, {}).items()
            if v is not None
        }
        state: QueryBuilderAgentState = res.get(DataQueryParameters.STATE, QueryBuilderAgentState())

        return response_str, DataQueryArtifact(data_responses=data_responses, state=state)
