import json

from aidial_sdk.chat_completion import Message, Request, Role
from langchain_core.runnables import Runnable
from pydantic import Field

from common.config import multiline_logger as logger
from common.data.base import DataResponse
from common.schemas import DataQueryTool as DataQueryToolConfig
from common.schemas.enums import DataQueryVersion, ToolTypes
from common.utils import MediaTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.config import ChainParametersConfig
from statgpt.schemas.query_builder import QueryBuilderAgentState
from statgpt.schemas.tool_artifact import DataQueryArtifact
from statgpt.utils.request_context import RequestContext

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

    def _get_attachments_metadata(self, data_response: DataResponse) -> list[dict]:
        attachments = []

        config = self._tool_config.details.attachments

        if config.custom_table:
            attachments.append(
                {
                    "title": data_response.enrich_attachment_name(config.custom_table.name),
                    "type": MediaTypes.MARKDOWN,
                }
            )

        if config.plotly_grid:
            attachments.append(
                {
                    "title": data_response.enrich_attachment_name(config.plotly_grid.name),
                    "type": MediaTypes.PLOTLY,
                }
            )

        if config.csv_file:
            attachments.append(
                {
                    "title": data_response.enrich_attachment_name(config.csv_file.name),
                    "type": MediaTypes.CSV,
                }
            )

        if config.plotly_graphs.enabled:
            for title, _ in data_response.get_plotly_graphs_with_names(config.plotly_graphs.name):
                attachments.append({"title": title, "type": MediaTypes.PLOTLY})

        return attachments

    def _extend_response(self, response: str, data_responses: dict[str, DataResponse]) -> str:
        custom_table_config = self._tool_config.details.attachments.custom_table

        for dataset_id, data_response in data_responses.items():
            attachments = self._get_attachments_metadata(data_response)
            if attachments:
                response += f"\n\n# Metadata of attached files:\n\n{json.dumps(attachments)}"

            if custom_table_config.enabled:
                dataframe_md = data_response.visual_dataframe.to_markdown()
                attachment_name = data_response.enrich_attachment_name(custom_table_config.name)
                response += (
                    f'\n\n# Content of the attachment "{attachment_name}":\n\n{dataframe_md}'
                )

        return response

    async def _arun(self, inputs: dict, query: str) -> tuple[str, DataQueryArtifact]:
        target = ChainParameters.get_target(inputs)
        target.append_name(f": {query}")

        version = self._tool_config.details.version
        implementation = _IMPLEMENTATIONS[version](self._tool_config.details, self._channel_config)

        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        # currently, this request seems not to be used anywhere.
        # all sub-chains of Data Query tool should use QUERY tool input parameter to access history.
        inputs[ChainParametersConfig.REQUEST] = self._dummy_request(
            ChainParameters.get_request(inputs), query
        )

        auth_context = ChainParameters.get_auth_context(inputs)
        request_context = RequestContext(api_key=auth_context.api_key, inputs=inputs)
        chain: Runnable = await implementation.create_chain(request_context)

        res: dict = await chain.ainvoke(inputs)
        logger.info(f"DataQueryTool result: {res!r}")

        response_str: str = res[DataQueryParameters.RESPONSE_FIELD]
        data_responses: dict[str, DataResponse] = {
            k: v
            for k, v in res.get(ChainParametersConfig.DATA_RESPONSES, {}).items()
            if v is not None
        }
        state: QueryBuilderAgentState = res.get(DataQueryParameters.STATE, QueryBuilderAgentState())

        if data_responses:
            response_str = self._extend_response(response_str, data_responses)

        return response_str, DataQueryArtifact(data_responses=data_responses, state=state)
