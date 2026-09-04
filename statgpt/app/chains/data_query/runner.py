import asyncio

from langchain_core.runnables import Runnable

from statgpt.app.chains.discovery_datasets import DiscoveryDatasetsRunner
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.data_query_outcome import DataQueryMcpPayload
from statgpt.app.schemas.discovery_datasets import DiscoveryDatasetsOutcome
from statgpt.app.schemas.query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas.data_query_tool import DataQueryDetails

from .parameters import DataQueryParameters
from .query_builder.factory import QueryBuilderFactory


class DataQueryRunner:
    """Runs the data query pipeline (and the discovery lookup beside it) for one query.

    Shared by the LangChain and MCP interfaces of the data query tool: it knows nothing about
    either framework and hands back a `DataQueryOutcome` for the caller to render.
    """

    def __init__(self, details: DataQueryDetails, channel_config: ChannelConfig):
        self._details = details
        self._channel_config = channel_config

    async def run(self, inputs: dict, query: str) -> DataQueryOutcome:
        factory = QueryBuilderFactory(self._details)

        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        chain: Runnable = await factory.create_chain(inputs)

        res, discovery = await self._run_with_discovery(chain, inputs, query)
        logger.info(f"DataQueryTool result: {res!r}")

        data_responses: dict[str, DataResponse] = {
            k: v
            for k, v in res.get(ChainParametersConfig.DATA_RESPONSES, {}).items()
            if v is not None
        }
        return DataQueryOutcome(
            response=res[DataQueryParameters.RESPONSE_FIELD],
            data_responses=data_responses,
            state=res.get(DataQueryParameters.STATE, QueryBuilderAgentState()),
            mcp_payload=res.get(DataQueryParameters.MCP_PAYLOAD, DataQueryMcpPayload()),
            eval_attachment=res.get(DataQueryParameters.EVAL_ATTACHMENT, DataQueryEvalAttachment()),
            discovery=discovery,
        )

    async def _run_with_discovery(
        self, chain: Runnable, inputs: dict, query: str
    ) -> tuple[dict, DiscoveryDatasetsOutcome | None]:
        """Run the query pipeline and, when configured, the discovery lookup beside it.

        The lookup shares nothing with the pipeline - it only reads from ``inputs`` - and it
        never raises, so a failure there cannot cost the user their data.
        """
        runner = DiscoveryDatasetsRunner.from_channel_config(self._channel_config)
        if runner is None:
            return await chain.ainvoke(inputs), None

        # Not a TaskGroup: it would wrap a pipeline failure in an ExceptionGroup, changing what
        # the tool raises on a failed data query.
        discovery_task = asyncio.create_task(runner.run(query, inputs))
        try:
            res = await chain.ainvoke(inputs)
        except BaseException:
            discovery_task.cancel()
            # Awaited, not just cancelled: `cancel()` only schedules the CancelledError, and
            # the lookup must not still be writing to the choice as it is torn down.
            await asyncio.gather(discovery_task, return_exceptions=True)
            raise
        return res, await discovery_task
