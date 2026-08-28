import asyncio
from typing import Annotated

from langchain_core.runnables import Runnable
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.discovery_datasets import DiscoveryDatasetsRunner
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.data_query_outcome import DataQueryMcpPayload
from statgpt.app.schemas.discovery_datasets import DiscoveryDatasetsOutcome
from statgpt.app.schemas.query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas.enums import InvocationSource, ToolTypes

from .parameters import DataQueryParameters
from .query_builder.factory import QueryBuilderFactory


class DataQueryArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
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

    async def _arun(self, inputs: dict, query: str) -> tuple[str, DataQueryArtifact]:
        factory = QueryBuilderFactory(self._tool_config.details)

        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        chain: Runnable = await factory.create_chain(inputs)

        res, discovery = await self._run_with_discovery(chain, inputs, query)
        logger.info(f"DataQueryTool result: {res!r}")

        response_str: str = res[DataQueryParameters.RESPONSE_FIELD]
        data_responses: dict[str, DataResponse] = {
            k: v
            for k, v in res.get(ChainParametersConfig.DATA_RESPONSES, {}).items()
            if v is not None
        }
        state: QueryBuilderAgentState = res.get(DataQueryParameters.STATE, QueryBuilderAgentState())
        mcp_payload: DataQueryMcpPayload = res.get(
            DataQueryParameters.MCP_PAYLOAD, DataQueryMcpPayload()
        )
        eval_attachment: DataQueryEvalAttachment = res.get(
            DataQueryParameters.EVAL_ATTACHMENT, DataQueryEvalAttachment()
        )

        discovery_block = discovery.rendered if discovery is not None else None
        if discovery_block and ChainParameters.get_invocation_source(inputs) is (
            InvocationSource.AGENT
        ):
            # The agent reads one string, so the block is appended to it. An MCP client reads a
            # list of content blocks and a structured payload, and gets the block as a block of
            # its own - see the provider - so folding it in here would duplicate it there and
            # leave markdown in the `message` field a widget parses.
            response_str = f"{response_str}\n\n{discovery_block}"

        return response_str, DataQueryArtifact(
            data_responses=data_responses,
            state=state,
            mcp_payload=mcp_payload,
            eval_attachment=eval_attachment,
            discovery_datasets_block=discovery_block,
            discovery_datasets_eval_attachment=(
                discovery.eval_attachment if discovery is not None else None
            ),
        )

    async def _run_with_discovery(
        self, chain: Runnable, inputs: dict, query: str
    ) -> tuple[dict, DiscoveryDatasetsOutcome | None]:
        """Run the query pipeline and, when configured, the discovery lookup beside it.

        The lookup searches a different service on the raw tool argument and shares nothing with
        the pipeline, so running it concurrently costs no wall clock. It only reads from
        ``inputs`` - the pipeline is the sole writer - and it never raises, so a failure there
        cannot cost the user their data.
        """
        runner = DiscoveryDatasetsRunner.from_channel_config(self._channel_config)
        if runner is None:
            return await chain.ainvoke(inputs), None

        # Not a TaskGroup: it would wrap a pipeline failure in an ExceptionGroup, and what the
        # tool raises on a failed data query is not this feature's business to change. Awaiting
        # the lookup only after the pipeline has succeeded gives the same sibling cancellation
        # without touching the exception.
        discovery_task = asyncio.create_task(runner.run(query, inputs))
        try:
            res = await chain.ainvoke(inputs)
        except BaseException:
            discovery_task.cancel()
            # Awaited, not just cancelled: `cancel()` only schedules the CancelledError, so
            # without this the lookup could still be writing to the choice after the tool has
            # raised and the choice is being torn down.
            await asyncio.gather(discovery_task, return_exceptions=True)
            raise
        return res, await discovery_task
