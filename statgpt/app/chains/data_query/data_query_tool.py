from typing import Annotated

from langchain_core.runnables import Runnable
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.discovery.fallback import refer_to_discovery
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.data_query_outcome import DataQueryMcpPayload
from statgpt.app.schemas.query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas.enums import ToolTypes

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

        res: dict = await chain.ainvoke(inputs)
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

        if referral := await self._discovery_referral(res, state, query):
            # Appended rather than woven in: the pipeline's own no-data message stays exactly as
            # configured, and a channel with the fallback off produces a byte-identical response.
            response_str = f"{response_str}\n\n{referral}"
            ChainParameters.get_target(res).append_content(f"\n\n{referral}")

        return response_str, DataQueryArtifact(
            data_responses=data_responses,
            state=state,
            mcp_payload=mcp_payload,
            eval_attachment=eval_attachment,
        )

    async def _discovery_referral(
        self, res: dict, state: QueryBuilderAgentState, query: str
    ) -> str:
        """Refer to Grade C discovery datasets when the pipeline found no data.

        Hooked here rather than inside the pipeline because the pipeline reaches its no-data
        conclusion from two distant places, and stamps the outcome on its state either way. One
        hook covers both, and whatever branch is added next.

        The countries come from the run that just failed: named entity recognition already
        extracted them, so the fallback needs no country prompt of its own.
        """
        countries = [
            entity.entity
            for entity in res.get(DataQueryParameters.COUNTRY_NAMED_ENTITIES, [])
            if entity.entity
        ]
        return await refer_to_discovery(
            question=query,
            status=state.status,
            countries=countries,
            config=self._tool_config.details.discovery_fallback,
            data_service=ChainParameters.get_data_service(res),
            auth_context=ChainParameters.get_auth_context(res),
        )
