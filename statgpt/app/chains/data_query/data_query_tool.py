from langchain_core.runnables import Runnable
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas import ToolArtifact, ToolMessageState
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


class DataQueryArgsWithDatasets(ToolArgs):
    """Args schema used when `use_internal_dataset_selection=False`.

    Adds a `datasets` argument the agent populates with explicit source IDs
    taken from the Available_Datasets tool response. Unknown source IDs trigger
    a descriptive error response (see `DataQueryTool._arun`) so the agent can
    correct itself.
    """

    query: str = Field(
        description=(
            "Concise data query covering indicators, time frame, countries / regions"
            " and other dimensions.\n"
            "* Works best for a single indicator per call (GDP, inflation, etc.).\n"
            "* Works very well with multiple values for non-indicator dimensions"
            " (countries, regions).\n"
            "* IMPORTANT — when you also populate the `datasets` argument, remove"
            " every reference to those dataset names / ids / providers from this"
            " `query` string. The downstream pipeline already knows the datasets"
            " from the `datasets` arg; leaving them in `query` biases indicator"
            " search and degrades results. Example: if the user asks"
            " 'OTC derivatives from BIS:WS_OTC_DERIV2(1.0)', set"
            " datasets=['BIS:WS_OTC_DERIV2(1.0)'] and query='OTC derivatives'."
        )
    )
    datasets: list[str] | None = Field(
        default=None,
        description=(
            "Source IDs of datasets the user EXPLICITLY named, taken verbatim"
            " from the Available_Datasets response (call that tool first if you"
            " haven't already in this conversation).\n"
            "\n"
            "Default None. Decide from the user's words,"
            " populate ONLY when the user names a source:"
            " a dataset/provider name, alias, or id that is either preceded by a"
            " source preposition (from, in, according to, using, based on,"
            " per), followed by `dataset`/`database`/`table`, or itself a"
            " source id / agency code (e.g. `IMF.STA:EER`, `IMF`, `BIS`)."
            " Otherwise set None — never infer a dataset from the query's topic"
            " or from how well a dataset's description fits the request.\n"
            "\n"
            "Trap: many datasets are named after the data they hold (Effective"
            " Exchange Rate → EER, Consumer Price Index → CPI). Bare indicator"
            " phrasing ('real effective exchange rate index') is NOT a source;"
            " it counts only with an explicit marker (e.g. 'from the EER"
            " dataset').\n"
            "\n"
            "When you populate this: resolve each source to ids from"
            " Available_Datasets (a provider resolves to ALL its datasets by"
            " prefix — `IMF` covers `IMF.STA`, `IMF.RES`) and strip the source"
            " words from `query`, leaving only indicator/topic terms. Use only"
            " ids present in Available_Datasets; unknown ids return a retriable"
            " error.\n"
            "\n"
            "Examples:\n"
            "- 'give me quarterly GDP' → None (even if a 'Quarterly GDP' dataset"
            " exists)\n"
            "- 'real effective exchange rate index for the US, monthly' → None"
            " (indicator phrasing, no named source).\n"
            "- 'REER from the EER dataset, monthly, USA' →"
            " datasets=['<EER id>'], query='REER, monthly, USA'.\n"
            "- 'main indicators according to IMF' → datasets=[<every IMF.* id>],"
            " query='main indicators'."
        ),
    )


class DataQueryTool(StatGptTool[DataQueryToolConfig], tool_type=ToolTypes.DATA_QUERY):

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[ToolArgs]:
        """Return the schema for the arguments that this tool accepts."""
        if tool_config.details.use_internal_dataset_selection:
            return DataQueryArgs
        return DataQueryArgsWithDatasets

    async def _resolve_agent_datasets(
        self, inputs: dict, source_ids: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Resolve agent-supplied source IDs to entity IDs.

        NOTE: matches against `dataset.source_id` — the current convention
        (URN-with-version for SDMX). See `docs/dataset_source_id_convention.md`.
        Only SDMX datasets are supported in scope; non-SDMX entries will simply
        miss the lookup.

        Returns (resolved_entity_ids, unknown_source_ids, all_available_source_ids).
        """
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        versioned_datasets = await data_service.list_available_datasets(auth_context)

        lookup = {vds.data.source_id: vds.data.entity_id for vds in versioned_datasets}
        all_source_ids = list(lookup.keys())

        resolved: list[str] = []
        unknown: list[str] = []
        for sid in source_ids:
            entity_id = lookup.get(sid)
            if entity_id is None:
                unknown.append(sid)
            else:
                resolved.append(entity_id)
        return resolved, unknown, all_source_ids

    @staticmethod
    def _invalid_datasets_response(unknown: list[str], available: list[str]) -> str:
        """Descriptive error returned to the agent when `datasets` contains unknown IDs."""
        unknown_str = ", ".join(f"`{sid}`" for sid in unknown)
        available_preview = ", ".join(f"`{sid}`" for sid in available[:20])
        more = "" if len(available) <= 20 else f" (and {len(available) - 20} more)"
        return (
            f"The following dataset source IDs were not recognized: {unknown_str}."
            f" Valid source IDs from the Available_Datasets tool response include:"
            f" {available_preview}{more}.\n\n"
            "Please retry this tool call using only source IDs from the"
            " Available_Datasets response, OR omit the `datasets` argument entirely"
            " if no dataset was explicitly mentioned by the user."
        )

    async def _arun(
        self, inputs: dict, query: str, datasets: list[str] | None = None
    ) -> tuple[str, ToolArtifact]:
        # Update the inputs
        inputs[ChainParametersConfig.QUERY] = query

        # Agent-supplied dataset path: validate source IDs against the channel's
        # available datasets and either return a descriptive error or stash the
        # resolved entity IDs for the downstream resolver in search_preparation.
        if not self._tool_config.details.use_internal_dataset_selection:
            resolved_entity_ids: list[str] = []
            if datasets:
                resolved, unknown, available = await self._resolve_agent_datasets(inputs, datasets)
                if unknown:
                    logger.warning(f"DataQueryTool: agent supplied unknown source_ids: {unknown}")
                    return (
                        self._invalid_datasets_response(unknown, available),
                        ToolArtifact(state=ToolMessageState(type=self.tool_type)),
                    )
                resolved_entity_ids = resolved
            inputs[ChainParametersConfig.AGENT_SUPPLIED_DATASET_ENTITY_IDS] = resolved_entity_ids

        factory = QueryBuilderFactory(self._tool_config.details)
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
