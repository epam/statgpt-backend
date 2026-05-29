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
        description="Concise data query that includes as detailed as possible information on indicators, time frame, "
        "countries, regions and other dimensions. \n\n* Tool works best for single indicator query (e.g. "
        "GDP, inflation), so try to send one query per indicator\n* At the same time tool works very well "
        "with query that includes multiple values for countries, regions and other dimensions (e.g. France "
        "and UK, Baltic countries and Poland)"
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
            " from the Available_Datasets tool response.\n"
            "\n"
            "**Default: None.** Leave unset and let the tool search across all"
            " available datasets. Only populate this when the user names a"
            " specific dataset, provider, or source — never because the query"
            " topic 'sounds like' a dataset's name or description.\n"
            "\n"
            "If you haven't seen Available_Datasets in this conversation, call"
            " it first.\n"
            "\n"
            "## Decision procedure (apply for each candidate dataset)\n"
            "1. Look for an explicit source phrase pointing to this dataset:"
            " 'from / in / according to / using / based on <X>', or the"
            " dataset's id / name / provider as a qualifier ('the EER dataset',"
            " 'IMF's CPI database').\n"
            "2. If NO such phrase exists → leave datasets unset. STOP. Do NOT"
            " consider how closely the dataset's title or description matches"
            " the user's topic. Many datasets are named after the indicators"
            " they contain.\n"
            "3. If YES → include the dataset's source ID and strip that source"
            " phrase from the `query` argument. Any indicator term stays in"
            " `query`.\n"
            "\n"
            "## Rules\n"
            "- Select IDs ONLY from the Available_Datasets response. Never"
            " invent IDs. Unknown source IDs return a descriptive error so you"
            " can correct and retry.\n"
            "- Provider mention (e.g. 'according to IMF', 'from BIS') includes"
            " EVERY dataset from that provider. Partial prefix matches count:"
            " 'IMF' matches 'IMF.STA' and 'IMF.RES'.\n"
            "\n"
            "## Worked examples\n"
            "### datasets=None (indicator / topic only, no explicit source)\n"
            "- 'give me consumer price index'\n"
            "- 'give me quarterly GDP'  (even if a 'Quarterly GDP' dataset"
            " exists)\n"
            "- 'give me inflation and growth indicators from 2010 to 2020'\n"
            "- 'What is the real effective exchange rate index for the United"
            " States, monthly, 2010=100'  (REER is an indicator; user did not"
            " name the EER dataset)\n"
            "- 'How many euros are equivalent to one unit of Panama's domestic"
            " currency for each quarter?'  (exchange-rate indicator phrasing"
            " only, no source phrase)\n"
            "\n"
            "### datasets populated (explicit dataset or provider named)\n"
            "- 'Give me CPI according to CPI dataset' →"
            " datasets=['<CPI source id>'], query='Give me CPI'\n"
            "- 'Give me all data from Consumer Price Index dataset' →"
            " datasets=['<CPI source id>'], query='Give me all data'\n"
            "- 'Real effective exchange rate from the EER dataset, monthly,"
            " USA' → datasets=['<EER source id>'], query='Real effective"
            " exchange rate, monthly, USA'\n"
            "- 'Query main indicators according to IMF' → datasets=[<every"
            " IMF / IMF.STA / IMF.RES source id>], query='Query main"
            " indicators'  (provider mentioned — include ALL of that"
            " provider)."
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
