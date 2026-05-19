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
            "Source IDs of datasets the user explicitly mentioned, taken verbatim"
            " from the Available_Datasets tool response. Leave unset when the user"
            " did not name a dataset — the tool will then query all available"
            " datasets. When you do populate this, also strip those dataset"
            " references from the `query` argument (see `query` description)."
            " Unknown source IDs return a descriptive error so you can retry.\n"
            "\n"
            "If you have not already seen the Available_Datasets response in this"
            " conversation, call the Available_Datasets tool first to obtain the"
            " list of valid source IDs before populating this argument.\n"
            "\n"
            "## Selection rules\n"
            "- Select datasets ONLY from the Available_Datasets list. Never invent IDs.\n"
            "- Select only datasets the user EXPLICITLY mentions (by name / id /"
            " provider / description). If unsure, leave this unset.\n"
            "- User may refer to a dataset by any metadata (name, id, provider,"
            " description).\n"
            "- Some datasets are NAMED LIKE the indicators they contain. The user"
            " asking for such an indicator is NOT a sufficient condition to pick"
            " the same-named dataset. Pick the dataset only if the user explicitly"
            " asks for it.\n"
            "- If the user asks for BOTH an indicator AND an identically-named"
            " dataset: (1) include the dataset here, (2) remove the dataset"
            " reference from `query`, (3) keep the indicator term in `query`.\n"
            "- Distinguish indicators vs datasets via grammar — datasets usually"
            " follow words like 'from', 'according to'.\n"
            "- If the user names a PROVIDER only (e.g. 'according to IMF'), include"
            " ALL datasets from that provider. Partial provider matches count:"
            " 'IMF' should also match 'IMF.STA' and 'IMF.RES'.\n"
            "\n"
            "## Worked examples\n"
            "1. 'give me consumer price index' → datasets=None, query unchanged"
            " (indicator only, no dataset specified).\n"
            "2. 'Give me CPI according to CPI dataset' → datasets=['<CPI dataset"
            " source_id>'], query='Give me CPI' (both indicator and dataset; strip"
            " the dataset reference).\n"
            "3. 'Give me all data from Consumer Price Index dataset' →"
            " datasets=['<CPI dataset source_id>'], query='Give me all data'"
            " (dataset only, no indicator).\n"
            "4. 'give me quarterly GDP' → datasets=None, query unchanged"
            " (indicator only — even if a 'Quarterly GDP' dataset exists, the user"
            " didn't ASK for the dataset).\n"
            "5. 'give me inflation and growth indicators from 2010 to 2020' →"
            " datasets=None (indicators only).\n"
            "6. 'Query main indicators according to IMF' → datasets=[<every IMF /"
            " IMF.STA / IMF.RES dataset source_id>], query='Query main indicators'"
            " (provider mentioned — include ALL of that provider).\n"
            "7. 'Consumer Price Index (CPI)' → datasets=None (bare indicator, no"
            " dataset ask).\n"
            "8. 'provide data on harmonized cpi for Finland' → datasets=None"
            " (indicator only)."
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
