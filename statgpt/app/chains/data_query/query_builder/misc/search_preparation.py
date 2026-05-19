from itertools import groupby
from operator import attrgetter

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough

from statgpt.app.chains.utils import dataset_utils
from statgpt.app.config import ChainParametersConfig
from statgpt.app.default_prompts import data_query_default_prompts
from statgpt.app.schemas.query_builder import (
    ChainState,
    DataSetsSelectionChainResponse,
    NamedEntitiesResponse,
    NamedEntity,
)
from statgpt.app.utils.callbacks import StageCallback
from statgpt.app.utils.formatters import DatasetFormatterConfig, DatasetsListFormatter
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import DataQueryDetails
from statgpt.common.schemas.data_query_tool import DataQueryPrompts

from .datasets_selection import DataSetsSelectionChain
from .datetime_chain import DateTimeDimensionChain
from .named_entities import NamedEntitiesChain
from .normalization import NormalizationChain


class SearchPreparationChainFactory:
    def __init__(self, config: DataQueryDetails):
        self._config = config

        prompts: DataQueryPrompts = self._config.prompts

        self._datetime_chain = DateTimeDimensionChain(
            llm_model_config=self._config.llm_models.time_period_model_config,
            system_prompt=prompts.datetime_prompt or data_query_default_prompts.datetime_prompt,
        )
        # self._group_expander_chain = GroupExpanderChain(
        #     llm_model_config=self._config.llm_models.group_expander_model_config,
        #     system_prompt=prompts.group_expander_prompt
        #     or data_query_default_prompts.group_expander_prompt,
        #     fallback_prompt=prompts.group_expander_fallback_prompt
        #     or data_query_default_prompts.group_expander_fallback_prompt,
        # )
        self._normalization_chain = NormalizationChain(
            llm_model_config=self._config.llm_models.query_normalization_model_config,
            system_prompt=prompts.normalization_prompt
            or data_query_default_prompts.normalization_prompt,
        )
        self._named_entities_chain = NamedEntitiesChain(
            llm_model_config=self._config.llm_models.named_entities_model_config,
            system_prompt=prompts.named_entities_prompt
            or data_query_default_prompts.named_entities_prompt,
        )

        self._use_internal_dataset_selection = config.use_internal_dataset_selection
        self._datasets_selection_chain: DataSetsSelectionChain | None = None
        if self._use_internal_dataset_selection:
            self._datasets_selection_chain = DataSetsSelectionChain(
                llm_model_config=self._config.llm_models.datasets_selection_model_config,
                system_user_prompt=prompts.dataset_selection_prompt
                or data_query_default_prompts.dataset_selection_prompt,
            )

    @staticmethod
    def _get_country_named_entities(inputs: dict) -> list[NamedEntity]:
        chain_state = ChainState(**inputs)
        country_named_entity_type = chain_state.data_service.get_country_named_entity_type()
        named_entities_response = chain_state.named_entities_response.entities
        country_entities = [
            ne
            for ne in named_entities_response
            if country_named_entity_type.lower().startswith(ne.entity_type.lower())
            # ToDo: remove this temporary workaround by allowing to define hints or descriptions for named entity types
        ]
        logger.info(
            f'Found {len(country_entities)} {country_named_entity_type} named entities: {country_entities}'
        )
        return country_entities

    @staticmethod
    def _resolve_agent_supplied_datasets(inputs: dict) -> dict:
        """Build a `datasets_selection_response` from the agent-supplied list.

        The DataQueryTool has already validated agent-supplied source IDs against
        the channel's available datasets (returning a descriptive error if any
        were unknown) and stashed the resolved entity IDs in `inputs` under
        AGENT_SUPPLIED_DATASET_ENTITY_IDS. This step just packages them into the
        same `DataSetsSelectionChainResponse` shape the LLM chain would have
        produced, so the downstream `_apply_datasets_selection_response` works
        unchanged. `rewritten_query` is the raw query — the agent is instructed
        via the `query` arg description to strip dataset references itself.
        """
        entity_ids: list[str] = inputs.get(
            ChainParametersConfig.AGENT_SUPPLIED_DATASET_ENTITY_IDS, []
        )
        inputs["datasets_selection_response"] = DataSetsSelectionChainResponse(
            dataset_ids=list(entity_ids),
            rewritten_query=inputs.get("normalized_query", ""),
        )
        return inputs

    @staticmethod
    def _apply_datasets_selection_response(inputs: dict) -> dict:
        """
        1. Apply datasets filter
        2. Remove datasets filter from normalized query
        """

        chain_state = ChainState(**inputs)
        datasets_selection_response = chain_state.datasets_selection_response
        versioned_datasets_dict = chain_state.versioned_datasets_dict

        if not (selected_datasets := datasets_selection_response.dataset_ids):
            # no datasets filter detected
            inputs['datasets_dict'] = versioned_datasets_dict
        else:
            # selected_datasets should already have hallucinations removed,
            # but let's use defensive programming and check again
            inputs['datasets_dict'] = {
                ds_id: versioned_datasets_dict[ds_id]
                for ds_id in selected_datasets
                if ds_id in versioned_datasets_dict
            }

        inputs['normalized_query'] = datasets_selection_response.rewritten_query

        return inputs

    async def _populate_normalization(self, stage: Stage, inputs: dict):
        normalized_query = inputs.get("normalized_query", "")
        if normalized_query:
            stage.append_content(f"Normalized Query: `{normalized_query}`\n")

    async def _populate_datetime(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        datetime_json = chain_state.date_time_query_response.model_dump_json(indent=2)
        stage.append_content(f"Date Time Query:\n```json\n{datetime_json}\n```\n")

    async def _populate_named_entities(self, stage: Stage, inputs: dict):
        named_entities_response = inputs.get("named_entities_response", NamedEntitiesResponse())
        if not named_entities_response:
            return

        entities = sorted(named_entities_response.entities, key=attrgetter("entity_type", "entity"))
        for k, g in groupby(entities, key=attrgetter("entity_type")):
            entities_str = ", ".join(f"**{entity.entity}**" for entity in g)
            stage.append_content(f"* _{k}_: " + entities_str + '\n')

    async def _populate_datasets_dict(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        channel_config = chain_state.data_service.channel_config

        formatter = DatasetsListFormatter(
            DatasetFormatterConfig(
                locale=channel_config.locale,
                citation=None,
                use_description=False,
            ),
            chain_state.auth_context,
        )
        content = await formatter.format([d.data for d in chain_state.datasets_dict.values()])

        stage.append_content(content)

    def create(self) -> Runnable:
        normalizing_stage_name = "Normalizing Query"
        normalizing_query_stage_callback = StageCallback(
            stage_name=normalizing_stage_name,
            content_appender=self._populate_normalization,
            debug_only=self._config.stages_config.is_stage_debug(normalizing_stage_name),
        )

        named_entities_stage_name = "Extracting Named Entities"
        named_entities_stage_callback = StageCallback(
            stage_name=named_entities_stage_name,
            content_appender=self._populate_named_entities,
            debug_only=self._config.stages_config.is_stage_debug(named_entities_stage_name),
        )

        if self._use_internal_dataset_selection:
            datasets_selection_chain = self._datasets_selection_chain
            assert datasets_selection_chain is not None
            datasets_step: Runnable = (
                RunnablePassthrough.assign(
                    datasets_selection_response=datasets_selection_chain.create_chain
                )
                | self._apply_datasets_selection_response
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Selecting Datasets", self._populate_datasets_dict, debug_only=True
                        ),
                        StageCallback(
                            "Normalized Query with Datasets Removed",
                            self._populate_normalization,
                            debug_only=True,
                        ),
                    ]
                )
            )
        else:
            datasets_step = (
                RunnableLambda(self._resolve_agent_supplied_datasets)
                | self._apply_datasets_selection_response
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Selecting Datasets (agent-supplied)",
                            self._populate_datasets_dict,
                            debug_only=True,
                        ),
                    ]
                )
            )

        chain = (
            RunnablePassthrough.assign(
                versioned_datasets_dict=dataset_utils.get_available_datasets,
            )
            # # unpack country groups in the user prompt
            # | RunnablePassthrough.assign(
            #     query_with_expanded_groups=self._group_expander_chain.create_chain,
            # )
            # normalize (summarize) conversation
            | RunnablePassthrough.assign(
                normalized_query=self._normalization_chain.create_chain,
            ).with_config(config=RunnableConfig(callbacks=[normalizing_query_stage_callback]))
            # save 'normalized_query' to separate variable, since it will be overwritten later
            | RunnablePassthrough.assign(normalized_query_raw=lambda d: d["normalized_query"])
            | datasets_step
            # extract named entities and time range
            | RunnablePassthrough.assign(
                named_entities_response=self._named_entities_chain.create_chain,
                date_time_query_response=self._datetime_chain.create_chain,
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        named_entities_stage_callback,
                        StageCallback(
                            "Extracting Time Range", self._populate_datetime, debug_only=True
                        ),
                    ]
                )
            )
            | RunnablePassthrough.assign(
                country_named_entities=self._get_country_named_entities,
            )
        )

        return chain
