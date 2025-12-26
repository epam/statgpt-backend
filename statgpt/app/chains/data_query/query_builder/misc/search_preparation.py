from itertools import groupby
from operator import attrgetter

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough

from statgpt.app.chains.utils import dataset_utils
from statgpt.app.default_prompts import data_query_default_prompts
from statgpt.app.schemas.query_builder import (
    ChainState,
    LLMSelectionDimensionCandidate,
    NamedEntitiesResponse,
    NamedEntity,
)
from statgpt.app.utils.callbacks import StageCallback
from statgpt.app.utils.formatters import DatasetFormatterConfig, DatasetsListFormatter
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.sdmx.common import DimensionVirtualCodeCategory
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
    def _add_all_values_to_nonindicator_candidates(
        inputs: dict,
    ) -> list[LLMSelectionDimensionCandidate]:
        """
        Append 'All values' candidates for non-indicator dimensions.
        This is used to allow LLM to select all values for non-indicator dimensions.
        """
        chain_state = ChainState(**inputs)
        dimension_candidates = chain_state.dimension_candidates_for_llm_selection
        datasets_dict = chain_state.datasets_dict
        index = len(dimension_candidates)
        for versioned_ds in datasets_dict.values():
            ds = versioned_ds.data
            dimensions = {dim.entity_id: dim for dim in ds.non_indicator_dimensions()}
            for dim_id, fixed_item in ds.config.dimension_all_values.items():
                if dim_id not in dimensions:
                    # skip indicator dimensions
                    continue
                dimension = dimensions[dim_id]
                # NOTE: we assume there are no such terms already present in dimension_candidates
                dimension_candidates.append(
                    LLMSelectionDimensionCandidate(
                        score=1.0,
                        dataset_id=ds.entity_id,
                        dimension_category=DimensionVirtualCodeCategory(
                            fixed_item=fixed_item,
                            dimension_id=dimension.entity_id,
                            dimension_name=dimension.name,
                            dimension_alias=dimension.alias,
                        ),
                        index=index,
                    )
                )
                index += 1
        return dimension_candidates

    @staticmethod
    def _apply_datasets_selection_response(inputs: dict) -> dict:
        """
        1. Filter datasets by selected IDs
        2. Update normalized query
        """

        chain_state = ChainState(**inputs)
        datasets_selection_response = chain_state.datasets_selection_response
        versioned_datasets_dict = chain_state.versioned_datasets_dict

        if not datasets_selection_response.dataset_ids:
            datasets_selection_response.dataset_ids = list(versioned_datasets_dict)

        inputs['datasets_selection_response'] = datasets_selection_response
        datasets_dict = {
            ds_id: versioned_datasets_dict[ds_id]
            for ds_id in datasets_selection_response.dataset_ids
            if ds_id in versioned_datasets_dict
        }
        inputs['datasets_dict'] = datasets_dict
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
            # detect specified datasets and remove them from normalized query
            | (
                # NOTE: here we overwrite "normalized_query" field
                RunnablePassthrough.assign(
                    datasets_selection_response=self._datasets_selection_chain.create_chain
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
