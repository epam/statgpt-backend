import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable

from common.config import multiline_logger as logger
from common.schemas import DataQueryDetails
from common.utils import async_utils
from common.utils.timer import debug_timer
from statgpt.chains.data_query.query_builder import utils as query_utils
from statgpt.schemas.query_builder import DatasetAvailabilityQueriesType
from statgpt.utils.formatters import (
    DatasetAvailabilityQueryFormatter,
    DatasetAvailabilityQueryFormatterConfig,
)

from .schemas import SearchInput


class DimensionSearchChainFactoryBase(ABC):
    def __init__(self, config: DataQueryDetails):
        self._config = config

    @abstractmethod
    def create(self) -> Runnable:
        pass

    @staticmethod
    async def _get_availability(inputs: dict, queries_key: str) -> DatasetAvailabilityQueriesType:
        search_input = SearchInput(**inputs)
        queries: DatasetAvailabilityQueriesType = inputs[queries_key]
        if len(queries) == 0:
            return {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Starting availability queries for {len(queries)} datasets from {queries_key}"
            )
            # Format queries for detailed logging
            query_formatter = DatasetAvailabilityQueryFormatter(
                config=DatasetAvailabilityQueryFormatterConfig(
                    locale=search_input.data_service.channel_config.locale,
                    add_value_ids=True,
                    header_level=4,
                    add_citation=False,
                ),
                auth_context=search_input.auth_context,
            )

            formatted_queries = await query_formatter.format_queries(
                dataset_queries=queries,
                datasets_dict=search_input.datasets_dict,
            )
            logger.debug(f"Availability queries to run:\n{formatted_queries}")

        tasks = []
        for dataset_id, query in queries.items():
            dataset = search_input.datasets_dict[dataset_id].data
            tasks.append(dataset.availability_query(query, search_input.auth_context))

        logger.debug(f"Running {len(tasks)} availability queries concurrently")
        task_results = await async_utils.gather_with_concurrency(10, *tasks)
        result = {dataset_id: query for dataset_id, query in zip(queries.keys(), task_results)}
        logger.debug(f"Completed {len(result)} availability queries")
        return result

    async def _get_availability_from_strong_queries(
        self, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        return await self._get_availability(inputs, "strong_queries")

    @staticmethod
    def _filter_queries_by_availability(
        inputs: dict, queries_key: str, availability_key: str
    ) -> dict:
        queries: DatasetAvailabilityQueriesType = inputs[queries_key]
        availability: DatasetAvailabilityQueriesType = inputs[availability_key]
        result = {}
        for dataset_id, query in queries.items():
            availability_query = availability.get(dataset_id)
            if not availability_query:
                continue
            result[dataset_id] = query.filter(availability_query)
        return result

    @classmethod
    def _filter_strong_queries_by_strong_availability(
        cls, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        return cls._filter_queries_by_availability(
            inputs, queries_key="strong_queries", availability_key="strong_availability"
        )

    @classmethod
    async def _populate_strong_queries_stage(cls, stage: Stage, inputs: dict):
        with debug_timer("_populate_strong_queries_stage"):
            search_input = SearchInput(**inputs)
            await DatasetAvailabilityQueryFormatter.populate_queries_stage(
                stage=stage,
                queries=search_input.strong_queries,
                auth_context=search_input.auth_context,
                datasets_dict=search_input.datasets_dict,
            )

    def _update_strong_queries_best_attempt_if_possible(self, inputs: dict):
        search_input = SearchInput(**inputs)
        strong_queries = search_input.strong_queries
        if not strong_queries:
            return inputs
        strong_filtered = query_utils.filter_empty_dataset_availability_queries(
            queries=strong_queries
        )
        if strong_filtered:
            inputs['strong_queries_best_nonempty_attempt'] = deepcopy(strong_filtered)
        return inputs
