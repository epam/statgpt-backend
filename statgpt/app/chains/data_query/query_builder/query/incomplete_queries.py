import asyncio
import json
from collections.abc import Iterator

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.data_query_outcome import (
    DimensionValueInfo,
    MissingDimensionInfo,
    MissingDimensionsInfo,
)
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import CategoricalDimension, DataSetAvailabilityQuery, DataSetQuery
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.utils import AttachmentsStorage, MediaTypes, attachments_storage_factory
from statgpt.common.utils.models import get_chat_model


class IncompleteQueriesChain:

    _system_prompt: str

    def __init__(self, llm_model_config: LLMModelConfig, system_prompt: str):
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt

    @classmethod
    async def _attach_custom_table(
        cls,
        attachments_storage: AttachmentsStorage,
        df: pd.DataFrame,
        choice: ChoiceI,
        filename: str,
        title: str,
    ):
        try:
            data_json = json.loads(df.to_json(orient='table'))
            height = min(600, 75 + 27 * df.shape[0])

            result = {
                'data': data_json,
                'metadata': {},
                'layout': {'height': height},
            }

            response = await attachments_storage.put_json(filename, json.dumps(result))
            choice.add_attachment(type=MediaTypes.TTYD_TABLE, title=title, url=response.url)
        except Exception as e:
            logger.exception(f"Failed to attach custom table:\n{e}")

    @staticmethod
    def iter_missing_dimensions(
        dataset: VersionedDataSet,
        query: DataSetQuery,
        availability: DataSetAvailabilityQuery,
    ) -> Iterator[tuple[CategoricalDimension, list[DimensionValueInfo]]]:
        """Yield each required-but-unspecified categorical dimension with its available values.

        A dimension is "missing" when the query carries no filter for it. Only categorical
        dimensions with available values (given the rest of the query) are yielded, so the
        caller can offer the user concrete values to choose from.
        """
        missing_dimensions = [
            d for d in dataset.data.dimensions() if d.entity_id not in query.dimensions_queries_dict
        ]
        for dimension in missing_dimensions:
            if not isinstance(dimension, CategoricalDimension):
                continue
            available_values_query = availability.dimensions_queries_dict.get(dimension.entity_id)
            if available_values_query is None or not (values := available_values_query.values):
                logger.warning(
                    f'There are no available values for dimension "{dimension.name}". '
                    'Can\'t offer available values for user to select from.'
                )
                continue
            entities = {v.query_id: v for v in dimension.available_values}
            value_infos = [
                DimensionValueInfo(
                    id=entities[value_id].query_id,
                    name=entities[value_id].name,
                    description=entities[value_id].description,
                )
                for value_id in values
            ]
            yield dimension, value_infos

    @classmethod
    def build_missing_dimensions_info(
        cls,
        dataset_id: str,
        dataset: VersionedDataSet,
        query: DataSetQuery,
        availability: DataSetAvailabilityQuery,
    ) -> MissingDimensionsInfo:
        """Build the typed missing-dimensions payload for the tool's structured content."""
        dimensions = [
            MissingDimensionInfo(
                dimension_id=dimension.entity_id,
                name=dimension.name,
                available_values=value_infos,
            )
            for dimension, value_infos in cls.iter_missing_dimensions(dataset, query, availability)
        ]
        return MissingDimensionsInfo(dataset_id=dataset_id, dimensions=dimensions)

    async def _add_missing_dimensions_in_attachments(
        self,
        attachments_storage: AttachmentsStorage,
        choice: ChoiceI,
        query: DataSetQuery,
        dataset: VersionedDataSet,
        availability: DataSetAvailabilityQuery,
    ) -> None:
        tasks = []
        for dimension, value_infos in self.iter_missing_dimensions(dataset, query, availability):
            title = f"{dimension.name} ({dimension.entity_id})"
            data = []
            for value in value_infos:
                item = {'ID': value.id, 'Name': value.name}
                if value.description:
                    item['Description'] = value.description
                data.append(item)
            df = pd.DataFrame.from_records(data)
            tasks.append(
                self._attach_custom_table(
                    attachments_storage, df, choice, dimension.get_file_name(), title
                )
            )

        if tasks:
            await asyncio.gather(*tasks)

    async def create_chain(self, inputs: dict, api_key: str) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._system_prompt),
                HumanMessagePromptTemplate.from_template("{query}"),
            ],
        )

        chain = (
            prompt_template
            | get_chat_model(
                api_key=api_key,
                model_config=self._llm_model_config,
            )
            | StrOutputParser()
        )
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )
        target = ChainParameters.get_target(inputs)
        response_content = ''
        async for chunk in chain.astream(inputs):
            target.append_content(chunk)
            response_content += chunk

        dataset_queries = ChainParameters.get_dataset_queries(inputs)
        if len(dataset_queries) > 1 or not dataset_queries:
            logger.exception(f"Expected exactly one dataset query, got {dataset_queries.keys()}")
        else:
            dataset_id, query = dataset_queries.popitem()
            dataset: VersionedDataSet = inputs["datasets_dict"][dataset_id]
            availability = inputs["strong_availability"][dataset_id]
            async with attachments_storage_factory(api_key) as attachments_storage:
                choice = ChainParameters.get_choice(inputs)
                await self._add_missing_dimensions_in_attachments(
                    attachments_storage, choice, query, dataset, availability
                )

        return RunnablePassthrough.assign(
            **{DataQueryParameters.RESPONSE_FIELD: lambda _: response_content},
        )
