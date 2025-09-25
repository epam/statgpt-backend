import asyncio
import json

import pandas as pd
from aidial_sdk.chat_completion import Choice
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import multiline_logger as logger
from common.data.base import CategoricalDimension, DataSet, DataSetAvailabilityQuery, DataSetQuery
from common.schemas import LLMModelConfig
from common.utils import AttachmentsStorage, MediaTypes, attachments_storage_factory
from common.utils.models import get_chat_model
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.parameters import ChainParameters


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
        choice: Choice,
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

    async def _add_missing_dimensions_in_attachments(
        self,
        attachments_storage: AttachmentsStorage,
        choice: Choice,
        query: DataSetQuery,
        dataset: DataSet,
        availability: DataSetAvailabilityQuery,
    ) -> None:
        missing_dimensions = [
            d for d in dataset.dimensions() if d.entity_id not in query.dimensions_queries_dict
        ]
        tasks = []
        for dimension in missing_dimensions:
            if isinstance(dimension, CategoricalDimension):
                title = f"{dimension.name} ({dimension.entity_id})"
                entities = {v.query_id: v for v in dimension.available_values}
                data = []
                available_values_query = availability.dimensions_queries_dict.get(
                    dimension.entity_id
                )
                if available_values_query is None or not (values := available_values_query.values):
                    logger.warning(
                        f'There are no available values for dimension "{dimension.name}". '
                        'Can\'t attach table with available values for user to select from.'
                    )
                    continue
                for value_id in values:
                    entity = entities[value_id]
                    item = {'ID': entity.query_id, 'Name': entity.name}
                    if entity.description:
                        item['Description'] = entity.description
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
            dataset = inputs["datasets_dict"][dataset_id]
            availability = inputs["strong_availability"][dataset_id]
            async with attachments_storage_factory(api_key) as attachments_storage:
                choice = ChainParameters.get_choice(inputs)
                await self._add_missing_dimensions_in_attachments(
                    attachments_storage, choice, query, dataset, availability
                )

        return RunnablePassthrough.assign(
            **{DataQueryParameters.RESPONSE_FIELD: lambda _: response_content},
        )
