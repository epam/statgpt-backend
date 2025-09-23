from __future__ import annotations

import asyncio
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from aidial_sdk.chat_completion import Button, FormMetaclass
from aidial_sdk.pydantic_v1 import BaseModel as PydanticV1BaseModel
from aidial_sdk.pydantic_v1 import Field as PydanticV1Field
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, computed_field
from sqlalchemy.ext.asyncio import AsyncSession

import common.models as models
from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import DataSet, DataSourceHandler, DimensionCategory
from common.data.sdmx.common import ComplexIndicator
from common.schemas import ChannelConfig, PreprocessingStatusEnum
from common.schemas.data_query_tool import SpecialDimensionsProcessor
from common.services import (
    ChannelService,
    DataSetService,
    DataSourceService,
    DataSourceTypeService,
    GlossaryOfTermsService,
)
from common.services.base import DbServiceBase
from common.settings.document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    SpecialDimensionValueDocumentMetadataFields,
    VectorStoreMetadataFields,
)
from common.utils.timer import debug_timer
from common.vectorstore import VectorStore, VectorStoreFactory
from statgpt import utils
from statgpt.settings.application import application_settings


@dataclass
class VectorStoreIndicator:
    # NOTE: can use pydantic model here
    document_id: int
    indicator: ComplexIndicator

    def __eq__(self, other):
        if not isinstance(other, VectorStoreIndicator):
            return NotImplemented
        return self.document_id == other.document_id

    def __hash__(self):
        return hash(self.document_id)


class ScoredCandidate(BaseModel, ABC):
    score: float
    dataset_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def query_id(self) -> str:
        pass

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    def candidates_to_llm_string(candidates: list[ScoredCandidate]):
        if not candidates:
            return ''
        # NOTE: we do not pass dataset info to LLM
        df = pd.DataFrame([c.model_dump(include={'query_id', 'name'}) for c in candidates])
        df = df[['query_id', 'name']]  # ensure columns order
        df.rename(columns={'query_id': 'id'}, inplace=True)
        # TODO: str sub below is a fix specific for complex indicators,
        # aiming to make it easier for LLM to rewrite complex indicator ids,
        # origianlly separated with '; '.
        # Once we implement better LLM input format (simple numerical ids),
        # will need to remove this line!
        df['id'] = df['id'].str.replace('; ', '.')
        text = utils.df_2_table_str(df=df, delimiter='|')
        text = text.strip()
        return text

    def __eq__(self, other):
        if not isinstance(other, ScoredCandidate):
            return NotImplemented
        return self.query_id == other.query_id and self.dataset_id == other.dataset_id

    def __hash__(self):
        return hash(f"{self.dataset_id}_{self.query_id}")


class ScoredDimensionCandidate(ScoredCandidate):
    dimension_category: DimensionCategory = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str:
        return self.dimension_category.query_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        return self.dimension_category.name

    @property
    def dimension_id(self) -> str:
        return self.dimension_category.dimension_id

    @property
    def dimension_alias_or_name(self) -> str:
        return self.dimension_category.dimension_alias or self.dimension_category.dimension_name

    def __hash__(self):
        return hash((self.dataset_id, self.dimension_id, self.query_id))

    def __eq__(self, other):
        if not isinstance(other, ScoredDimensionCandidate):
            return NotImplemented
        return (
            self.dataset_id == other.dataset_id
            and self.dimension_id == other.dimension_id
            and self.query_id == other.query_id
        )


class ScoredIndicatorCandidate(ScoredCandidate):
    indicator: ComplexIndicator = Field(exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_id(self) -> str:
        return self.indicator.query_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        return self.indicator.name


class ChannelServiceFacade(DbServiceBase):
    def __init__(self, session: AsyncSession, channel: models.Channel) -> None:
        super().__init__(session, asyncio.Lock())
        self._channel = channel
        self._vector_store_factory: VectorStoreFactory = VectorStoreFactory(
            session=self._session, session_lock=self._session_lock
        )
        self._handler_classes: dict[int, type[DataSourceHandler]] = {}

        # Track this instance for GC debugging
        if application_settings.memory_debug:
            from statgpt.utils.gc_debug import gc_debugger

            gc_debugger.track_object(self, f"ChannelServiceFacade_{id(self)}")
            gc_debugger.track_object(session, f"AsyncSession_{id(session)}")
        self._indicators_vector_store: VectorStore | None = None
        self._dimensions_vector_store: VectorStore | None = None
        self._special_dimensions_vector_store: VectorStore | None = None

    @property
    def channel(self) -> models.Channel:
        return self._channel

    async def _get_indicators_vector_store(self, auth_context: AuthContext) -> VectorStore:
        if self._indicators_vector_store is not None:
            return self._indicators_vector_store
        with debug_timer("chat_facade._get_indicators_vector_store"):
            vector_store = await self._vector_store_factory.get_vector_store(
                collection_name=self._channel.indicator_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        self._indicators_vector_store = vector_store
        return self._indicators_vector_store

    async def _get_dimensions_vector_store(self, auth_context: AuthContext) -> VectorStore:
        if self._dimensions_vector_store is not None:
            return self._dimensions_vector_store
        with debug_timer("chat_facade._get_dimensions_vector_store"):
            vector_store = await self._vector_store_factory.get_vector_store(
                collection_name=self._channel.available_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        self._dimensions_vector_store = vector_store
        return self._dimensions_vector_store

    async def _get_special_dimensions_vector_store(self, auth_context: AuthContext) -> VectorStore:
        if self._special_dimensions_vector_store is not None:
            return self._special_dimensions_vector_store
        with debug_timer("chat_facade._get_special_dimensions_vector_store"):
            vector_store = await self._vector_store_factory.get_vector_store(
                collection_name=self._channel.special_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        self._special_dimensions_vector_store = vector_store
        return self._special_dimensions_vector_store

    @classmethod
    async def get_all_channels(cls, session: AsyncSession) -> list["ChannelServiceFacade"]:
        channels = await ChannelService(session).get_channels_db(limit=None, offset=0)

        return [cls(session=session, channel=item) for item in channels]

    @classmethod
    async def get_channel(cls, session: AsyncSession, deployment_id: str) -> "ChannelServiceFacade":
        channel = await ChannelService(session).get_channel_by_deployment_id(deployment_id)
        return cls(session=session, channel=channel)

    @property
    def deployment_id(self) -> str:
        return self._channel.deployment_id

    @property
    def channel_config(self) -> ChannelConfig:
        return ChannelConfig.model_validate(self._channel.details)

    @property
    def dial_channel_configuration(self) -> dict[str, typing.Any]:
        conversation_starters_config = self.channel_config.conversation_starters
        if conversation_starters_config is None:
            logger.info(
                f"No conversation starters configuration found for channel {self._channel.title}"
            )

            class InitConfiguration(PydanticV1BaseModel, metaclass=FormMetaclass):
                class Config:
                    chat_message_input_disabled = False

            return InitConfiguration.schema()
        logger.info(
            f"Conversation starters configuration found for channel {self._channel.title}, {conversation_starters_config=}"
        )
        buttons = [
            Button(
                const=i,
                submit=True,
                title=button.title,
                populateText=button.text,
            )
            for i, button in enumerate(conversation_starters_config.buttons)
        ]

        class StatGPTConfiguration(PydanticV1BaseModel, metaclass=FormMetaclass):
            class Config:
                chat_message_input_disabled = False

            starter: int | None = PydanticV1Field(
                description=conversation_starters_config.intro_text,
                buttons=buttons,
            )
            timezone: str = PydanticV1Field(
                description="Timezone in IANA format, e.g. 'Europe/Berlin', 'America/New_York'. "
                "Used to interpret and display dates and times.",
                default="UTC",
            )

        return StatGPTConfiguration.schema()

    def get_named_entity_types(self) -> list[str]:
        return self.channel_config.list_named_entity_types()

    def get_country_named_entity_type(self) -> str:
        return self.channel_config.country_named_entity_type.strip()

    async def get_available_terms(self) -> list[models.GlossaryTerm]:
        glossary_service = GlossaryOfTermsService(
            session=self._session, session_lock=self._session_lock
        )
        return await glossary_service.get_term_models_by_channel(
            channel_id=self._channel.id, limit=None, offset=0
        )

    async def _get_indicators_from_documents(
        self, documents: Iterable[Document]
    ) -> list[VectorStoreIndicator]:
        res = []

        data_sources = {
            ds.id: ds
            for ds in await DataSourceService(
                self._session, session_lock=self._session_lock
            ).get_data_sources_models(
                limit=None,
                offset=0,
                ids={
                    doc.metadata[IndicatorDocumentMetadataFields.DATA_SOURCE_ID]
                    for doc in documents
                },
            )
        }

        for doc in documents:
            data_source = data_sources[doc.metadata[IndicatorDocumentMetadataFields.DATA_SOURCE_ID]]
            handler = await self._get_handler_class(data_source.type, config=data_source.details)
            res.append(
                VectorStoreIndicator(
                    document_id=doc.metadata[VectorStoreMetadataFields.DOCUMENT_ID],
                    indicator=await handler.get_indicator_from_document(doc),
                )
            )

        return res

    async def _get_dimension_categories_from_documents(
        self, documents: Iterable[Document]
    ) -> list[DimensionCategory]:
        result = []
        data_sources = {
            ds.id: ds
            for ds in await DataSourceService(
                self._session, session_lock=self._session_lock
            ).get_data_sources_models(
                limit=None,
                offset=0,
                ids={
                    doc.metadata[DimensionValueDocumentMetadataFields.DATA_SOURCE_ID]
                    for doc in documents
                },
            )
        }
        handlers = {
            ds.id: await self._get_handler_class(ds.type, ds.details)
            for ds in data_sources.values()
        }
        for doc in documents:
            handler = handlers[doc.metadata[DimensionValueDocumentMetadataFields.DATA_SOURCE_ID]]
            result.append(await handler.document_to_dimension_category(doc))
        return result

    async def get_indicators_by_ids(
        self, indicator_ids: Iterable[int], auth_context: AuthContext
    ) -> list[VectorStoreIndicator]:
        vector_store = await self._get_indicators_vector_store(auth_context)
        documents = await vector_store.get_documents(ids=indicator_ids)
        return await self._get_indicators_from_documents(documents)

    async def get_ind_id_2_datasets(
        self, indicators: Iterable[VectorStoreIndicator], auth_context: AuthContext
    ) -> dict[int, list[DataSet]]:
        vector_store = await self._get_indicators_vector_store(auth_context)
        res = {}
        # TODO: parallelize this or rewrite the SQL query:
        for indicator in indicators:
            dataset_ids = await vector_store.get_dataset_ids_by_documents_ids(
                ids=[indicator.document_id]
            )
            res[indicator.document_id] = await self.get_datasets_by_ids(str(i) for i in dataset_ids)
        logger.info(f"get_ind_id_2_datasets result: {res}")
        return res

    async def _load_channel_datasets_models(self) -> list[models.DataSet]:
        channel_datasets = await DataSetService(
            self._session, session_lock=self._session_lock
        ).get_channel_dataset_models_with_ds(
            limit=None,
            offset=0,
            channel_id=self._channel.id,
            status=PreprocessingStatusEnum.COMPLETED,
        )
        datasets = [channel_ds.dataset for channel_ds in channel_datasets]
        return datasets

    async def _load_data_sources_models(
        self, datasets: list[models.DataSet]
    ) -> dict[int, models.DataSource]:
        data_sources = {
            ds.id: ds
            for ds in await DataSourceService(
                self._session, session_lock=self._session_lock
            ).get_data_sources_models(limit=None, offset=0, ids={ds.source_id for ds in datasets})
        }
        return data_sources

    async def _load_datasets(
        self, auth_context: AuthContext, filter_available: bool = False
    ) -> list[DataSet]:
        datasets = await self._load_channel_datasets_models()
        data_sources = await self._load_data_sources_models(datasets)

        res = []
        for dataset in datasets:
            data_source = data_sources[dataset.source_id]
            handler = await self._get_handler_class(data_source.type, config=data_source.details)
            if not filter_available or await handler.is_dataset_available(
                dataset.details, auth_context
            ):
                ds = await handler.get_dataset(
                    entity_id=str(dataset.id),
                    title=dataset.title,
                    config=dataset.details,
                    auth_context=auth_context,
                    allow_offline=True,
                    allow_cached=True,
                )
                if ds.status.status == 'online':
                    res.append(ds)

        return res

    async def list_available_datasets(self, auth_context: AuthContext) -> list[DataSet]:
        return await self._load_datasets(auth_context, filter_available=True)

    async def get_dataset_by_source_id(
        self, auth_context: AuthContext, dataset_id: str
    ) -> DataSet | None:
        datasets = await self._load_datasets(auth_context, filter_available=True)
        for ds in datasets:
            if ds.source_id == dataset_id:
                return ds
        return None

    async def group_indicators_by_dataset(
        self, indicators: Iterable[VectorStoreIndicator], auth_context: AuthContext
    ) -> dict[DataSet, set[VectorStoreIndicator]]:
        """
        NOTE: grouping changes order of indicators, since:
        * indicators are grouped by dataset
        * we use dict to store intermediate mapping (ind_id_2_datasets)
        * we use sets to store dataset indicators
        """
        ind_id_2_datasets = await self.get_ind_id_2_datasets(
            indicators=indicators, auth_context=auth_context
        )
        ind_id_2_ind = {ind.document_id: ind for ind in indicators}

        res: dict[DataSet, set[VectorStoreIndicator]] = defaultdict(set)
        for ind_id, datasets in ind_id_2_datasets.items():
            for ds in datasets:
                res[ds].add(ind_id_2_ind[ind_id])

        return res

    async def get_datasets_by_ids(self, dataset_ids: Iterable[str]) -> list[DataSet]:
        db_datasets = await DataSetService(
            self._session, session_lock=self._session_lock
        ).get_datasets_models(limit=None, offset=0, ids=[int(i) for i in dataset_ids])
        logger.info(f"{db_datasets=}")

        data_sources = {
            ds.id: ds
            for ds in await DataSourceService(
                self._session, session_lock=self._session_lock
            ).get_data_sources_models(
                limit=None, offset=0, ids={ds.source_id for ds in db_datasets}
            )
        }
        logger.info(f"{data_sources=}")

        tasks = []
        for dataset in db_datasets:
            data_source = data_sources[dataset.source_id]
            handler = await self._get_handler_class(data_source.type, config=data_source.details)
            tasks.append(
                handler.get_dataset(
                    entity_id=str(dataset.id),
                    title=dataset.title,
                    config=dataset.details,
                    allow_offline=False,
                )
            )

        return [ds for ds in await asyncio.gather(*tasks)]

    async def _get_handler_class(
        self, data_source_type: models.DataSourceType, config: dict
    ) -> DataSourceHandler:
        type_id = data_source_type.id

        if type_id not in self._handler_classes:
            handler_class = await DataSourceTypeService.get_data_source_handler_class(
                data_source_type
            )
            self._handler_classes[type_id] = handler_class

        cls = self._handler_classes[type_id]
        handler_config = cls.parse_config(config)
        return cls(handler_config)

    async def search_dimensions_scored(
        self,
        query: str,
        auth_context: AuthContext,
        k: int = 10,
        datasets: set[str] | None = None,
    ) -> list[ScoredDimensionCandidate]:
        vector_store = await self._get_dimensions_vector_store(auth_context)
        dataset_ids = {int(ds) for ds in datasets} if datasets else None
        with debug_timer("chat_facade.search_dimensions_scored.similarity_search"):
            documents_with_scores_and_ds_id = (
                await vector_store.search_with_similarity_score_and_dataset_id(
                    query, k=k, dataset_ids=dataset_ids
                )
            )
        with debug_timer("search_dimensions_scored.post_process_documents"):
            # NOTE: we assume that documents_with_scores_and_ds_id is a list of tuples
            # (Document, score, dataset_id)
            documents = []
            scores = []
            dataset_ids = []
            for doc, score, ds_id in documents_with_scores_and_ds_id:
                documents.append(doc)
                scores.append(score)
                dataset_ids.append(str(ds_id))
            dimension_categories = await self._get_dimension_categories_from_documents(documents)
            result = []
            for category, score, ds_id in zip(dimension_categories, scores, dataset_ids):
                result.append(
                    ScoredDimensionCandidate(
                        dimension_category=category, score=score, dataset_id=ds_id
                    )
                )
        return result

    async def search_special_dimension_scored(
        self,
        query: str,
        special_dimension_processor: SpecialDimensionsProcessor,
        auth_context: AuthContext,
        datasets: set[str] | None,
        k: int = 10,
    ) -> list[ScoredDimensionCandidate]:
        vector_store = await self._get_special_dimensions_vector_store(auth_context)

        with debug_timer("chat_facade.search_special_dimension_scored.similarity_search"):
            documents_with_scores_and_ds_id = (
                await vector_store.search_with_similarity_score_and_dataset_id(
                    query,
                    k=k,
                    dataset_ids={int(ds) for ds in datasets} if datasets else None,
                    metadata_filters={
                        SpecialDimensionValueDocumentMetadataFields.PROCESSOR_ID: {
                            special_dimension_processor.id
                        }
                    },
                )
            )

        with debug_timer("search_special_dimension_scored.post_process_documents"):
            # NOTE: we assume that documents_with_scores_and_ds_id is a list of tuples
            # (Document, score, dataset_id)
            documents = []
            scores = []
            dataset_ids = []
            for doc, score, ds_id in documents_with_scores_and_ds_id:
                documents.append(doc)
                scores.append(score)
                dataset_ids.append(str(ds_id))
            dimension_categories = await self._get_dimension_categories_from_documents(documents)
            result = []
            for category, score, ds_id in zip(dimension_categories, scores, dataset_ids):
                result.append(
                    ScoredDimensionCandidate(
                        dimension_category=category, score=score, dataset_id=ds_id
                    )
                )
        return result

    async def search_indicators_scored(
        self,
        query: str,
        auth_context: AuthContext,
        k: int = 10,
        datasets: set[str] | None = None,
    ) -> list[ScoredIndicatorCandidate]:
        """TODO: update this method to use new searcher"""

        vector_store = await self._get_indicators_vector_store(auth_context)
        dataset_ids = {int(ds) for ds in datasets} if datasets else None
        documents_with_scores_and_ds_id = (
            await vector_store.search_with_similarity_score_and_dataset_id(
                query, k=k, dataset_ids=dataset_ids
            )
        )
        documents = []
        scores = []
        dataset_ids = []
        for doc, score, ds_id in documents_with_scores_and_ds_id:
            documents.append(doc)
            scores.append(score)
            dataset_ids.append(str(ds_id))
        indicators = await self._get_indicators_from_documents(documents)
        result = []
        for indicator, score, ds_id in zip(indicators, scores, dataset_ids):
            result.append(
                ScoredIndicatorCandidate(
                    indicator=indicator.indicator, score=score, dataset_id=ds_id
                )
            )
        return result
