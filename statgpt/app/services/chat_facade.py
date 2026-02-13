from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from aidial_sdk.chat_completion.form import Button, FormMetaclass
from aidial_sdk.pydantic.v2 import ConfigDict as DialConfigDict
from aidial_sdk.pydantic.v2 import Field as DialField
from pydantic import BaseModel, ConfigDict, Field, computed_field
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
from statgpt.app import utils
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import (
    BaseIndicator,
    DataSet,
    DatasetHierarchy,
    DataSourceHandler,
    DimensionCategory,
)
from statgpt.common.data.sdmx.common import ComplexIndicator
from statgpt.common.data.sdmx.common.config import UrnReference
from statgpt.common.schemas import ChannelConfig, ChannelDatasetVersion
from statgpt.common.schemas.data_query_tool import SpecialDimensionsProcessor
from statgpt.common.services import (
    ChannelService,
    DataSetService,
    DataSourceService,
    DataSourceTypeService,
    GlossaryOfTermsService,
)
from statgpt.common.services.base import DbServiceBase
from statgpt.common.settings.application import application_settings
from statgpt.common.settings.document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    SpecialDimensionValueDocumentMetadataFields,
)
from statgpt.common.utils.timer import debug_timer
from statgpt.common.vectorstore import ScoredVectorStoreDocument, VectorStore, VectorStoreFactory

_log = logging.getLogger(__name__)


@dataclass
class VectorStoreIndicator:
    # NOTE: can use pydantic model here
    document: ScoredVectorStoreDocument
    indicator: BaseIndicator

    @property
    def document_id(self) -> int:
        return self.document.document_id

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
    def convert_candidates_to_llm_string(candidates: list[ScoredCandidate]):
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


@dataclass
class VersionedDataSet:
    model: models.DataSet
    version: ChannelDatasetVersion
    data: DataSet


class BaseChannelConfiguration(BaseModel, metaclass=FormMetaclass):
    model_config = DialConfigDict(chat_message_input_disabled=False)

    timezone: str = DialField(
        description="Timezone in IANA format, e.g. 'Europe/Berlin', 'America/New_York'. "
        "Used to interpret and display dates and times.",
        default="UTC",
    )
    enable_debug_attachments: bool = DialField(
        description="Enable debug attachments in the chat responses.",
        default=dial_app_settings.dial_show_debug_attachments,
    )


class ChannelServiceFacade(DbServiceBase):
    def __init__(self, session: AsyncSession, channel: models.Channel) -> None:
        super().__init__(session, asyncio.Lock())
        self._channel = channel
        self._handler_classes: dict[int, type[DataSourceHandler]] = {}

        # Track this instance for GC debugging
        if application_settings.gc_debug:
            from statgpt.common.utils.gc_debug import gc_debugger

            gc_debugger.track_object(self, f"ChannelServiceFacade_{id(self)}")

    @property
    def channel(self) -> models.Channel:
        return self._channel

    async def _get_indicators_vector_store(self, auth_context: AuthContext) -> VectorStore:
        with debug_timer("chat_facade._get_indicators_vector_store"):
            vector_store_factory = VectorStoreFactory(
                session=self._session, session_lock=self._session_lock
            )
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.indicator_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

    async def _get_dimensions_vector_store(self, auth_context: AuthContext) -> VectorStore:
        with debug_timer("chat_facade._get_dimensions_vector_store"):
            vector_store_factory = VectorStoreFactory(
                session=self._session, session_lock=self._session_lock
            )
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.available_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

    async def _get_special_dimensions_vector_store(self, auth_context: AuthContext) -> VectorStore:
        with debug_timer("chat_facade._get_special_dimensions_vector_store"):
            vector_store_factory = VectorStoreFactory(
                session=self._session, session_lock=self._session_lock
            )
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.special_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

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
    def dial_channel_configuration(self) -> dict[str, Any]:
        conversation_starters_config = self.channel_config.conversation_starters
        if conversation_starters_config is None:
            _log.info(
                f"No conversation starters configuration found for channel {self._channel.title}"
            )

            return BaseChannelConfiguration.model_json_schema()
        intro_text: str = conversation_starters_config.intro_text
        _log.info(
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

        class StatGPTConfiguration(BaseChannelConfiguration):
            starter: int | None = DialField(
                default=None,
                description=intro_text,
                buttons=buttons,
            )

        return StatGPTConfiguration.model_json_schema()

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
        self, documents: Iterable[ScoredVectorStoreDocument]
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
                    document=doc,
                    indicator=await handler.get_indicator_from_document(doc),  # type: ignore
                )
            )

        return res

    async def _get_dimension_categories_from_documents(
        self, documents: Iterable[ScoredVectorStoreDocument]
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

    @staticmethod
    def _get_config_for_query(
        db_dataset: models.DataSet,
        version: ChannelDatasetVersion,
    ) -> dict:
        """Get config for querying - prefer resolved_config if available."""
        if version.resolved_config is not None:
            return version.resolved_config
        return db_dataset.details

    async def _load_datasets(self, auth_context: AuthContext) -> list[VersionedDataSet]:
        dataset_service = DataSetService(self._session, session_lock=self._session_lock)
        data_source_service = DataSourceService(self._session, session_lock=self._session_lock)

        last_versions = await dataset_service.get_latest_successful_dataset_versions_for_channel(
            channel_id=self._channel.id
        )
        versions = {
            k: item.last_completed_version
            for k, item in last_versions.items()
            if item.last_completed_version is not None
        }
        dataset_models = await dataset_service.get_datasets_models(
            limit=None, offset=0, ids=versions.keys()
        )
        data_sources = {
            ds.id: ds
            for ds in await data_source_service.get_data_sources_models(
                limit=None, offset=0, ids={ds.source_id for ds in dataset_models}
            )
        }

        res = []
        for db_dataset in dataset_models:
            data_source = data_sources[db_dataset.source_id]
            handler = await self._get_handler_class(data_source.type, config=data_source.details)
            config_to_use = self._get_config_for_query(db_dataset, versions[db_dataset.id])
            if await handler.is_dataset_available(config_to_use, auth_context):
                ds = await handler.get_dataset(
                    entity_id=db_dataset.id_,
                    title=db_dataset.title,
                    config=config_to_use,
                    auth_context=auth_context,
                    allow_offline=True,
                    allow_cached=True,
                )
                if ds.status.status == 'online':
                    res.append(
                        VersionedDataSet(model=db_dataset, version=versions[db_dataset.id], data=ds)
                    )

        return res

    async def list_available_datasets(self, auth_context: AuthContext) -> list[VersionedDataSet]:
        return await self._load_datasets(auth_context)

    async def get_dataset_by_urn(
        self, version: str, agency_id: str, resource_id: str, auth_context: AuthContext
    ) -> VersionedDataSet | None:
        dataset_service = DataSetService(self._session, session_lock=self._session_lock)
        data_source_service = DataSourceService(self._session, session_lock=self._session_lock)

        dataset_urn = UrnReference(agency_id=agency_id, resource_id=resource_id, version=version)
        last_versions = await dataset_service.get_latest_successful_dataset_versions_for_channel(
            channel_id=self._channel.id
        )
        versions = {
            k: item.last_completed_version
            for k, item in last_versions.items()
            if item.last_completed_version is not None
        }
        dataset_models = await dataset_service.get_datasets_models(
            limit=None, offset=0, ids=versions.keys()
        )
        data_sources = {
            ds.id: ds
            for ds in await data_source_service.get_data_sources_models(
                limit=None, offset=0, ids={ds.source_id for ds in dataset_models}
            )
        }

        res = None
        for db_dataset in dataset_models:
            data_source = data_sources[db_dataset.source_id]
            handler = await self._get_handler_class(data_source.type, config=data_source.details)
            config_to_use = self._get_config_for_query(db_dataset, versions[db_dataset.id])
            db_dataset_urn = UrnReference.model_validate(db_dataset.details['urn'])
            if (
                db_dataset_urn.short_urn() == dataset_urn.short_urn()
            ) and await handler.is_dataset_available(config_to_use, auth_context):
                ds = await handler.get_dataset(
                    entity_id=db_dataset.id_,
                    title=db_dataset.title,
                    config=config_to_use,
                    auth_context=auth_context,
                    allow_offline=True,
                    allow_cached=True,
                )
                if ds.status.status == 'online':
                    res = VersionedDataSet(
                        model=db_dataset, version=versions[db_dataset.id], data=ds
                    )

        return res

    async def get_dataset_hierarchy(self, auth_context: AuthContext) -> DatasetHierarchy | None:
        """Get first available dataset hierarchy from the channel data sources."""

        data_source_service = DataSourceService(self._session, session_lock=self._session_lock)
        data_sources = await data_source_service.get_data_sources_models_by(
            channel_id=self._channel.id
        )

        for source in data_sources:
            handler = await self._get_handler_class(source.type, config=source.details)
            hierarchy = await handler.get_dataset_hierarchy(auth_context=auth_context)
            if hierarchy is not None:
                return hierarchy
        return None

    async def get_dataset_by_source_id(
        self, auth_context: AuthContext, dataset_id: str
    ) -> DataSet | None:
        datasets = await self._load_datasets(auth_context)
        for ds in datasets:
            if ds.data.source_id == dataset_id:
                return ds.data
        return None

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
        *,
        auth_context: AuthContext,
        k: int = 10,
        dataset_versions: Iterable[int],
    ) -> list[ScoredDimensionCandidate]:
        vector_store = await self._get_dimensions_vector_store(auth_context)
        version_ids = set(dataset_versions)
        with debug_timer("chat_facade.search_dimensions_scored.similarity_search"):
            documents = await vector_store.search_with_similarity_score(
                query, k=k, version_ids=version_ids
            )

        with debug_timer("search_dimensions_scored.post_process_documents"):
            dimension_categories = await self._get_dimension_categories_from_documents(documents)
            result = []
            for doc, category in zip(documents, dimension_categories):
                result.append(
                    ScoredDimensionCandidate(
                        dimension_category=category, score=doc.score, dataset_id=str(doc.dataset_id)
                    )
                )
        return result

    async def search_special_dimension_scored(
        self,
        query: str,
        *,
        special_dimension_processor: SpecialDimensionsProcessor,
        auth_context: AuthContext,
        k: int = 10,
        version_ids: Iterable[int],
    ) -> list[ScoredDimensionCandidate]:
        vector_store = await self._get_special_dimensions_vector_store(auth_context)

        with debug_timer("chat_facade.search_special_dimension_scored.similarity_search"):
            documents = await vector_store.search_with_similarity_score(
                query,
                k=k,
                version_ids=set(version_ids),
                metadata_filters={
                    SpecialDimensionValueDocumentMetadataFields.PROCESSOR_ID: {
                        special_dimension_processor.id
                    }
                },
            )

        with debug_timer("search_special_dimension_scored.post_process_documents"):
            dimension_categories = await self._get_dimension_categories_from_documents(documents)
            result = []
            for doc, category in zip(documents, dimension_categories):
                result.append(
                    ScoredDimensionCandidate(
                        dimension_category=category, score=doc.score, dataset_id=str(doc.dataset_id)
                    )
                )
        return result

    async def search_indicators_scored(
        self,
        query: str,
        *,
        auth_context: AuthContext,
        k: int = 10,
        dataset_versions: Iterable[int],
    ) -> list[ScoredIndicatorCandidate]:
        """TODO: update this method to use new searcher"""

        vector_store = await self._get_indicators_vector_store(auth_context)
        version_ids = set(dataset_versions)
        documents = await vector_store.search_with_similarity_score(
            query, k=k, version_ids=version_ids
        )
        indicators = await self._get_indicators_from_documents(documents)
        result = []
        for indicator in indicators:
            result.append(
                ScoredIndicatorCandidate(
                    indicator=indicator.indicator,  # TODO: fix ScoredIndicatorCandidate
                    score=indicator.document.score,
                    dataset_id=str(indicator.document.dataset_id),
                )
            )
        return result
