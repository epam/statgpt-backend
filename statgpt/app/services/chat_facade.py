import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from aidial_sdk.chat_completion.form import Button, FormMetaclass
from aidial_sdk.pydantic.v2 import ConfigDict as DialConfigDict
from aidial_sdk.pydantic.v2 import Field as DialField
from pydantic import BaseModel, ConfigDict, Field, computed_field

import statgpt.common.models as models
import statgpt.common.schemas as schemas
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
from statgpt.common.data.manager import DataManager
from statgpt.common.data.sdmx.common import ComplexIndicator
from statgpt.common.models.database import get_readonly_session_context_manager
from statgpt.common.schemas import ChannelConfig, ChannelDatasetVersion
from statgpt.common.schemas.data_query_tool import SpecialDimensionsProcessor
from statgpt.common.services import (
    ChannelService,
    DataSetService,
    DataSourceService,
    GlossaryOfTermsService,
)
from statgpt.common.settings.application import application_settings
from statgpt.common.settings.dataflow_loader import DataflowLoaderSettings
from statgpt.common.settings.document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    SpecialDimensionValueDocumentMetadataFields,
)
from statgpt.common.utils import AsyncLoadingCache, async_utils
from statgpt.common.utils.timer import debug_timer
from statgpt.common.vectorstore import ScoredVectorStoreDocument, VectorStore, VectorStoreFactory

_log = logging.getLogger(__name__)

_dataflow_loader_settings = DataflowLoaderSettings()

_INDICATORS_TOTAL_TOKEN = "{indicators_total}"


def _substitute_indicators_total(text: str | None, count: int) -> str | None:
    """Replace the supported `{indicators_total}` token with the live count."""
    if text is None:
        return None
    return text.replace(_INDICATORS_TOTAL_TOKEN, str(count))


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
    def convert_candidates_to_llm_string(candidates: list["ScoredCandidate"]):
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
class _DatasetRecord:
    """Lightweight dataset record extracted from an ORM model within a session."""

    id: int
    id_: uuid.UUID
    title: str
    details: dict[str, Any]
    source_id: int

    @classmethod
    def from_model(cls, db: models.DataSet) -> "_DatasetRecord":
        return cls(
            id=db.id,
            id_=db.id_,
            title=db.title,
            details=db.details,
            source_id=db.source_id,
        )


@dataclass
class VersionedDataSet:
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


class DeepResearchChannelConfiguration(BaseChannelConfiguration):
    """Configuration variant that advertises the Deep Research toggle.

    Used only for channels that have the Deep Research tool configured, so the
    frontend renders the control exclusively where the capability is available.
    """

    model_config = DialConfigDict(chat_message_input_disabled=False)

    deep_research: bool = DialField(
        title="Deep research",
        description="Run the request in Deep Research mode.",
        default=False,
    )


class ChannelServiceFacade:
    _indicators_total_cache: AsyncLoadingCache[int] = AsyncLoadingCache(
        ttl=dial_app_settings.indicators_total_cache_ttl
    )

    def __init__(self, channel: schemas.Channel) -> None:
        self._channel = channel
        self._handler_classes: dict[int, type[DataSourceHandler]] = {}

        # Track this instance for GC debugging
        if application_settings.gc_debug:
            from statgpt.common.utils.gc_debug import gc_debugger

            gc_debugger.track_object(self, f"ChannelServiceFacade_{id(self)}")

    @property
    def channel(self) -> schemas.Channel:
        return self._channel

    async def _get_indicators_vector_store(self, auth_context: AuthContext) -> VectorStore:
        with debug_timer("chat_facade._get_indicators_vector_store"):
            vector_store_factory = VectorStoreFactory()
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.indicator_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

    async def _get_non_indicator_dimensions_vector_store(
        self, auth_context: AuthContext
    ) -> VectorStore:
        with debug_timer("chat_facade._get_non_indicator_dimensions_vector_store"):
            vector_store_factory = VectorStoreFactory()
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.non_indicator_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

    async def _get_special_dimensions_vector_store(self, auth_context: AuthContext) -> VectorStore:
        with debug_timer("chat_facade._get_special_dimensions_vector_store"):
            vector_store_factory = VectorStoreFactory()
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=self._channel.special_dimensions_table_name,
                embedding_model_name=self._channel.llm_model,
                auth_context=auth_context,
            )
        return vector_store

    @classmethod
    async def get_channel(cls, deployment_id: str) -> "ChannelServiceFacade":
        async with get_readonly_session_context_manager() as session:
            service = ChannelService(session)
            channel = await service.get_channel_schema_by_deployment_id(deployment_id)
        return cls(channel=channel)

    @property
    def deployment_id(self) -> str:
        return self._channel.deployment_id

    @property
    def channel_config(self) -> ChannelConfig:
        return self._channel.details

    def _is_deep_research_available(self) -> bool:
        """Whether the channel has the Deep Research tool configured and enabled."""
        tool = self.channel_config.deep_research
        return tool is not None and tool.enabled

    async def get_dial_channel_configuration(self, auth_context: AuthContext) -> dict[str, Any]:
        base_configuration_cls: type[BaseChannelConfiguration] = (
            DeepResearchChannelConfiguration
            if self._is_deep_research_available()
            else BaseChannelConfiguration
        )

        conversation_starters_config = self.channel_config.conversation_starters
        if conversation_starters_config is None:
            _log.info(
                f"No conversation starters configuration found for channel {self._channel.title}"
            )

            return base_configuration_cls.model_json_schema()

        _log.info(
            f"Conversation starters configuration found for channel {self._channel.title}, {conversation_starters_config=}"
        )
        intro_text: str | None = conversation_starters_config.intro_text
        title = conversation_starters_config.title
        input_placeholder = conversation_starters_config.input_placeholder

        needs_count = any(
            s is not None and _INDICATORS_TOTAL_TOKEN in s
            for s in (intro_text, title, input_placeholder)
        )
        if needs_count:
            count = await self._get_indicators_total(auth_context)
            intro_text = _substitute_indicators_total(intro_text, count)
            title = _substitute_indicators_total(title, count)
            input_placeholder = _substitute_indicators_total(input_placeholder, count)

        buttons = [
            Button(
                const=i,
                submit=True,
                title=button.title,
                populateText=button.text,
            )
            for i, button in enumerate(conversation_starters_config.buttons)
        ]

        other_fields: dict[str, Any] = {}
        if title is not None:
            other_fields["title"] = title
        if input_placeholder is not None:
            other_fields["json_schema_extra"] = {"statgpt:inputPlaceholder": input_placeholder}

        class StatGPTConfiguration(base_configuration_cls):  # type: ignore[misc,valid-type]
            starter: int | None = DialField(
                default=None, description=intro_text, buttons=buttons, **other_fields
            )

        return StatGPTConfiguration.model_json_schema()

    def get_named_entity_types(self) -> list[str]:
        return self.channel_config.list_named_entity_types()

    def get_country_named_entity_type(self) -> str:
        return self.channel_config.country_named_entity_type.strip()

    async def get_available_terms(self) -> list[schemas.GlossaryTerm]:
        async with get_readonly_session_context_manager() as session:
            service = GlossaryOfTermsService(session, session_lock=None)
            return await service.get_term_schemas_by_channel(
                channel_id=self._channel.id, limit=None, offset=0
            )

    async def _get_indicators_from_documents(
        self, documents: Iterable[ScoredVectorStoreDocument]
    ) -> list[VectorStoreIndicator]:
        doc_list = list(documents)
        if not doc_list:
            return []

        source_ids = {
            doc.metadata[IndicatorDocumentMetadataFields.DATA_SOURCE_ID] for doc in doc_list
        }
        async with get_readonly_session_context_manager() as session:
            source_service = DataSourceService(session, session_lock=None)
            data_sources = await source_service.get_data_sources_schemas(
                limit=None, offset=0, ids=source_ids
            )

        handlers = {}
        for ds in data_sources:
            handlers[ds.id] = await self._get_handler_class(ds.type.id, ds.type.name, ds.details)

        res = []
        for doc in doc_list:
            handler = handlers[doc.metadata[IndicatorDocumentMetadataFields.DATA_SOURCE_ID]]
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
        doc_list = list(documents)
        if not doc_list:
            return []

        source_ids = {
            doc.metadata[DimensionValueDocumentMetadataFields.DATA_SOURCE_ID] for doc in doc_list
        }
        async with get_readonly_session_context_manager() as session:
            source_service = DataSourceService(session, session_lock=None)
            data_sources = await source_service.get_data_sources_schemas(
                limit=None, offset=0, ids=source_ids
            )

        handlers = {}
        for ds in data_sources:
            handlers[ds.id] = await self._get_handler_class(ds.type.id, ds.type.name, ds.details)

        result = []
        for doc in doc_list:
            handler = handlers[doc.metadata[DimensionValueDocumentMetadataFields.DATA_SOURCE_ID]]
            result.append(await handler.document_to_dimension_category(doc))
        return result

    @staticmethod
    def _get_config_for_query(
        dataset_details: dict,
        version: ChannelDatasetVersion,
    ) -> dict:
        """Get config for querying - prefer resolved_config if available."""
        if version.resolved_config is not None:
            return version.resolved_config
        return dataset_details

    async def _load_datasets(self, auth_context: AuthContext) -> list[VersionedDataSet]:
        async with get_readonly_session_context_manager() as session:
            dataset_service = DataSetService(session, session_lock=None)
            data_source_service = DataSourceService(session, session_lock=None)

            last_versions = (
                await dataset_service.get_latest_successful_dataset_versions_for_channel(
                    channel_id=self._channel.id
                )
            )
            versions = {
                k: item.last_completed_version
                for k, item in last_versions.items()
                if item.last_completed_version is not None
            }
            dataset_models = await dataset_service.get_datasets_models(
                limit=None, offset=0, ids=versions.keys()
            )
            datasets = [_DatasetRecord.from_model(db) for db in dataset_models]
            data_sources = {
                ds.id: ds
                for ds in await data_source_service.get_data_sources_schemas(
                    limit=None, offset=0, ids={ds.source_id for ds in datasets}
                )
            }

        async def load_one(dataset: _DatasetRecord) -> VersionedDataSet | None:
            data_source = data_sources[dataset.source_id]
            handler = await self._get_handler_class(
                data_source.type.id, data_source.type.name, data_source.details
            )
            version = versions[dataset.id]
            config_to_use = self._get_config_for_query(dataset.details, version)
            if not await handler.is_dataset_available(config_to_use, auth_context):
                return None
            ds = await handler.get_dataset(
                entity_id=dataset.id_,
                title=dataset.title,
                config=config_to_use,
                auth_context=auth_context,
                allow_offline=True,
                allow_cached=True,
            )
            if ds.status.status != 'online':
                return None
            return VersionedDataSet(version=version, data=ds)

        loaded = await async_utils.gather_with_concurrency(
            _dataflow_loader_settings.dataset_concurrency_limit,
            *(load_one(dataset) for dataset in datasets),
        )
        return [ds for ds in loaded if ds is not None]

    async def list_available_datasets(self, auth_context: AuthContext) -> list[VersionedDataSet]:
        return await self._load_datasets(auth_context)

    async def get_indicator_counts(
        self, auth_context: AuthContext, versioned_datasets: list[VersionedDataSet]
    ) -> dict[str, int]:
        """Get indicator count per dataset `entity_id`."""
        vector_store = await self._get_indicators_vector_store(auth_context)
        version_to_entity = {
            ds.version.version_data_id: ds.data.entity_id for ds in versioned_datasets
        }
        counts_by_version = await vector_store.get_size_per_version(set(version_to_entity.keys()))
        return {
            version_to_entity[vid]: count
            for vid, count in counts_by_version.items()
            if vid in version_to_entity
        }

    async def _get_indicators_total(self, auth_context: AuthContext) -> int:
        """Total indicator count across the channel's latest successful dataset versions.

        Lightweight DB-only count: skips per-handler `is_dataset_available` checks
        (no external SDMX/QuantHub calls). May include indicators from datasets
        that are temporarily offline.

        Result is cached in-process per channel for `indicators_total_cache_ttl`
        seconds. The count is auth-context-independent — concurrent callers share
        a single load via `AsyncLoadingCache`'s per-key lock.
        """

        async def _load() -> int:
            async with get_readonly_session_context_manager() as session:
                dataset_service = DataSetService(session, session_lock=None)
                latest = await dataset_service.get_latest_successful_dataset_versions_for_channel(
                    channel_id=self._channel.id
                )
            version_ids = {
                item.last_completed_version.version_data_id
                for item in latest.values()
                if item.last_completed_version is not None
            }
            if not version_ids:
                return 0
            vector_store = await self._get_indicators_vector_store(auth_context)
            return await vector_store.get_size(version_ids)

        return await self._indicators_total_cache.get(str(self._channel.id), _load)

    async def get_dataset_hierarchy(self, auth_context: AuthContext) -> DatasetHierarchy | None:
        """Get first available dataset hierarchy from the channel data sources."""
        async with get_readonly_session_context_manager() as session:
            data_source_service = DataSourceService(session, session_lock=None)
            data_sources = await data_source_service.get_data_sources_schemas_by(
                channel_id=self._channel.id
            )

        for ds in data_sources:
            handler = await self._get_handler_class(ds.type.id, ds.type.name, ds.details)
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
        self, type_id: int, type_name: str, config: dict
    ) -> DataSourceHandler:
        if type_id not in self._handler_classes:
            self._handler_classes[type_id] = DataManager.get_data_source_handler_class(type_name)

        cls = self._handler_classes[type_id]
        handler_config = cls.parse_config(config)
        return cls(handler_config)

    async def search_non_indicator_dimensions_scored(
        self,
        query: str,
        *,
        auth_context: AuthContext,
        k: int = 10,
        dataset_versions: Iterable[int],
    ) -> list[ScoredDimensionCandidate]:
        vector_store = await self._get_non_indicator_dimensions_vector_store(auth_context)
        version_ids = set(dataset_versions)
        with debug_timer("chat_facade.search_non_indicator_dimensions_scored.similarity_search"):
            documents = await vector_store.search_with_similarity_score(
                query, k=k, version_ids=version_ids
            )

        with debug_timer("search_non_indicator_dimensions_scored.post_process_documents"):
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
