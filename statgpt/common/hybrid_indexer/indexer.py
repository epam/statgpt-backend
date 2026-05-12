import asyncio
import collections
import json
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import logger
from statgpt.common.data import base
from statgpt.common.data.base import (
    DimensionQuery,
    QueryOperator,
    VirtualDimension,
    VirtualDimensionCategory,
)
from statgpt.common.data.sdmx.common import CodeCategory, SdmxCodeListDimension
from statgpt.common.data.sdmx.common.indicator import ComplexIndicator
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet
from statgpt.common.default_prompts import HybridIndexerDefaultPrompts
from statgpt.common.schemas import HybridSearchConfig
from statgpt.common.settings.hybrid_index import HybridIndexSettings
from statgpt.common.utils.elastic import ElasticIndex
from statgpt.common.utils.models import get_chat_model
from statgpt.common.utils.timer import debug_timer
from statgpt.common.vectorstore import ScoredVectorStoreDocument, VectorStore

from . import schemas

_log = logging.getLogger(__name__)


class Indexer:

    class _Series:
        def __init__(
            self,
            dimension_queries: list[DimensionQuery],
            indicator_name: str,
            indicator_where: list[dict[str, str]],
        ):
            self._dimension_queries = dimension_queries
            self.indicator_name = indicator_name
            self.indicator_where = indicator_where

        def get_id(self) -> str:
            return ".".join(f"{dq.dimension_id}_{dq.values[0]}" for dq in self._dimension_queries)

        def serialize_dimension_queries(self) -> str:
            return json.dumps([{dq.dimension_id: dq.values[0]} for dq in self._dimension_queries])

    @staticmethod
    def _log_prompt(prompt: ChatPromptTemplate, title: str = '') -> None:
        logger.info(f'{title}:\n\n{prompt.pretty_repr()}')

    def __init__(
        self,
        config: HybridSearchConfig,
        models_api_key: str,
        matching_index: ElasticIndex,
        indicators_index: ElasticIndex,
        vectorstore: VectorStore,
        normalize: bool,
        harmonize: bool,
    ):
        self._config = config
        self._models_api_key = models_api_key
        self._settings = HybridIndexSettings()
        self._matching_index = matching_index
        self._indicators_index = indicators_index
        self._vectorstore = vectorstore
        self._is_normalize = normalize
        self._is_harmonize = harmonize

    async def index(
        self,
        dataset: base.DataSet,
        version_id: int,
        version_ids: set[int],
        max_n_indicators: int | None,
        auth_context: AuthContext,
    ) -> dict:
        stats: dict = {}
        if self._is_normalize:
            stats["normalization"] = await self._normalize(
                dataset, version_id, max_n_indicators, auth_context
            )
        if self._is_harmonize:
            version_ids = {version_id}.union(version_ids)
            stats["harmonization"] = await self._harmonize(
                dataset, version_id, version_ids, max_n_indicators
            )
        return stats

    def _create_normalize_llm_chain(self) -> Runnable:
        normalization_prompt = HybridIndexerDefaultPrompts.get_normalize_prompts()
        self._log_prompt(normalization_prompt, title='normalization prompt')

        normalize_llm = get_chat_model(
            api_key=self._models_api_key, model_config=self._config.normalize_model_config
        ).with_structured_output(schema=schemas.NormalizationOutput, method="json_schema")

        llm_chain = normalization_prompt | normalize_llm | (lambda d: d.normalized.lower())

        return RunnablePassthrough.assign(
            _result=self._safe_invoke(llm_chain=llm_chain, label='Normalization')
        ) | RunnablePassthrough.assign(
            normalized=lambda d: d['_result'][0],
            error=lambda d: d['_result'][1],
        )

    def _create_harmonize_llm_chain(self) -> Runnable:
        harmonization_prompt = HybridIndexerDefaultPrompts.get_harmonize_prompts()
        self._log_prompt(harmonization_prompt, title='harmonization prompt')

        harmonize_llm = get_chat_model(
            api_key=self._models_api_key, model_config=self._config.harmonize_model_config
        ).with_structured_output(schema=schemas.HarmonizationOutput, method="json_schema")

        llm_chain = harmonization_prompt | harmonize_llm | (lambda d: d.primary)

        return (
            RunnablePassthrough.assign(
                input=lambda d: d['index'].name_normalized,
                indicators=lambda d: d['indicators_str'],
            )
            | RunnablePassthrough.assign(
                _result=self._safe_invoke(llm_chain=llm_chain, label='Harmonization')
            )
            | RunnablePassthrough.assign(
                primary=lambda d: d['_result'][0],
                normalized=lambda d: d['_result'][0],
                error=lambda d: d['_result'][1],
            )
        )

    @staticmethod
    def _safe_invoke(llm_chain: Runnable, label: str) -> Runnable:
        async def _safe_llm_chain_invoke(inputs: dict) -> tuple[str, str | None]:
            try:
                return await llm_chain.ainvoke(inputs), None
            except Exception as e:
                fallback = inputs.get('input', '').lower()
                _log.exception(f"{label} failed for '{fallback}', using fallback")
                return fallback, str(e)

        return RunnableLambda(_safe_llm_chain_invoke)

    def _matching_index_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(
                series=lambda d: self._create_series(d['dataset'], d['indicator'])
            )
            | RunnablePassthrough.assign(input=lambda d: d['series'].indicator_name)
            | self._create_normalize_llm_chain()
            | RunnablePassthrough.assign(document=self._create_matching_index)
        )

    async def _normalize(
        self,
        dataset: base.DataSet,
        version_id: int,
        max_n_indicators: int | None,
        auth_context: AuthContext,
    ) -> dict:
        indicators = await dataset.get_indicators(auth_context=auth_context, allow_cached=True)
        if max_n_indicators is not None:
            indicators = indicators[:max_n_indicators]

        with debug_timer(f"hybrid.normalizing.{dataset.name}"):
            txt = f"{len(indicators)} indicators for dataset {dataset.name} ({dataset.entity_id})"
            _log.info(f'Normalizing {txt}')
            index_chain = self._matching_index_chain()
            results = await index_chain.abatch(
                inputs=[
                    dict(dataset=dataset, version_id=version_id, indicator=indicator)
                    for indicator in indicators
                ],
                config=RunnableConfig(
                    max_concurrency=self._settings.concurrency_limit,
                ),
            )
            documents, stats = self._collect_results(results, total=len(indicators))
            _log.info(f'Finished normalizing {txt}')

        # push index to elasticsearch & vectorstore
        await self._matching_index.add_bulk(
            documents=(doc.model_dump(mode='json') for doc in documents)
        )
        await self._vectorstore.add_documents(
            [
                Document(page_content=doc.name_normalized, metadata=doc.model_dump(mode='json'))
                for doc in documents
            ],
            dataset_id=dataset.id,
            version_id=version_id,
        )

        return stats

    async def _harmonize(
        self,
        dataset: base.DataSet,
        version_id: int,
        version_ids: set[int],
        max_n_indicators: int | None,
    ) -> dict:
        indexer_config: base.IndexerConfig = dataset.config.indexer
        unpack = indexer_config.indicator.unpack
        super_primary = indexer_config.indicator.super_primary

        cache: dict[str, str] = {}
        matching_items = await self._es_get_all_matching_by(version_id=version_id)

        if max_n_indicators is not None:
            matching_items = matching_items[:max_n_indicators]

        chain = self._create_harmonize_chain(unpack)
        with debug_timer(f"hybrid.harmonizing.{dataset.name}"):
            _log.info(
                f"Harmonizing {len(matching_items)} indicators for dataset {dataset.name} (id={dataset.entity_id})"
            )
            results = await chain.abatch(
                inputs=[
                    dict(
                        dataset=dataset,
                        version_id=version_id,
                        version_ids=version_ids,
                        index=index,
                        unpack=unpack,
                        super_primary=super_primary,
                        cache=cache,
                    )
                    for index in matching_items
                ],
                config=RunnableConfig(
                    max_concurrency=self._settings.concurrency_limit,
                ),
            )
            documents, stats = self._collect_results(results, total=len(matching_items))
            _log.info(
                f"Harmonizing {len(matching_items)} indicators for dataset {dataset.name} ({dataset.entity_id}) completed"
            )

        await self._indicators_index.add_bulk(
            documents=(doc.model_dump(mode='json') for doc in documents)
        )

        return stats

    @staticmethod
    def _collect_results(results: list[dict], total: int) -> tuple[list, dict]:
        documents = [r['document'] for r in results]
        errors = [r['error'] for r in results if r['error'] is not None]
        error_types = dict(collections.Counter(errors))
        if errors:
            _log.warning(f"Failed for {len(errors)} out of {total} indicators")
        return documents, {"total": total, "errors": len(errors), "error_types": error_types}

    def _create_harmonize_chain(self, unpack: bool) -> Runnable:
        if unpack:
            return self._create_harmonize_unpack_chain()
        else:
            return self._create_plain_harmonize_chain()

    def _create_harmonize_unpack_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(match_candidates=self._find_match_candidates)
            | RunnablePassthrough.assign(
                indicators_str=lambda d: self._get_indicators_str(d['match_candidates'])
            )
            | self._create_harmonize_llm_chain()
            | RunnablePassthrough.assign(document=self._create_indicator_index)
        )

    async def _find_match_candidates(self, d: dict) -> list[dict]:
        query = d['index'].name_normalized
        version_ids = d['version_ids']
        max_output = 16
        max_query = 32
        return await self._hybrid_candidates(
            query, version_ids=version_ids, max_output=max_output, max_query=max_query
        )

    def _create_plain_harmonize_chain(self) -> Runnable:
        normalize_llm_chain = self._create_normalize_llm_chain()
        return (
            RunnablePassthrough.assign(
                primary=lambda d: self._get_primary_from_series(
                    dataset=d['dataset'],
                    series_str=d['index'].series,
                    super_primary=d['super_primary'],
                )
            )
            | self._resolve_normalized_primary(normalize_llm_chain)
            | self._save_to_cache
            | RunnablePassthrough.assign(document=self._create_indicator_index)
        )

    @staticmethod
    def _resolve_normalized_primary(llm_chain: Runnable) -> Runnable:
        async def _resolve(d: dict) -> dict:
            primary = d['primary']
            cache = d['cache']
            if primary in cache:
                return {**d, 'normalized': cache[primary], 'error': None}
            return await llm_chain.ainvoke({**d, 'input': primary})

        return RunnableLambda(_resolve)

    @staticmethod
    def _save_to_cache(d: dict) -> dict:
        primary = d['primary']
        cache = d['cache']
        if primary not in cache:
            cache[primary] = d['normalized']
        return d

    @staticmethod
    def _get_indicators_str(indicators: list[dict]) -> str:
        return "\n".join(f" - {ind['metadata']['name_normalized']}" for ind in indicators)

    @classmethod
    def _get_primary_from_series(
        cls, dataset: base.DataSet, series_str: str, super_primary: bool
    ) -> str:
        series = json.loads(series_str)
        # ToDo: better understand what exactly that code does
        if super_primary:
            super_primary_value = cls.__get_primary_from_dimension(dataset, series[0])
            primary1 = cls.__get_primary_from_dimension(dataset, series[1])
            primary2 = cls.__get_primary_from_dimension(dataset, series[2])
            return f"{super_primary_value}, {primary1}, {primary2}"
        return cls.__get_primary_from_dimension(dataset, series[0])

    @staticmethod
    def _resolve_dimension_term(
        dataset: base.DataSet, dimension_id: str, code: str
    ) -> CodeCategory | VirtualDimensionCategory:
        dimension = dataset.dimension(dimension_id)
        item: CodeCategory | VirtualDimensionCategory | None
        if isinstance(dimension, SdmxCodeListDimension):
            item = dimension.code_list[code]
        elif isinstance(dimension, VirtualDimension):
            item = dimension.value
        else:
            raise RuntimeError(
                f"Unsupported dimension type {type(dimension)} for dimension {dimension_id}"
                f" of dataset {dataset.source_id}"
            )
        if item is None:
            raise RuntimeError(
                f"Cannot find code {code} in dimension {dimension_id}"
                f" of dataset {dataset.source_id}"
            )
        return item

    @classmethod
    def __get_primary_from_dimension(cls, dataset: base.DataSet, dimension_dict: dict) -> str:
        dimension_id = next(iter(dimension_dict))
        code = dimension_dict[dimension_id]
        return cls._resolve_dimension_term(dataset, dimension_id, code).name

    @classmethod
    def _min_max(cls, score: float, min: float, max: float) -> float:
        return (score - min) / (max - min)

    @classmethod
    def _sem_teor_min_max(cls, score: float, max: float) -> float:
        return cls._min_max(score, -1, max)

    @classmethod
    def _lex_teor_min_max(cls, score: float, max: float) -> float:
        return cls._min_max(score, 0, max)

    def _convex_combination(self, sem: float, lex: float) -> float:
        alpha = self._config.indexer_alpha
        return alpha * sem + (1 - alpha) * lex

    @staticmethod
    def _create_indicator_index(d: dict) -> schemas.IndicatorIndex:
        # TODO: "indicator" index is better to be renamed to "harmonized" index
        dataset: base.DataSet = d['dataset']
        version_id: int = d['version_id']
        index: schemas.MatchingIndex = d['index']
        primary: str = d['primary']
        primary_normalized: str = d['normalized']

        return schemas.IndicatorIndex(
            id=index.id,
            dataset_id=dataset.entity_id,
            dataset_name=dataset.name,
            version_id=version_id,
            series=index.series,
            name=index.name,
            name_normalized=index.name_normalized,
            where=index.where,
            primary=primary,
            primary_normalized=primary_normalized,
        )

    @staticmethod
    def _create_matching_index(d: dict) -> schemas.MatchingIndex:
        dataset: base.DataSet = d['dataset']
        dataset_id: str = dataset.entity_id
        version_id: int = d['version_id']
        series = d['series']
        normalized: str = d['normalized']

        return schemas.MatchingIndex(
            id=f"{dataset_id} {version_id} {series.get_id()}",
            dataset_id=dataset_id,
            dataset_name=dataset.name,
            version_id=version_id,
            series=series.serialize_dimension_queries(),
            name=series.indicator_name,
            name_normalized=normalized,
            where=series.indicator_where,
        )

    def _create_series(self, dataset: Sdmx21DataSet, indicator: ComplexIndicator) -> _Series:
        name_parts = []
        where = []
        dimension_queries = []

        for indicator_component in indicator.indicators:
            dimension_id = indicator_component.dimension_id
            if dimension_id is None:
                raise RuntimeError(
                    f"Cannot index indicator {indicator.query_id} as it has no dimension_id"
                )

            term_id = indicator_component.entity_id

            dimension_query = DimensionQuery(
                dimension_id=dimension_id,
                values=[term_id],
                operator=QueryOperator.IN,
            )
            dimension_queries.append(dimension_query)

            dimension = dataset.dimension(dimension_id)
            term = self._resolve_dimension_term(dataset, dimension_id, term_id)

            if term_id not in self._config.ignored_term_ids:
                name_parts.append(term.name)

            where.append({dimension.name: term.name})

        indicator_name = ", ".join(name_parts)

        series = self._Series(
            dimension_queries=dimension_queries,
            indicator_name=indicator_name,
            indicator_where=where,
        )
        return series

    async def _lexical(
        self, inputs: str, version_ids: set[int], max_query: int
    ) -> dict[str, float]:
        query = {
            "bool": {
                "must": [{"match": {"name_normalized": {"query": inputs}}}],
                "filter": [{"terms": {"version_id": list(version_ids)}}],
            }
        }
        result = await self._matching_index.search(query=query, size=max_query)
        lex_indexed = {}
        lex_max_score = result.hits.max_score

        for hit in result.hits.hits:
            norm_score = self._lex_teor_min_max(hit.score, lex_max_score) if lex_max_score else 0
            if hit.id not in lex_indexed:
                lex_indexed[hit.id] = norm_score

        return lex_indexed

    async def _semantic_raw(
        self, query: str, max_query: int, version_ids: set[int]
    ) -> list[ScoredVectorStoreDocument]:
        return await self._vectorstore.search_with_similarity_score(
            query, k=max_query, version_ids=version_ids
        )

    def _semantic(self, result: list[ScoredVectorStoreDocument]) -> dict[str, float]:
        sem_indexed = {}
        sem_max_score = result[0].score

        for document in result:
            _id: str = document.metadata['id']
            if _id not in sem_indexed:
                norm_score = self._sem_teor_min_max(document.score, sem_max_score)
                sem_indexed[_id] = norm_score

        return sem_indexed

    def _hybrid_combination(
        self,
        result: list[ScoredVectorStoreDocument],
        lex_indexed: dict[str, float],
        sem_indexed: dict[str, float],
    ) -> dict[str, dict]:
        hybrid = {}

        for document in result:
            _id: str = document.metadata['id']

            if _id not in hybrid:
                sem = sem_indexed[_id]
                lex = lex_indexed.get(_id, 0)
                hybrid[_id] = {
                    'score': self._convex_combination(sem, lex),
                    'metadata': document.metadata,
                }
        return hybrid

    async def _hybrid_candidates(
        self, query: str, version_ids: set[int], max_output=16, max_query=32
    ) -> list[dict]:

        async with asyncio.TaskGroup() as tg:
            lex_task = tg.create_task(self._lexical(query, version_ids, max_query))
            sem_task = tg.create_task(self._semantic_raw(query, max_query, version_ids=version_ids))
        lex_indexed = lex_task.result()
        sem_raw = sem_task.result()
        sem_indexed = self._semantic(sem_raw)
        hybrid = self._hybrid_combination(sem_raw, lex_indexed, sem_indexed)
        hybrid_sorted = sorted(hybrid.items(), key=lambda x: x[1]['score'], reverse=True)

        dataset_dict: dict[str, list[dict]] = collections.defaultdict(list)
        for _id, entry in hybrid_sorted:
            metadata = entry['metadata']
            dataset_id = str(metadata['dataset_id'])
            dataset_dict[dataset_id].append({"id": _id, "metadata": metadata})

        result: list = []
        i = 0
        while len(result) < max_output:
            no_more_candidates = True
            for dataset_id in dataset_dict:
                candidates = dataset_dict[dataset_id]
                if len(candidates) > i:
                    result.append(candidates[i])
                    no_more_candidates = False
            i += 1
            if no_more_candidates:
                break
        return result

    async def _es_get_all_matching_by(
        self, version_id: int, max_query: int = 10000
    ) -> list[schemas.MatchingIndex]:
        """Get all matching index entries for a specified dataset version."""

        query = {
            "term": {"version_id": version_id},
        }
        result = await self._matching_index.search(query=query, size=max_query)

        if result.hits.total.value > max_query:
            # Our indexes contain less than 10 thousand documents, but added a check just in case
            raise RuntimeError(f"Too many documents to export: {result.hits.total.value}")

        return [schemas.MatchingIndex.model_validate(hit.source) for hit in result.hits.hits]
