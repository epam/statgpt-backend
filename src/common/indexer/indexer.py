import collections
import json
import logging

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough

from common.auth.auth_context import AuthContext
from common.data import base
from common.data.base import DimensionQuery, QueryOperator
from common.prompts import IndexerPrompts
from common.settings.hybrid_index import HybridIndexSettings
from common.utils.elastic import ElasticIndex
from common.utils.models import get_chat_model
from common.utils.timer import debug_timer
from common.vectorstore import VectorStore

from . import schemas

_log = logging.getLogger(__name__)


class Indexer:
    ALPHA = 0.99

    class _Series:
        def __init__(
            self,
            dimension_queries: list[DimensionQuery],
            indicator_name: str,
            indicator_description: str,
            indicator_where: list[dict[str, str]],
        ):
            self._dimension_queries = dimension_queries
            self.indicator_name = indicator_name
            self.indicator_description = indicator_description
            self.indicator_where = indicator_where

        def get_id(self) -> str:
            query_ids = []
            for dimension_query in self._dimension_queries:
                query_ids.append(f"{dimension_query.dimension_id}_{dimension_query.values[0]}")
            return ".".join(query_ids)

        def serialize_dimension_queries(self) -> str:
            query_ids = []
            for dimension_query in self._dimension_queries:
                query_ids.append({dimension_query.dimension_id: dimension_query.values[0]})
            return json.dumps(query_ids)

    def __init__(
        self,
        models_api_key: str,
        matching_index: ElasticIndex,
        indicators_index: ElasticIndex,
        vectorstore: VectorStore,
        normalize: bool,
        harmonize: bool,
    ):
        self._settings = HybridIndexSettings()
        self._matching_index = matching_index
        self._indicators_index = indicators_index
        self._vectorstore = vectorstore
        self._is_normalize = normalize
        self._is_harmonize = harmonize

        normalize_model = get_chat_model(
            api_key=models_api_key, model_config=self._settings.normalize_model_config
        )
        harmonize_model = get_chat_model(
            api_key=models_api_key, model_config=self._settings.harmonize_model_config
        )

        self._normalize_chain = (
            IndexerPrompts.get_normalize_prompts()
            | normalize_model.with_structured_output(method="json_mode")
            | (lambda d: d['normalized'].lower())
        )
        self._harmonize_chain = (
            IndexerPrompts.get_harmonize_prompts()
            | harmonize_model.with_structured_output(method="json_mode")
            | (lambda d: d['primary'])
        )

    async def index(
        self, dataset: base.DataSet, max_n_indicators: int | None, auth_context: AuthContext
    ) -> None:
        if self._is_normalize:
            await self._normalize(dataset, max_n_indicators, auth_context)
        if self._is_harmonize:
            await self._harmonize(dataset, max_n_indicators)

    def _matching_index_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(
                series=lambda d: self._create_series(d['dataset'], d['indicator'])
            )
            | RunnablePassthrough.assign(
                normalized=(
                    RunnablePassthrough.assign(input=lambda d: d['series'].indicator_name)
                    | self._normalize_chain
                )
            )
            | self._create_matching_index
        )

    async def _normalize(
        self, dataset: base.DataSet, max_n_indicators: int | None, auth_context: AuthContext
    ) -> None:
        indicators = await dataset.get_indicators(auth_context=auth_context)
        if max_n_indicators is not None:
            indicators = indicators[0:max_n_indicators]

        with debug_timer(f"hybrid.normalizing.{dataset.name}"):
            _log.info(
                f"Normalizing {len(indicators)} indicators for dataset {dataset.name} (id={dataset.entity_id})"
            )
            index_chain = self._matching_index_chain()
            documents = await index_chain.abatch(
                inputs=[dict(dataset=dataset, indicator=indicator) for indicator in indicators],
                config=RunnableConfig(
                    max_concurrency=self._settings.concurrency_limit,
                ),
            )

            _log.info(
                f"Normalizing {len(documents)} indicators for dataset {dataset.name} ({dataset.entity_id}) completed"
            )

        # push index to elasticsearch & vectorstore
        await self._matching_index.add_bulk(
            documents=(doc.model_dump(mode='json') for doc in documents)
        )
        await self._vectorstore.add_documents(
            [
                Document(page_content=doc.name_normalized, metadata=doc.model_dump(mode='json'))
                for doc in documents
            ],
            dataset_id=int(dataset.entity_id),
        )

    async def _harmonize(self, dataset: base.DataSet, max_n_indicators: int | None) -> None:
        indexer_config: base.IndexerConfig = dataset.config.indexer
        unpack = indexer_config.indicator.unpack
        super_primary = indexer_config.indicator.super_primary

        cache: dict[str, str] = {}
        matching_items = await self._es_get_all_matching_by(dataset.entity_id)

        if max_n_indicators is not None:
            matching_items = matching_items[:max_n_indicators]

        chain = self._create_harmonize_chain(unpack)
        with debug_timer(f"hybrid.harmonizing.{dataset.name}"):
            _log.info(
                f"Harmonizing {len(matching_items)} indicators for dataset {dataset.name} (id={dataset.entity_id})"
            )
            documents = await chain.abatch(
                inputs=[
                    dict(
                        dataset=dataset,
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
            _log.info(
                f"Harmonizing {len(matching_items)} indicators for dataset {dataset.name} ({dataset.entity_id}) completed"
            )

        await self._indicators_index.add_bulk(
            documents=(doc.model_dump(mode='json') for doc in documents)
        )

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
            | RunnablePassthrough.assign(
                primary=(
                    RunnablePassthrough.assign(
                        statement=lambda d: d['index'].name_normalized,
                        indicators=lambda d: d['indicators_str'],
                    )
                    | self._harmonize_chain
                )
            )
            | RunnablePassthrough.assign(
                primary_normalized=lambda d: d[
                    'primary'
                ],  # no need to normalize as extracted from normalized
            )
            | self._create_indicator_index
        )

    async def _find_match_candidates(self, d: dict) -> list[dict]:
        query = d['index'].name_normalized
        max_output = 16
        max_query = 32
        return await self._hybrid_candidates(query, max_output=max_output, max_query=max_query)

    def _create_plain_harmonize_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(
                primary=lambda d: self._get_primary_from_series(
                    dataset=d['dataset'],
                    series_str=d['index'].series,
                    super_primary=d['super_primary'],
                )
            )
            | RunnablePassthrough.assign(
                primary_normalized=lambda d: self._normalized_primary_chain(
                    primary=d['primary'], cache=d['cache']
                )
            )
            | self._save_to_cache
            | self._create_indicator_index
        )

    def _normalized_primary_chain(self, primary: str, cache: dict[str, str]) -> Runnable:
        if primary in cache:
            return RunnableLambda(lambda _: cache[primary])
        else:
            return RunnablePassthrough.assign(input=lambda _: primary) | self._normalize_chain

    @staticmethod
    def _save_to_cache(d: dict) -> dict:
        primary = d['primary']
        normalized = d['primary_normalized']
        cache = d['cache']
        if primary not in cache:
            cache[primary] = normalized
        return d

    @staticmethod
    def _get_indicators_str(indicators: list[dict]) -> str:
        indicators_str = ""
        for indicator in indicators:
            metadata = indicator['metadata']
            indicators_str += f" - {metadata['name_normalized']}\n"
        return indicators_str

    @staticmethod
    def _get_primary_from_series(
        dataset: base.DataSet, series_str: str, super_primary: bool
    ) -> str:
        series = json.loads(series_str)
        _log.debug("getting primary from series: %s", series)
        # ToDo: better understand what exactly that code does
        if super_primary:
            super_primary_value = Indexer.__get_primary_from_dimension(dataset, series[0])
            primary1 = Indexer.__get_primary_from_dimension(dataset, series[1])
            primary2 = Indexer.__get_primary_from_dimension(dataset, series[2])
            return f"{super_primary_value}, {primary1}, {primary2}"
        return Indexer.__get_primary_from_dimension(dataset, series[0])

    @staticmethod
    def __get_primary_from_dimension(dataset: base.DataSet, dimension_dict: dict) -> str:
        dimension_id = list(dimension_dict.keys())[0]
        dimension = dataset.dimension(dimension_id)
        code = dimension_dict[dimension_id]
        item = dimension.code_list[code]  # type: ignore[attr-defined]
        if item is None:
            raise RuntimeError(
                f"Cannot find code {code} in dimension {dimension_id} of dataset {dataset.source_id}"
            )
        return item.name

    @classmethod
    def _min_max(cls, score: float, min: float, max: float) -> float:
        return (score - min) / (max - min)

    @classmethod
    def _sem_teor_min_max(cls, score: float, max: float) -> float:
        return cls._min_max(score, -1, max)

    @classmethod
    def _lex_teor_min_max(cls, score: float, max: float) -> float:
        return cls._min_max(score, 0, max)

    @classmethod
    def _convex_combination(cls, sem, lex):
        return cls.ALPHA * sem + (1 - cls.ALPHA) * lex

    @staticmethod
    def _create_indicator_index(d: dict) -> schemas.IndicatorIndex:
        dataset: base.DataSet = d['dataset']
        index: schemas.MatchingIndex = d['index']
        primary: str = d['primary']
        primary_normalized: str = d['primary_normalized']

        return schemas.IndicatorIndex(
            id=index.id,
            dataset_id=dataset.entity_id,
            dataset_name=dataset.name,
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
        series = d['series']
        normalized: str = d['normalized']

        return schemas.MatchingIndex(
            id=f"{dataset_id} {series.get_id()}",
            dataset_id=dataset_id,
            dataset_name=dataset.name,
            series=series.serialize_dimension_queries(),
            name=series.indicator_name,
            name_normalized=normalized,
            where=series.indicator_where,
        )

    @staticmethod
    def _create_series(dataset: base.DataSet, indicator) -> _Series:
        indexer_config: base.IndexerConfig = dataset.config.indexer

        description = ""
        name = []
        where = []
        dimension_queries = []

        for indicator_component in indicator.indicators:
            dimension_id = indicator_component.dimension_id
            code = indicator_component.entity_id

            dimension_query = DimensionQuery(
                dimension_id=dimension_id,
                values=[code],
                operator=QueryOperator.IN,
            )
            dimension_queries.append(dimension_query)

            dimension = dataset.dimension(dimension_id)
            item = dimension.code_list[code]  # type: ignore[attr-defined]
            name.append(f"{item.name}")
            where.append({dimension.name: item.name})

            if indexer_config.indicator.use_code_list_description and item.description is not None:
                item_description = item.description
                if item_description and len(item_description.strip()) > 0:
                    description += item_description
            if indexer_config.indicator.annotations is not None:
                description_annotation = indexer_config.indicator.annotations.description
                if description_annotation is not None and len(description_annotation.strip()) > 0:
                    description += item.annotation(description_annotation)

        indicator_name = ", ".join(name)

        if not description or len(description) == 0:
            indexer_description = (
                indexer_config.description if indexer_config.description else dataset.description
            )
            description = f"Present in dataset:\n{indexer_description}"
        else:
            description = f"Description:\n{description}"

        series = Indexer._Series(dimension_queries, indicator_name, description, where)
        return series

    async def _lexical(self, input: str, max_query: int) -> dict[str, float]:
        query = {
            "bool": {
                "must": [{"match": {"name_normalized": {"query": input}}}],
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

    async def _semantic_raw(self, query: str, max_query: int) -> list[tuple[Document, float]]:
        return await self._vectorstore.search_with_similarity_score(query, k=max_query)

    async def _semantic(self, result: list[tuple[Document, float]]) -> dict[str, float]:
        sem_indexed = {}
        sem_max_score = result[0][1]

        for doc, score in result:
            _id: str = doc.metadata['id']
            if _id not in sem_indexed:
                norm_score = self._sem_teor_min_max(score, sem_max_score)
                sem_indexed[_id] = norm_score

        return sem_indexed

    def _hybrid_combination(
        self,
        result: list[tuple[Document, float]],
        lex_indexed: dict[str, float],
        sem_indexed: dict[str, float],
    ) -> dict[str, dict]:
        hybrid = {}

        for doc, score in result:
            _id: str = doc.metadata['id']

            if _id not in hybrid:
                sem = sem_indexed[_id]
                lex = 0 if _id not in lex_indexed else lex_indexed[_id]
                hybrid[_id] = {
                    'score': self._convex_combination(sem, lex),
                    'metadata': doc.metadata,
                }
        return hybrid

    async def _hybrid_candidates(self, query: str, max_output=16, max_query=32) -> list[dict]:
        lex_indexed = await self._lexical(query, max_query)
        sem_raw = await self._semantic_raw(query, max_query)
        sem_indexed = await self._semantic(sem_raw)
        hybrid = self._hybrid_combination(sem_raw, lex_indexed, sem_indexed)
        hybrid_sorted = sorted(hybrid.items(), key=lambda x: x[1]['score'], reverse=True)

        dataset_dict: dict[str, list[dict]] = collections.defaultdict(list)
        for _id, _ in hybrid_sorted:
            metadata = hybrid[_id]['metadata']
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
        self, dataset_id: str, max_query: int = 10000
    ) -> list[schemas.MatchingIndex]:
        """Get all matching index entries for a dataset."""

        query = {
            "term": {"dataset_id": dataset_id},
        }
        result = await self._matching_index.search(query=query, size=max_query)

        if result.hits.total.value > max_query:
            # Our indexes contain less than 10 thousand documents, but added a check just in case
            raise RuntimeError(f"Too many documents to export: {result.hits.total.value}")

        return [schemas.MatchingIndex.model_validate(hit.source) for hit in result.hits.hits]
