import collections
import json
import re
import time
import typing as t
from collections.abc import Generator
from typing import Any, NamedTuple

from pydantic import BaseModel

from statgpt.app.default_prompts import hybrid_search_default_prompts
from statgpt.app.schemas.query_builder import (
    DatasetAvailabilityQueriesType,
    DateTimeQueryResponse,
    HybridMatchTimings,
    HybridSearchTimings,
    NamedEntity,
)
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.dial_stages import BufferedStagesManager, ContentStageI, StageI
from statgpt.common.config.logging import logger
from statgpt.common.data.base import DimensionQuery, QueryOperator
from statgpt.common.hybrid_indexer.schemas import IndicatorIndex, MatchingIndex
from statgpt.common.schemas import HybridSearchConfig
from statgpt.common.utils import async_utils
from statgpt.common.utils.elastic import ElasticIndex, SearchResult
from statgpt.common.utils.models import get_chat_model
from statgpt.common.vectorstore import ScoredVectorStoreDocument, VectorStore

DatasetDimTermsSetType: t.TypeAlias = dict[str, dict[str, set[str]]]
""" dataset_id -> dimension_id -> set of allowed/selected terms (values) """


class PlainItemScored(BaseModel):
    score: float
    metadata: MatchingIndex


class HarmonizedItemScored(BaseModel):
    score: float
    metadata: IndicatorIndex


class Primary(BaseModel):
    primary: str
    indicators: list[dict]


PlainItemsScoredDict: t.TypeAlias = dict[str, PlainItemScored]
HarmonizedItemsScoredDict: t.TypeAlias = dict[str, HarmonizedItemScored]


class SearchParams(NamedTuple):
    alpha: float

    max_candidates: int
    """The maximum number of candidates returned by combining the results of semantic and lexical search."""

    max_semantic_candidates: int
    max_lexical_candidates: int


class HybridCandidate(BaseModel):
    # NOTE: can use IndicatorIndex model instead
    id: str
    dataset_id: str
    primary: str
    name: str
    name_original: str
    where: list[dict[str, str]]
    series: list[dict[str, str]]


class HybridCandidateScored(HybridCandidate):
    score: int


class HybridSearchResultInner(BaseModel):
    lexical: HarmonizedItemsScoredDict
    semantic: PlainItemsScoredDict
    llm_scored: list[HybridCandidateScored]
    final: DatasetDimTermsSetType
    timings: HybridMatchTimings


class HybridSearchResult(BaseModel):
    lexical: list[dict]
    semantic: list[dict]
    llm_scored: list[HybridCandidateScored]
    final_queries: dict[str, list[DimensionQuery]]
    timings: HybridSearchTimings


class HybridSearcher:
    RE_HIGHLIGHT_MATCH_1 = re.compile(r"(\(?<em>.+</em>\)*,?)")

    class HybridMatch:

        def __init__(self, outer):
            self._outer = outer
            self._hybrid = None
            self._max_component = None
            self._candidates = None
            self._candidates_primary_components = None

        def _add_llm_scores_to_indexed(
            self, indexed: dict, scores: list[dict[str, int]]
        ) -> list[HybridCandidateScored]:
            result = []
            for score_dict in scores:
                id_, score = list(score_dict.items())[0]
                if id_ == '0':
                    # best from previous batches - skip
                    continue
                item = HybridCandidateScored(**indexed[id_], score=int(score))
                result.append(item)
            return result

        async def search(
            self,
            stage: ContentStageI,
            query: str,
            version_ids: set[int],
            availability: DatasetDimTermsSetType,
        ) -> tuple[
            HarmonizedItemsScoredDict,
            PlainItemsScoredDict,
            list[HybridCandidateScored],
            list[dict],
            str,
            HybridMatchTimings,
        ]:
            timings = HybridMatchTimings(query=query)
            t_total = time.perf_counter()

            t_query_planner = time.perf_counter()
            search_params = await self._query_planner(stage, query, version_ids)
            timings.query_planner = time.perf_counter() - t_query_planner

            reasoning = "\n        relevance score: 1 - 3 (1 - lowest, 3 - highest)\n"
            lexical, semantic, candidates = await self._hybrid_candidates(
                query, version_ids, availability, search_params, timings
            )

            batches, indexed = self._prepare_for_relevance(candidates)

            dataset_max_score: dict[str, float] = {}
            if stage:
                stage.append_content(reasoning)

            is_only_one_dataset_available = len(availability) == 1
            llm_scored = []
            selected: list[dict] = []
            t_relevance_total = time.perf_counter()
            for items in batches:
                items = self._pre_append_confirmed(items, selected)

                t_relevance_batch = time.perf_counter()
                relevance = await self._relevance_candidates(query, items)
                timings.relevance_per_batch.append(time.perf_counter() - t_relevance_batch)

                llm_scored += self._add_llm_scores_to_indexed(indexed=indexed, scores=relevance)

                batch_selected, batch_reasoning = self._filter_candidates(
                    stage, indexed, relevance, dataset_max_score, is_only_one_dataset_available
                )
                selected += batch_selected
                reasoning += batch_reasoning
            timings.relevance_total = time.perf_counter() - t_relevance_total
            timings.total = time.perf_counter() - t_total
            return lexical, semantic, llm_scored, selected, reasoning, timings

        async def _query_planner(
            self, stage: ContentStageI, query: str, version_ids: set[int]
        ) -> SearchParams:
            primaries, total, candidates, good_candidates = await self._outer.lexical_pre_match(
                query=query,
                highlight_field="primary_normalized",
                version_ids=version_ids,
                max_candidates=self._outer.config.max_lexical_pre_match_candidates,
            )
            if len(good_candidates) > 0:
                # re-calculate numbers based on good candidates
                primaries, total, _, _ = await self._outer.lexical_pre_match(
                    query=" ".join(good_candidates),
                    highlight_field="primary_normalized",
                    version_ids=version_ids,
                    max_candidates=self._outer.config.max_lexical_pre_match_candidates,
                )

            if stage:
                good_candidates_str = "[" + "]  [".join(good_candidates) + "]"
                stage.append_content(
                    f"\n    > [planner] full-text candidates - total: {total}, primary: {primaries}, candidates: {good_candidates_str}\n"
                )

            res: SearchParams
            if len(good_candidates) == 0:
                # fallback to semantic
                res = SearchParams(
                    alpha=self._outer.config.fallback_alpha,
                    max_candidates=2 * self._outer.config.max_candidates,
                    max_lexical_candidates=2 * self._outer.config.max_lexical_candidates,
                    max_semantic_candidates=2 * self._outer.config.max_semantic_candidates,
                )
            else:
                # go with hybrid
                if primaries > self._outer.config.max_candidates:
                    res = SearchParams(
                        alpha=self._outer.config.hybrid_alpha,
                        max_candidates=2 * self._outer.config.max_candidates,
                        max_lexical_candidates=2 * self._outer.config.max_lexical_candidates,
                        max_semantic_candidates=2 * self._outer.config.max_semantic_candidates,
                    )
                else:
                    res = SearchParams(
                        alpha=self._outer.config.default_alpha,
                        max_candidates=self._outer.config.max_candidates,
                        max_lexical_candidates=self._outer.config.max_lexical_candidates,
                        max_semantic_candidates=self._outer.config.max_semantic_candidates,
                    )

            if stage:
                stage.append_content(f"    >  =>  {res!r}\n")
            return res

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
        def _convex_combination(cls, sem: float, lex: float, alpha: float) -> float:
            return alpha * sem + (1 - alpha) * lex

        async def _relevance_candidates(self, query: str, items):
            items_str = self._format_relevance_items(items)
            output = await self._outer._relevancy_chain.ainvoke(
                {"statement": query.lower(), "items": items_str}
            )
            return output['relevance']

        async def _lexical(
            self, user_query: str, version_ids: set[int], max_query: int
        ) -> HarmonizedItemsScoredDict:
            query = {
                "bool": {
                    "must": [
                        {"match": {"primary_normalized": {"query": user_query}}},
                    ],
                    "should": [
                        {"match": {"name_normalized": {"query": user_query, "boost": 0.3}}},
                    ],
                    "filter": [{"terms": {"version_id": list(version_ids)}}],
                }
            }
            result: SearchResult = await self._outer._indicators_index.search(
                query=query, size=max_query
            )

            items_dict = {}
            lex_max_score = result.hits.max_score
            for hit in result.hits.hits:
                item = IndicatorIndex.model_validate(hit.source)
                if item.id not in items_dict:
                    norm_score = (
                        self._lex_teor_min_max(hit.score, lex_max_score) if lex_max_score else 0
                    )
                    items_dict[item.id] = HarmonizedItemScored(score=norm_score, metadata=item)

            return items_dict

        async def _es_get_by_id(self, _id: str) -> IndicatorIndex | None:
            query = {"term": {"id.keyword": _id}}
            result: SearchResult = await self._outer._indicators_index.search(query=query, size=1)
            if len(result.hits.hits) == 0:
                logger.warning(f"HybridMatch: cannot find by id {_id}")
                return None
            res_raw = result.hits.hits[0].source
            res = IndicatorIndex.model_validate(res_raw)
            return res

        async def _semantic_raw(
            self, user_query: str, version_ids: set[int], max_query: int
        ) -> list[ScoredVectorStoreDocument]:
            return await self._outer._vectorstore.search_with_similarity_score(
                user_query, k=max_query, version_ids=version_ids
            )

        def _semantic_result(self, result: list[ScoredVectorStoreDocument]) -> PlainItemsScoredDict:
            items_dict = {}
            sem_max_score = result[0].score

            for document in result:
                item = MatchingIndex.model_validate(document.metadata)
                if item.id not in items_dict:
                    norm_score = self._sem_teor_min_max(document.score, sem_max_score)
                    items_dict[item.id] = PlainItemScored(score=norm_score, metadata=item)

            return items_dict

        async def _hybrid_combination(
            self,
            sem_raw: list[ScoredVectorStoreDocument],
            lexical: HarmonizedItemsScoredDict,
            semantic: PlainItemsScoredDict,
            alpha: float,
        ) -> HarmonizedItemsScoredDict:
            hybrid = {}

            for document in sem_raw:
                _id = document.metadata['id']

                if _id not in semantic:
                    continue  # filtered out by availability

                sem_score = semantic[_id].score
                lex_score = 0 if _id not in lexical else lexical[_id].score

                metadata: IndicatorIndex | None
                if _id in lexical:
                    metadata = lexical[_id].metadata
                else:
                    metadata = await self._es_get_by_id(_id)

                if metadata is None:
                    continue
                if _id not in hybrid:
                    score = self._convex_combination(sem_score, lex_score, alpha)
                    hybrid[_id] = HarmonizedItemScored(score=score, metadata=metadata)

            return hybrid

        @staticmethod
        def _filer_candidates_by_availability(
            candidates: PlainItemsScoredDict | HarmonizedItemsScoredDict,
            availability: DatasetDimTermsSetType,
        ) -> dict:
            filtered = {}

            for _id, data in candidates.items():
                metadata = data.metadata
                dataset_id = metadata.dataset_id
                if dataset_id not in availability:
                    continue

                series: list[dict[str, str]] = json.loads(metadata.series)
                matched = True
                for series_dict in series:
                    dimension_id = list(series_dict.keys())[0]
                    value = series_dict[dimension_id]
                    if dimension_id not in availability[dataset_id]:
                        matched = False
                        break
                    if value not in availability[dataset_id][dimension_id]:
                        matched = False
                        break
                if matched:
                    filtered[_id] = data

            return filtered

        async def _hybrid_candidates(
            self,
            query: str,
            version_ids: set[int],
            availability: DatasetDimTermsSetType,
            search_params: SearchParams,
            timings: HybridMatchTimings,
        ) -> tuple[HarmonizedItemsScoredDict, PlainItemsScoredDict, list[dict]]:

            t_total = time.perf_counter()

            t_lexical = time.perf_counter()
            lex = await self._lexical(
                query, version_ids, max_query=search_params.max_lexical_candidates
            )
            timings.lexical = time.perf_counter() - t_lexical
            lex_filtered: HarmonizedItemsScoredDict = self._filer_candidates_by_availability(
                lex, availability
            )

            t_semantic_raw = time.perf_counter()
            sem_raw: list[ScoredVectorStoreDocument] = await self._semantic_raw(
                query, version_ids, max_query=search_params.max_semantic_candidates
            )
            timings.semantic_raw = time.perf_counter() - t_semantic_raw
            sem_indexed = self._semantic_result(sem_raw)
            sem_filtered: PlainItemsScoredDict = self._filer_candidates_by_availability(
                sem_indexed, availability
            )

            t_hybrid_combination = time.perf_counter()
            hybrid = await self._hybrid_combination(
                sem_raw=sem_raw,
                lexical=lex_filtered,
                semantic=sem_filtered,
                alpha=search_params.alpha,
            )
            timings.hybrid_combination = time.perf_counter() - t_hybrid_combination

            hybrid_sorted: list[tuple[str, HarmonizedItemScored]] = sorted(
                hybrid.items(), key=lambda x: x[1].score, reverse=True
            )
            dataset2primaries = self._get_dataset_2_primaries(
                hybrid_sorted, max_output=search_params.max_candidates
            )
            candidates: list[dict] = self._get_dataset_indicators_round_robin(
                dataset2primaries, max_output=search_params.max_candidates
            )
            candidates = self.make_sure_top_n_is_included(
                candidates, hybrid_sorted, max_output=search_params.max_candidates
            )
            candidates = self._include_lexical_only_candidates(
                candidates,
                lex_filtered,
                max_lexical_only=self._outer.config.max_lexical_only_candidates,
            )

            timings.hybrid_candidates_total = time.perf_counter() - t_total
            return lex_filtered, sem_filtered, candidates

        def _include_lexical_only_candidates(
            self,
            candidates: list[dict],
            lex_filtered: HarmonizedItemsScoredDict,
            max_lexical_only: int,
        ) -> list[dict]:
            """Force the top lexical-only candidates into the LLM relevance set.

            Fusion (`_hybrid_combination`) is anchored on the semantic results: it iterates the
            semantic candidates and drops any id missing from them. A strong keyword match that
            falls outside the semantic top-k therefore never enters the candidate set, never
            reaches the LLM relevance judge, and can never be selected — a pure recall loss on
            rare/coded indicators with weak embeddings. This rescues up to ``max_lexical_only`` such
            candidates (already availability-filtered, ranked by lexical score) that are not already
            present, appending them on top of the diversified hybrid candidates so the judge can
            still score them.
            """
            if max_lexical_only <= 0:
                return candidates

            existing_ids = {c['id'] for c in candidates}
            lexical_sorted = sorted(lex_filtered.items(), key=lambda x: x[1].score, reverse=True)

            added = 0
            for _id, data in lexical_sorted:
                if added >= max_lexical_only:
                    break
                if _id in existing_ids:
                    continue
                candidates.append({"id": _id, "metadata": data.metadata.model_dump()})
                existing_ids.add(_id)
                added += 1
            return candidates

        def make_sure_top_n_is_included(
            self,
            candidates: list[dict],
            hybrid_sorted: list[tuple[str, HarmonizedItemScored]],
            max_output: int,
        ) -> list[dict]:
            max_top_n = int(max_output / self._outer.config.max_output_div)
            top_n = {k: v.metadata for k, v in hybrid_sorted[0:max_top_n]}

            candidates_ids = {c['id'] for c in candidates}
            top_n = {k: v for k, v in top_n.items() if k not in candidates_ids}

            for _id, metadata in top_n.items():
                metadata_dict = metadata.model_dump()
                i = max_output - 1
                while i >= 0:
                    candidate = candidates[i]
                    if candidate["metadata"]['primary_normalized'] == metadata.primary_normalized:
                        candidates.insert(i, {"id": _id, "metadata": metadata_dict})
                        break
                    i -= 1
                else:
                    candidates.append({"id": _id, "metadata": metadata_dict})
            return candidates[0:max_output]

        def _get_dataset_2_primaries(
            self, hybrid_sorted: list[tuple[str, HarmonizedItemScored]], max_output: int
        ) -> dict[str, list[Primary]]:
            max_primary = max_output - int(max_output / self._outer.config.max_output_div)

            primaries_set: set[str] = set()
            dataset2primaries: dict[str, list[Primary]] = {}
            for _id, data in hybrid_sorted:
                metadata = data.metadata

                if metadata.dataset_id in dataset2primaries:
                    continue

                cur_ind_dict = {"id": _id, "metadata": metadata.model_dump()}

                dataset2primaries[metadata.dataset_id] = [
                    Primary(primary=metadata.primary_normalized, indicators=[cur_ind_dict])
                ]

            for _id, data in hybrid_sorted:
                metadata = data.metadata

                if len(dataset2primaries[metadata.dataset_id]) >= max_output - len(
                    dataset2primaries
                ):
                    continue

                cur_ind_dict = {"id": _id, "metadata": metadata.model_dump()}

                primary = metadata.primary_normalized
                found = False
                for p in dataset2primaries[metadata.dataset_id]:
                    if p.primary == primary:
                        if p.indicators[0]['id'] != _id:
                            p.indicators.append(cur_ind_dict)
                        found = True
                        break

                should_create_new_primary = (
                    primary in primaries_set or len(primaries_set) < max_primary
                )
                if not found and should_create_new_primary:
                    primaries_set.add(primary)
                    dataset2primaries[metadata.dataset_id].append(
                        Primary(primary=primary, indicators=[cur_ind_dict])
                    )
            return dataset2primaries

        def _get_dataset_indicators_round_robin(
            self, dataset2primaries: dict[str, list[Primary]], max_output: int
        ) -> list[dict]:
            generator_dict = {
                dataset_id: self._hybrid_candidates_dataset_generator(primaries)
                for dataset_id, primaries in dataset2primaries.items()
            }
            result = []
            have_more_candidates = True
            while have_more_candidates:
                have_more_candidates = False
                for dataset_id in dataset2primaries.keys():
                    try:
                        candidate: dict = next(generator_dict[dataset_id])
                        result.append(candidate)
                        have_more_candidates = True

                        if len(result) >= max_output:
                            return result
                    except StopIteration:
                        pass
            return result

        @staticmethod
        def _hybrid_candidates_dataset_generator(
            primaries: list[Primary],
        ) -> Generator[dict, None, None]:
            # ~~~ Simplified Example: ~~~
            # [{"indicators": [101, 102]}, {"indicators": [201, 202, 203]}, {"indicators": [301]}]
            # -> [101, 201, 301, 102, 202, 203]
            i = 0
            have_more_candidates = True
            while have_more_candidates:
                have_more_candidates = False
                for p in primaries:
                    if len(p.indicators) > i:
                        have_more_candidates = True
                        yield p.indicators[i]
                i += 1

        @staticmethod
        def _pre_append_confirmed(items: list[dict[str, Any]], confirmed):
            if len(confirmed) > 0:
                max_score = 0
                best = {
                    "id": "0",
                    'dataset_id': confirmed[0]['dataset_id'],
                    'primary': confirmed[0]['primary'],
                    "name": confirmed[0]['name'],
                    "where": confirmed[0]['where'],
                }
                for candidate in confirmed:
                    score = candidate['score']
                    if score > max_score:
                        best = {
                            "id": "0",
                            'dataset_id': candidate['dataset_id'],
                            'primary': candidate['primary'],
                            "name": candidate['name'],
                            "where": candidate['where'],
                        }
                        max_score = score
                return [best] + items
            return items

        @staticmethod
        def _get_first_non_primary_index(dataset_id):
            # TODO: fix this hack
            if dataset_id == "72a9a53d-39b3-4ddb-9b71-6d7e83c3c9e1":
                return 3
            return 1

        @staticmethod
        def _is_index_number(key):
            if key.isnumeric():
                number = int(key)
                if 0 <= number <= 1000:
                    return True
            return False

        def _format_relevance_items(self, items):
            primary_dict = {}
            for item in items:
                primary = item['primary']
                _id = item['id']
                name = item['name']
                where = item['where']
                if primary not in primary_dict:
                    primary_dict[primary] = {}
                if 1 <= len(where) <= 2:
                    primary_dict[primary][_id] = name
                else:
                    start = self._get_first_non_primary_index(item['dataset_id'])
                    total = self._format_relevance_unpack(where[start:])
                    primary_dict[primary][_id] = f"{primary}, {total}"
            return self._relevance_dict_to_md(primary_dict)

        @staticmethod
        def _format_relevance_unpack(where):
            total = None
            for where_dict in where:
                dimension_name = list(where_dict.keys())[0]
                value = where_dict[dimension_name]
                if value == "Not applicable" or value == "Not specified":
                    continue
                if total:
                    total += ", "
                else:
                    total = ""
                # seems like there is no value in concept
                # total += f"{dimension_name}: {value}"
                total += f"{value.lower()}"
            return total

        def _relevance_dict_to_md(self, primary_dict):
            markdown = ""
            for primary in primary_dict:
                first = True
                for key in primary_dict[primary]:
                    if self._is_index_number(key):
                        if not primary_dict[primary][key]:
                            markdown += f"- ({key}) {primary}\n"
                            continue

                        if first:
                            markdown += f"- {primary}\n"
                            first = False
                        markdown += f"    - ({key}) {primary_dict[primary][key]}\n"
                        continue
                    markdown += f"- {primary}\n"
                    markdown += f"    - {key}\n"

                    for number in primary_dict[primary][key]:
                        markdown += f"        - ({number}) {primary_dict[primary][key][number]}\n"
            return markdown

        def _prepare_for_relevance(
            self, candidates: list[dict]
        ) -> tuple[list[list[dict[str, Any]]], dict[str, dict[str, Any]]]:
            items = []

            batches = []
            indexed = {}
            for i, candidate in enumerate(candidates, start=1):
                _id = candidate['id']
                metadata = candidate['metadata']

                items.append(
                    {
                        "id": f"{i}",
                        'dataset_id': metadata['dataset_id'],
                        'primary': metadata['primary_normalized'],
                        "name": metadata['name_normalized'],
                        "where": metadata['where'],
                    }
                )
                indexed[str(i)] = {
                    'id': _id,
                    'dataset_id': metadata['dataset_id'],
                    'primary': metadata['primary_normalized'],
                    'name': metadata['name_normalized'],
                    'name_original': metadata['name'],
                    'where': metadata['where'],
                    'series': json.loads(metadata['series']),
                }
                if i % self._outer.config.batch_size == 0:
                    batches.append(items)
                    items = []

            if len(items) > 0:
                batches.append(items)
            return batches, indexed

        def _filter_candidates(
            self,
            stage: ContentStageI,
            indexed,
            relevance,
            dataset_max_score,
            is_only_one_dataset_available: bool,
        ) -> tuple[list[dict[str, Any]], str]:
            result = []
            reasoning = ""

            for candidate_dict in relevance:
                _id = str(list(candidate_dict.keys())[0])
                if _id == "0":
                    continue
                candidate = indexed[_id]
                score = int(candidate_dict[_id])
                dataset_id = str(candidate['dataset_id'])
                self._dataset_max_score(
                    _id,
                    score,
                    dataset_id,
                    dataset_max_score,
                    is_only_one_dataset_available,
                    multi_dataset_threshold=self._outer.config.multi_dataset_score_threshold,
                    single_dataset_threshold=self._outer.config.single_dataset_score_threshold,
                )

                if score > 0:
                    reasoning_item = f"        - [{score}]    {candidate['name_original']}\n"
                    reasoning += reasoning_item
                    if stage:
                        stage.append_content(reasoning_item)
            if stage:
                stage.append_content("\n")

            max_overall_score = (
                max(dataset_max_score.values()) if len(dataset_max_score) > 0 else None
            )
            for candidate_dict in relevance:
                _id = str(list(candidate_dict.keys())[0])
                if _id == "0":
                    continue
                candidate = indexed[_id]
                score = int(candidate_dict[_id])
                candidate['score'] = score
                dataset_id = str(candidate['dataset_id'])

                if self._outer.config.use_only_best_score:
                    is_good_candidate = max_overall_score and score == max_overall_score
                else:
                    is_good_candidate = (
                        dataset_id in dataset_max_score and dataset_max_score[dataset_id] == score
                    )
                if is_good_candidate:
                    result.append(candidate)
            return result, reasoning

        @staticmethod
        def _dataset_max_score(
            _id,
            score,
            dataset_id,
            dataset_max_score,
            is_only_one_dataset_available: bool,
            multi_dataset_threshold: int,
            single_dataset_threshold: int,
        ):
            max_score = 0 if dataset_id not in dataset_max_score else dataset_max_score[dataset_id]
            max_score = max(max_score, score)
            # keep highly and extremely relevant by default
            # and only in case of single dataset available allow somewhat relevant
            if max_score >= multi_dataset_threshold or (
                is_only_one_dataset_available and max_score >= single_dataset_threshold
            ):
                dataset_max_score[dataset_id] = max_score

    def __init__(
        self,
        config: HybridSearchConfig,
        models_api_key: str,
        matching_index: ElasticIndex,
        indicators_index: ElasticIndex,
        vectorstore: VectorStore,
    ):
        self._config = config
        self._llm = get_chat_model(
            api_key=models_api_key,
            model_config=config.search_model_config,
        )
        self._matching_index = matching_index
        self._indicators_index = indicators_index
        self._vectorstore = vectorstore

        self._normalization_chain = (
            hybrid_search_default_prompts.normalization_prompt.get_template()
            | self._llm.with_structured_output(method="json_mode")
        )
        self._separate_subjects_chain = (
            hybrid_search_default_prompts.separate_subjects_prompt.get_template()
            | self._llm.with_structured_output(method="json_mode")
        )
        relevancy_prompt = (
            config.prompts.relevancy_prompts or hybrid_search_default_prompts.relevancy_prompt
        )
        self._relevancy_chain = relevancy_prompt.get_template() | self._llm.with_structured_output(
            method="json_mode"
        )

    @property
    def config(self) -> HybridSearchConfig:
        return self._config

    async def _normalize_input(
        self,
        query: str,
        named_entities: list[NamedEntity],
        period: DateTimeQueryResponse,
        forbidden: set[str],
    ) -> str:
        named_entities_to_remove = set(self.config.named_entities_to_remove)
        entities_str = ""
        if named_entities and named_entities_to_remove:
            for entity in named_entities:
                if entity.entity_type in named_entities_to_remove:
                    entities_str += f" - {entity.entity} ({entity.entity_type}) (REMOVE)\n"
                else:
                    entities_str += f" - {entity.entity} ({entity.entity_type}) (DO NOT REMOVE)\n"
            if entities_str:
                entities_str = "Named Entities:\n" + entities_str

        period_str = ""
        if period and (period.start or period.end):
            if period.start and period.end:
                period_str = f"from {period.start} to {period.end}"
            elif period.start:
                period_str = f"from {period.start}"
            elif period.end:
                period_str = f"to {period.end}"
            period_str = "Time Period:\n" + period_str

        removal_step = ""
        if entities_str and period_str:
            removal_step = (
                "- from the input: "
                "keep entities marked (DO NOT REMOVE), "
                "remove entities marked (REMOVE) "
                "and remove all parts related to Time Period. "
                "If an entity or part of entity appears in multiple categories "
                "and at least one instance is marked (DO NOT REMOVE), keep entity"
            )
        elif entities_str:
            removal_step = (
                "- from the input: "
                "keep entities marked (DO NOT REMOVE), "
                "remove entities marked (REMOVE). "
                "If an entity or part of entity appears in multiple categories "
                "and at least one instance is marked (DO NOT REMOVE), keep entity"
            )
        elif period_str:
            removal_step = "- from the input remove all parts related to Time Period. Only period"

        forbidden_to_remove_str = ""
        forbidden_step = ""
        if forbidden:
            forbidden_to_remove_str = ", ".join(forbidden)
            forbidden_to_remove_str = f"Forbidden to remove words:\n{forbidden_to_remove_str}\n"
            forbidden_step = "- do not remove forbidden to remove words from the input if they are present in input"

        output = await self._normalization_chain.ainvoke(
            {
                "removal_step": removal_step,
                "forbidden_step": forbidden_step,
                "input": query,
                "entities": entities_str,
                "period": period_str,
                "forbidden": forbidden_to_remove_str,
            }
        )
        return output['cleaned_input']

    async def _separate_subjects(self, query: str, forbidden: set[str]) -> list[str]:
        forbidden_str = ""
        if forbidden:
            for item in forbidden:
                if len(item.split()) > 0:
                    forbidden_str += f" - {item}"
            if forbidden_str:
                forbidden_str = f"Forbidden to split phrases:\n{forbidden_str}\n"

        forbidden_step = ""
        if forbidden_str:
            forbidden_step = "- do not split the input into separate queries in the middle of the forbidden to split phrases if they present in input"

        output = await self._separate_subjects_chain.ainvoke(
            {
                "forbidden_step": forbidden_step,
                "input": query,
                "forbidden": forbidden_str,
            }
        )
        return output['queries']

    async def _tokenize(self, value: str) -> str:
        value = value.lower()
        tokens = await self._matching_index.analyze(text=value)
        return " ".join(t.token for t in tokens)

    async def _search_by_query(
        self,
        stage: ContentStageI,
        index: int,
        query: str,
        version_ids: set[int],
        availability: DatasetDimTermsSetType,
    ) -> HybridSearchResultInner:
        if stage:
            stage.append_content(f"\n{index}. {query}\n")

        hybrid_match = self.HybridMatch(self)
        lexical, semantic, llm_scored, selected, reasoning, timings = await hybrid_match.search(
            stage, query, version_ids, availability
        )
        final = self._best_of(selected)
        res = HybridSearchResultInner(
            lexical=lexical,
            semantic=semantic,
            llm_scored=llm_scored,
            final=final,
            timings=timings,
        )
        return res

    async def search(
        self,
        *,
        stage: StageI,
        query: str,
        datasets: dict[str, VersionedDataSet],
        named_entities: list[NamedEntity],
        period: DateTimeQueryResponse,
        availability_queries: DatasetAvailabilityQueriesType,
    ) -> HybridSearchResult:
        availability_dict = self._dimension_queries_to_dict(availability_queries)
        version_ids: set[int] = {ds.version.version_data_id for ds in datasets.values()}

        timings = HybridSearchTimings()
        t_total = time.perf_counter()

        t_lexical_pre_match = time.perf_counter()
        primaries, total, candidates, good_candidates = await self.lexical_pre_match(
            query, "name_normalized", version_ids, self.config.max_lexical_pre_match_candidates
        )
        timings.lexical_pre_match = time.perf_counter() - t_lexical_pre_match
        forbidden = good_candidates | candidates
        logger.info(
            f"[search], {len(good_candidates)} good candidates, {len(candidates)} candidates "
            f"(elapsed: {time.perf_counter() - t_total:0.3f} sec)"
        )
        logger.info(f"[search], {good_candidates=}")
        logger.info(f"[search], {candidates=}")

        t_normalize_input = time.perf_counter()
        normalized = await self._normalize_input(query, named_entities, period, forbidden)
        normalized = normalized.lower()
        timings.normalize_input = time.perf_counter() - t_normalize_input

        if stage:
            stage.append_content("> [raw input query]:\n")
            stage.append_content(f"```\n{query}\n```\n")

            stage.append_content("> [normalized input query for search]:\n")
            stage.append_content(f"```\n{normalized}\n```\n")

            stage.append_content("> [full text] potential known terms:\n")
            forbidden_str = "[" + "]  [".join(forbidden) + "]"
            stage.append_content(f"```\n{forbidden_str}\n```\n")

        logger.info(f"[search], {normalized=}, (elapsed {time.perf_counter() - t_total:0.3f} sec)")

        t_separate_subjects = time.perf_counter()
        queries = await self._separate_subjects(normalized, good_candidates)
        timings.separate_subjects = time.perf_counter() - t_separate_subjects
        logger.info(f"[search], {queries=}, (elapsed {time.perf_counter() - t_total:0.3f} sec)")

        # Each subquery runs in parallel; writing to a shared real `stage`
        # would interleave output. The manager hands every task its own
        # buffered substitute and flushes them sequentially on exit (or a
        # shared DummyStage when the outer stage is disabled).
        with BufferedStagesManager(stage) as stage_manager:
            tasks = [
                self._search_by_query(
                    stage=stage_manager.create(),
                    index=index,
                    query=query,
                    version_ids=version_ids,
                    availability=availability_dict,
                )
                for index, query in enumerate(queries, start=1)
            ]
            t_parallel_subqueries = time.perf_counter()
            partial: list[HybridSearchResultInner] = await async_utils.gather_with_concurrency(
                20, *tasks
            )
            timings.parallel_subqueries_wall = time.perf_counter() - t_parallel_subqueries

        timings.per_subquery = [item.timings for item in partial]

        lexical_merged = self._merge_scored_dicts([item.lexical for item in partial])
        semantic_merged = self._merge_scored_dicts([item.semantic for item in partial])
        llm_scored_merged = [s for item in partial for s in item.llm_scored]

        final_merged: DatasetDimTermsSetType = {}
        for item in partial:
            self._merge_partial(final_merged, item.final)

        final_queries = self._merge_dimensions_into_queries(final_merged)
        logger.info(f"Final hybrid queries: {final_queries}")

        timings.total = time.perf_counter() - t_total

        return HybridSearchResult(
            lexical=lexical_merged,
            semantic=semantic_merged,
            llm_scored=llm_scored_merged,
            final_queries=final_queries,
            timings=timings,
        )

    @staticmethod
    def _scored_dict_to_list(
        score_dict: PlainItemsScoredDict | HarmonizedItemsScoredDict,
    ) -> list[dict]:
        res = []
        for id_, data in sorted(score_dict.items(), key=lambda x: x[1].score, reverse=True):
            res.append({'score': data.score, **data.metadata.model_dump()})  # type: ignore
        return res

    @classmethod
    def _merge_scored_dicts(
        cls, dicts: list[PlainItemsScoredDict | HarmonizedItemsScoredDict]
    ) -> list[dict]:
        scored = [item for d in dicts for item in cls._scored_dict_to_list(d)]
        res = sorted(scored, key=lambda x: x['score'], reverse=True)
        return res

    @staticmethod
    def _best_of(matching_results):
        result = {}
        for matching_result in matching_results:
            dataset_id = matching_result['dataset_id']
            series = matching_result['series']
            if dataset_id not in result:
                result[dataset_id] = {}
            for dimension_dict in series:
                dimension_id = list(dimension_dict.keys())[0]
                code = dimension_dict[dimension_id]
                if dimension_id not in result[dataset_id]:
                    result[dataset_id][dimension_id] = set()
                result[dataset_id][dimension_id] |= {code}

        return result

    @staticmethod
    def _merge_search_result(dq_dataset, partial):
        if not partial or len(partial) == 0:
            return
        for dataset_id in partial:
            if dataset_id not in dq_dataset:
                dq_dataset[dataset_id] = {}
            for dimension_id in partial[dataset_id]:
                if dimension_id not in dq_dataset[dataset_id]:
                    dq_dataset[dataset_id][dimension_id] = set()
                dq_dataset[dataset_id][dimension_id] |= set(partial[dataset_id][dimension_id])

    @staticmethod
    def _merge_partial(result: DatasetDimTermsSetType, partial: DatasetDimTermsSetType):
        if not partial or len(partial) == 0:
            return

        for dataset_id in partial:
            if dataset_id not in result:
                result[dataset_id] = {}
            for dimension_id in partial[dataset_id]:
                if dimension_id not in result[dataset_id]:
                    result[dataset_id][dimension_id] = set()
                result[dataset_id][dimension_id] |= partial[dataset_id][dimension_id]

    @staticmethod
    def _merge_dimensions_into_queries(
        dq_dataset: DatasetDimTermsSetType,
    ) -> dict[str, list[DimensionQuery]]:
        result: dict[str, list[DimensionQuery]] = {}
        for dataset_id in dq_dataset:
            dataset_id_str = str(dataset_id)
            if dataset_id_str not in result:
                result[dataset_id_str] = []
            for dimension_id in dq_dataset[dataset_id]:
                dq = DimensionQuery(
                    dimension_id=dimension_id,
                    values=dq_dataset[dataset_id][dimension_id],
                    operator=QueryOperator.IN,
                )
                result[dataset_id_str].append(dq)
        return result

    @staticmethod
    def _dimension_queries_to_dict(
        dimension_queries_dict: DatasetAvailabilityQueriesType,
    ) -> DatasetDimTermsSetType:
        availability_dict: dict[str, dict] = collections.defaultdict(dict)
        for dataset_id in dimension_queries_dict:
            for dimension_query in dimension_queries_dict[dataset_id].dimensions_queries:
                dimension_id = dimension_query.dimension_id
                values = dimension_query.values
                availability_dict[dataset_id][dimension_id] = set(values)
        return availability_dict

    async def lexical_pre_match(
        self, query: str, highlight_field: str, version_ids: set[int], max_candidates: int
    ) -> tuple[int, int, set[str], set[str]]:
        search_result = await self._hints_by_lexical(
            query, highlight_field, version_ids, max_candidates
        )

        candidates: set[str] = set()
        good_candidates: set[str] = set()
        skipped = 0
        for hit in search_result.hits.hits:
            if hit.highlight is None:
                # TODO: review this and fix if possible
                logger.warning("No highlight found for hit, skipping.")
                skipped += 1
                continue

            highlight = hit.highlight[highlight_field][0]
            primary = hit.source[highlight_field]
            primary_tokenized = await self._tokenize(primary)

            running = []
            for token in highlight.split():
                match = self.RE_HIGHLIGHT_MATCH_1.match(token)
                if match:
                    matched = match.group(0)
                    matched = matched.replace("<em>", "")
                    matched = matched.replace("</em>", "")
                    running.append(matched)
                else:
                    tokenized = await self._tokenize(token)
                    if len(running) > 0 and (not tokenized or len(tokenized.strip()) == 0):
                        running.append(token)
                        continue
                    if len(running) == 0:
                        continue
                    await self._assess_candidate(
                        candidates, good_candidates, primary_tokenized, running
                    )
                    running = []
            await self._assess_candidate(candidates, good_candidates, primary_tokenized, running)

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} of {len(search_result.hits.hits)} "
                "hits due to missing highlights"
            )

        good_candidates = await self._remove_duplicates(good_candidates)
        candidates = await self._cleanup_candidates(good_candidates, candidates)

        total = search_result.hits.total.value
        primaries = (
            len(search_result.aggregations['primary']['buckets'])
            if search_result.aggregations
            else 0
        )
        return primaries, total, candidates, good_candidates

    async def _assess_candidate(self, candidates, good_candidates, primary_tokenized, running):
        if len(running) > 0:
            candidate = " ".join(running)
            exact_tokenized = await self._tokenize(candidate)
            if primary_tokenized == exact_tokenized or len(exact_tokenized.split()) > 1:
                if candidate in good_candidates:
                    return
                good_candidates |= {candidate}
            else:
                candidates |= {candidate}

    async def _remove_duplicates(self, good_candidates):
        good_candidates = sorted(good_candidates, key=lambda x: len(x), reverse=True)
        good_list = set()
        result = set()
        for good_candidate in good_candidates:
            tokenized = await self._tokenize(good_candidate)
            if tokenized in good_list:
                continue
            found = False
            for good in good_list:
                if tokenized in good:
                    found = True
                    break
            if found:
                continue
            good_list |= {tokenized}
            result |= {good_candidate}
        return result

    async def _cleanup_candidates(self, good_candidates, candidates):
        good_list = set()
        for good_candidate in good_candidates:
            tokenized = await self._tokenize(good_candidate)
            good_list |= {tokenized}

        result = set()
        candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
        for candidate in candidates:
            tokenized = await self._tokenize(candidate)
            if tokenized in good_list:
                continue
            found = False
            for good in good_list:
                if tokenized in good:
                    found = True
                    break
            if found:
                continue
            for good in result:
                if tokenized in good:
                    found = True
                    break
            if found:
                continue
            result |= {candidate}
        return result

    async def _hints_by_lexical(
        self, user_query: str, highlight_field: str, version_ids: set[int], max_candidates: int
    ) -> SearchResult:

        query = {
            "bool": {
                "must": [
                    {"match": {"primary_normalized": {"query": user_query}}},
                ],
                "should": [
                    {"match": {"name_normalized": {"query": user_query, "boost": 0.3}}},
                ],
                "filter": [{"terms": {"version_id": list(version_ids)}}],
            }
        }

        aggs = {
            "primary": {
                "terms": {"field": "primary_normalized.keyword", "size": 1000},
                "aggs": {"dataset_id": {"terms": {"field": "dataset_id.keyword", "size": 10}}},
            }
        }

        highlight = {
            "fields": {
                f"{highlight_field}": {"number_of_fragments": 0, "fragment_size": 2147483647}
            }
        }
        return await self._indicators_index.search(
            query=query,
            aggs=aggs,
            highlight=highlight,
            explain=True,
            size=max_candidates,
        )
