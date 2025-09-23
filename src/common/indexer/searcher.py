import json
import re
import time
from collections.abc import Generator

from langchain_core.documents import Document

from common.config.logging import logger
from common.data.base import DimensionQuery, QueryOperator
from common.prompts import IndexerPrompts
from common.settings.hybrid_index import HybridIndexSettings
from common.utils import Cache, async_utils
from common.utils.elastic import ElasticIndex, SearchResult
from common.utils.models import get_chat_model
from common.vectorstore import VectorStore

_CACHE: Cache[dict] = Cache(ttl=15 * 60)  # 15 min


class Search:
    MAX_OUTPUT = 32
    # TODO: re-use channel config?
    NER_TO_REMOVE = {"Country/Reference area", "Counterpart area"}
    RE_HIGHLIGHT_MATCH_1 = re.compile(r"(\(?<em>.+</em>\)*,?)")

    class HybridMatch:
        MAX_OUTPUT_MULT = 2
        MAX_OUTPUT_DIV = 4
        BATCH_SIZE = 32

        def __init__(self, outer):
            self._outer = outer
            self._hybrid = None
            self._max_component = None
            self._candidates = None
            self._candidates_primary_components = None

        async def search(self, stage, query: str, availability, max_output: int):
            plan = await self.query_planner(stage, query, availability, max_output)

            result: list = []
            reasoning = "\n        relevance score: 1 - 3 (1 - lowest, 3 - highest)\n"
            candidates = await self._hybrid_candidates(
                plan['query'], availability, plan['max_output'], plan['alpha']
            )
            batches, indexed = self._prepare_for_relevance(candidates)
            dataset_max_score: dict[str, float] = {}
            if stage:
                stage.append_content(reasoning)
            for items in batches:
                items = self._pre_append_confirmed(items, result)
                relevance = await self._relevance_candidates(query, items)
                candidates, batch_reasoning = self._filter_candidates(
                    stage, indexed, relevance, availability, dataset_max_score
                )
                result += candidates
                reasoning += batch_reasoning
            return result, reasoning

        async def query_planner(self, stage, query: str, availability, max_output: int):
            primaries, total, candidates, good_candidates = await self._outer.lexical_pre_match(
                query, "primary_normalized", availability, max_output
            )
            if len(good_candidates) > 0:
                # re-calculate numbers based on good candidates
                primaries, total, _, _ = await self._outer.lexical_pre_match(
                    " ".join(good_candidates), "primary_normalized", availability, max_output
                )

            if stage:
                good_candidates_str = "[" + "]  [".join(good_candidates) + "]"
                stage.append_content(
                    f"\n    > [planner] full-text candidates - total: {total}, primary: {primaries}, candidates: {good_candidates_str}\n"
                )

            # default, not used now, potentially can be used in more complex cases
            plan_output = max_output
            alpha = 0.9

            if len(good_candidates) == 0:
                # fallback to semantic
                alpha = 0.999
                plan_output = 2 * max_output
            else:
                # go with hybrid
                if primaries > max_output:
                    alpha = 0.8
                    plan_output = 2 * max_output
                else:
                    alpha = 0.9
                    plan_output = max_output

            if stage:
                stage.append_content(f"    >  =>  max candidates {plan_output}, alpha: {alpha}\n")
            return {"query": query, "max_output": plan_output, "alpha": alpha}

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

        async def _relevance_candidates(self, input, items):
            items_str = self._format_relevance_items(items)
            output = await self._outer._relevance_chain.ainvoke(
                {"statement": input.lower(), "items": items_str}
            )
            return output['relevance']

        async def _lexical(self, user_query: str, max_query: int) -> dict[str, dict]:
            query = {
                "bool": {
                    "must": [
                        {"match": {"primary_normalized": {"query": user_query}}},
                    ],
                    "should": [
                        {"match": {"name_normalized": {"query": user_query, "boost": 0.3}}},
                    ],
                }
            }
            result: SearchResult = await self._outer._indicators_index.search(
                query=query, size=max_query
            )
            lex_indexed: dict[str, dict] = {}
            lex_max_score = result.hits.max_score

            for hit in result.hits.hits:
                norm_score = (
                    self._lex_teor_min_max(hit.score, lex_max_score) if lex_max_score else 0
                )
                self._result_to_dict(hit.source, norm_score, lex_indexed, "LEX ")

            return lex_indexed

        async def _es_get_by_id(self, _id: str) -> dict:
            query = {"term": {"id.keyword": _id}}
            result: SearchResult = await self._outer._indicators_index.search(query=query, size=1)
            return result.hits.hits[0].source

        async def _semantic_raw(
            self, user_query: str, max_query: int
        ) -> list[tuple[Document, float]]:
            return await self._outer._vectorstore.search_with_similarity_score(
                user_query, max_query
            )

        async def _semantic(self, result: list[tuple[Document, float]]) -> dict[str, dict]:
            sem_indexed: dict[str, dict] = {}
            sem_max_score = result[0][1]

            for doc, score in result:
                metadata = doc.metadata
                norm_score = self._sem_teor_min_max(score, sem_max_score)
                self._result_to_dict(metadata, norm_score, sem_indexed, "SEM ")

            return sem_indexed

        @staticmethod
        def _result_to_dict(
            source: dict, norm_score: float, indexed: dict[str, dict], log_prefix: str = ""
        ) -> None:
            _id = source['id']
            if _id not in indexed:
                indexed[_id] = {"score": norm_score, "metadata": source}
                score_str = "{:0.3f}".format(norm_score)
                logger.info(f"{log_prefix}{score_str}    {source['name']}")

        async def _hybrid_combination(
            self,
            result: list[tuple[Document, float]],
            lex_indexed: dict[str, dict],
            sem_indexed: dict[str, dict],
            alpha: float,
        ) -> dict[str, dict]:
            hybrid: dict[str, dict] = {}

            for doc, _ in result:
                metadata = doc.metadata
                _id = metadata['id']

                sem = sem_indexed[_id]['score']
                lex = 0 if _id not in lex_indexed else lex_indexed[_id]['score']

                # get enriched metadata
                if _id in lex_indexed:
                    metadata = lex_indexed[_id]['metadata']
                else:
                    metadata = await self._es_get_by_id(_id)

                if _id not in hybrid:
                    score = self._convex_combination(sem, lex, alpha)
                    hybrid[_id] = {
                        'score': score,
                        'metadata': metadata,
                    }
            return hybrid

        @staticmethod
        def _filter_hybrid_by_availability(hybrid, dimension_queries_dict):
            if not dimension_queries_dict:
                return hybrid

            result = {}
            for key in hybrid:
                metadata = hybrid[key]['metadata']
                dataset_id = str(metadata['dataset_id'])
                series = json.loads(metadata['series'])
                found = True
                for dimension_dict in series:
                    dimension_id = list(dimension_dict.keys())[0]
                    value = dimension_dict[dimension_id]
                    if dataset_id not in dimension_queries_dict:
                        found = False
                        break
                    dimension_queries = dimension_queries_dict[dataset_id]
                    if dimension_id not in dimension_queries:
                        found = False
                        break
                    availability = dimension_queries[dimension_id]
                    if value not in availability:
                        found = False
                        break
                if found:
                    result[key] = hybrid[key]
            return result

        async def _hybrid_candidates(
            self, query: str, availability, max_output: int, alpha: float
        ) -> list[dict]:
            max_query: int = self.MAX_OUTPUT_MULT * max_output
            lex_indexed: dict[str, dict] = await self._lexical(query, 2 * max_query)
            sem_raw: list[tuple[Document, float]] = await self._semantic_raw(query, max_query)
            sem_indexed: dict[str, dict] = await self._semantic(sem_raw)
            hybrid: dict[str, dict] = await self._hybrid_combination(
                sem_raw, lex_indexed, sem_indexed, alpha
            )
            # TODO: think about using of availability
            # hybrid = self._filter_hybrid_by_availability(hybrid, availability)

            hybrid_sorted: list[tuple[str, dict]] = sorted(
                hybrid.items(), key=lambda x: x[1]['score'], reverse=True
            )
            dataset_dict = self._build_ordered_dataset_dict(hybrid_sorted, max_output)
            candidates: list[dict] = self._get_dataset_indicators_round_robin(
                dataset_dict, max_output
            )
            candidates = self.make_sure_top_n_is_included(candidates, hybrid_sorted, max_output)

            for candidate in candidates:
                metadata = candidate['metadata']
                logger.info(f"TOP_{max_output}    [{metadata['dataset_id']}]  {metadata['name']}")
            return candidates

        def make_sure_top_n_is_included(
            self, candidates: list[dict], hybrid_sorted: list[tuple[str, dict]], max_output: int
        ) -> list[dict]:
            max_top_n = int(max_output / self.MAX_OUTPUT_DIV)
            top_n = {k: v['metadata'] for k, v in hybrid_sorted[0:max_top_n]}

            candidates_ids = {c['id'] for c in candidates}
            top_n = {k: v for k, v in top_n.items() if k not in candidates_ids}

            for _id, metadata in top_n.items():
                primary = metadata['primary_normalized']
                i = max_output - 1
                while i >= 0:
                    candidate = candidates[i]
                    if candidate["metadata"]['primary_normalized'] == primary:
                        candidates.insert(i, {"id": _id, "metadata": metadata})
                        break
                    i -= 1
                else:
                    candidates.append({"id": _id, "metadata": metadata})
            return candidates[0:max_output]

        def _build_ordered_dataset_dict(
            self, hybrid_sorted: list[tuple[str, dict]], max_output: int
        ) -> dict[str, list[dict]]:
            max_primary = max_output - int(max_output / self.MAX_OUTPUT_DIV)

            primary_dict: set[str] = set()
            dataset_dict = {}
            for _id, data in hybrid_sorted:
                metadata = data['metadata']

                dataset_id = str(metadata['dataset_id'])
                if dataset_id in dataset_dict:
                    continue

                primary = metadata['primary_normalized']
                dataset_dict[dataset_id] = [
                    {'primary': primary, 'indicators': [{"id": _id, "metadata": metadata}]}
                ]

            for _id, data in hybrid_sorted:
                score: float = data['score']
                meta: dict = data['metadata']

                logger.info(f"HYBRID {score:0.3f}    {meta['name']}")

                dataset_id = str(meta['dataset_id'])
                if len(dataset_dict[dataset_id]) >= max_output - len(dataset_dict):
                    continue

                primary = meta['primary_normalized']
                found = False
                for primaries_record in dataset_dict[dataset_id]:
                    if primaries_record['primary'] == primary:
                        if primaries_record['indicators'][0]['id'] != _id:
                            primaries_record['indicators'].append({"id": _id, "metadata": meta})
                        found = True
                        break

                should_create_new_primary = (
                    primary in primary_dict or len(primary_dict) < max_primary
                )
                if not found and should_create_new_primary:
                    primary_dict.add(primary)
                    dataset_dict[dataset_id].append(
                        {'primary': primary, 'indicators': [{"id": _id, "metadata": meta}]}
                    )
            return dataset_dict

        def _get_dataset_indicators_round_robin(
            self, dataset_dict: dict[str, list[dict]], max_output: int
        ) -> list[dict]:
            generator_dict = {
                dataset_id: self._hybrid_candidates_dataset_generator(primaries)
                for dataset_id, primaries in dataset_dict.items()
            }
            result = []
            have_more_candidates = True
            while have_more_candidates:
                have_more_candidates = False
                for dataset_id in dataset_dict:
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
            primaries: list[dict],
        ) -> Generator[dict, None, None]:
            # ~~~ Simplified Example: ~~~
            # [{"indicators": [101, 102]}, {"indicators": [201, 202, 203]}, {"indicators": [301]}]
            # -> [101, 201, 301, 102, 202, 203]
            i = 0
            have_more_candidates = True
            while have_more_candidates:
                have_more_candidates = False
                for primary in primaries:
                    candidates = primary['indicators']
                    if len(candidates) > i:
                        have_more_candidates = True
                        yield candidates[i]
                i += 1

        @staticmethod
        def _pre_append_confirmed(items, confirmed):
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
            if int(dataset_id) == 7:
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
            logger.info(markdown)
            return markdown

        @classmethod
        def _prepare_for_relevance(cls, candidates):
            items = []

            batches = []
            indexed = {}
            i = 1
            for candidate in candidates:
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
                if i % cls.BATCH_SIZE == 0:
                    batches.append(items)
                    items = []
                i += 1
            if len(items) > 0:
                batches.append(items)
            return batches, indexed

        def _filter_candidates(self, stage, indexed, relevance, availability, dataset_max_score):
            result = []
            reasoning = ""

            for candidate_dict in relevance:
                _id = str(list(candidate_dict.keys())[0])
                if _id == "0":
                    continue
                candidate = indexed[_id]
                score = int(candidate_dict[_id])
                dataset_id = str(candidate['dataset_id'])
                series = candidate['series']
                self._dataset_max_score(
                    _id, availability, score, dataset_id, dataset_max_score, series
                )

                logger.info(f" - {candidate['name_original']} [{score}]")
                if score > 0:
                    reasoning_item = f"        - [{score}]    {candidate['name_original']}\n"
                    reasoning += reasoning_item
                    if stage:
                        stage.append_content(reasoning_item)
            if stage:
                stage.append_content("\n")

            for candidate_dict in relevance:
                _id = str(list(candidate_dict.keys())[0])
                if _id == "0":
                    continue
                candidate = indexed[_id]
                score = int(candidate_dict[_id])
                candidate['score'] = score
                dataset_id = str(candidate['dataset_id'])
                if dataset_id not in dataset_max_score or score < dataset_max_score[dataset_id]:
                    continue
                result.append(candidate)
            return result, reasoning

        @staticmethod
        def _dataset_max_score(_id, availability, score, dataset_id, dataset_max_score, series):
            if dataset_id not in availability:
                return

            for series_dict in series:
                dimension_id = list(series_dict.keys())[0]
                value = series_dict[dimension_id]
                if dimension_id not in availability[dataset_id]:
                    return
                if value not in availability[dataset_id][dimension_id]:
                    return

            max_score = 0 if dataset_id not in dataset_max_score else dataset_max_score[dataset_id]
            max_score = max(max_score, score)
            # keep highly and extremely relevant by default
            # and only in case of single dataset available allow somewhat relevant
            if max_score >= 2 or (len(availability) == 1 and max_score > 0):
                dataset_max_score[dataset_id] = max_score

    def __init__(
        self,
        models_api_key: str,
        matching_index: ElasticIndex,
        indicators_index: ElasticIndex,
        vectorstore: VectorStore,
    ):
        settings = HybridIndexSettings()
        self._llm = get_chat_model(
            api_key=models_api_key,
            model_config=settings.searcher_model_config,
        )
        self._matching_index = matching_index
        self._indicators_index = indicators_index
        self._vectorstore = vectorstore

        self._normalize_chain = (
            IndexerPrompts.get_search_normalize_prompts()
            | self._llm.with_structured_output(method="json_mode")
        )
        self._separate_subjects_chain = (
            IndexerPrompts.get_separate_subjects_prompts()
            | self._llm.with_structured_output(method="json_mode")
        )
        self._relevance_chain = (
            IndexerPrompts.get_relevance_prompts()
            | self._llm.with_structured_output(method="json_mode")
        )

    async def _normalize_input(self, input, named_entities, period, forbidden):
        entities_str = ""
        if named_entities and len(named_entities) > 0:
            for entity in named_entities:
                if entity.entity_type not in Search.NER_TO_REMOVE:
                    continue
                entities_str += f" - {entity.entity} ({entity.entity_type})\n"
            if entities_str != "":
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
        if entities_str != "" or period_str != "":
            if entities_str and period_str:
                removal_step = "- from the input remove parts related to the Named Entities and Time Period. Only listed entities and period"
            elif entities_str:
                removal_step = "- from the input remove parts related to the Named Entities. Only listed entities"
            elif period_str:
                removal_step = "- from the input remove parts related to Time Period. Only period"

        forbidden_str = ""
        if forbidden and len(forbidden) > 0:
            forbidden_str = ", ".join(forbidden)
            forbidden_str = f"Forbidden to remove words:\n{forbidden_str}\n"

        forbidden_step = ""
        if forbidden_str != "":
            forbidden_step = (
                "- do not remove forbidden to remove words from the input if they present in input"
            )

        output = await self._normalize_chain.ainvoke(
            {
                "removal_step": removal_step,
                "forbidden_step": forbidden_step,
                "input": input,
                "entities": entities_str,
                "period": period_str,
                "forbidden": forbidden_str,
            }
        )
        return output['cleaned_input']

    async def _separate_subjects(self, input, forbidden):
        forbidden_str = ""
        if forbidden and len(forbidden) > 0:
            for item in forbidden:
                if len(item.split()) > 0:
                    forbidden_str += f" - {item}"
            if forbidden_str != "":
                forbidden_str = f"Forbidden to split phrases:\n{forbidden_str}\n"

        forbidden_step = ""
        if forbidden_str != "":
            forbidden_step = "- do not split the input into separate queries in the middle of the forbidden to split phrases if they present in input"

        output = await self._separate_subjects_chain.ainvoke(
            {
                "forbidden_step": forbidden_step,
                "input": input,
                "forbidden": forbidden_str,
            }
        )
        return output['queries']

    async def _tokenize(self, value: str) -> str:
        value = value.lower()
        tokens = await self._matching_index.analyze(text=value)
        return " ".join(t.token for t in tokens)

    async def _search_by_query(self, stage, query, availability, max_output):
        if stage:
            stage.append_content(f"\n1. {query}\n")

        tokenized = await self._tokenize(query)
        cached = _CACHE.get(tokenized)
        if cached:
            if stage:
                stage.append_content("\n    > [cache] using cached result\n\n")
                stage.append_content(cached['reasoning'])
            return cached['result']

        hybrid_match = Search.HybridMatch(self)
        matching_result, reasoning = await hybrid_match.search(
            stage, query, availability, max_output
        )
        search_result = self._best_of(matching_result)

        if len(search_result) > 0:
            _CACHE.set(tokenized, {'result': search_result, 'reasoning': reasoning})
        return search_result

    async def search(
        self, stage, query: str, datasets=None, named_entities=None, period=None, availability=None
    ):
        availability = self._dimension_queries_to_dict(availability)

        pc0 = time.perf_counter()
        logger.info("[search], entering ...")

        primaries, total, candidates, good_candidates = await self.lexical_pre_match(
            query, "name_normalized", availability, Search.MAX_OUTPUT
        )
        forbidden = good_candidates | candidates
        elapsed = time.perf_counter() - pc0
        logger.info(
            f"[search], good candidates: {len(good_candidates)}, candidates: {len(candidates)}, ({elapsed:0.3f})"
        )
        logger.info(f"[search], {good_candidates=}")
        logger.info(f"[search], {candidates=}")

        if stage:
            stage.append_content("> [full text] potential known terms:\n")
            stage.append_content("```\n")
            forbidden_str = "[" + "]  [".join(forbidden) + "]"
            stage.append_content(f"{forbidden_str}\n")
            stage.append_content("```\n")

        normalized = await self._normalize_input(query, named_entities, period, forbidden)
        normalized = normalized.lower()
        elapsed = time.perf_counter() - pc0
        logger.info(f"[search], {normalized=}, ({elapsed:0.3f})")

        queries = await self._separate_subjects(normalized, good_candidates)
        elapsed = time.perf_counter() - pc0
        logger.info(f"[search], {queries=}, ({elapsed:0.3f})")

        async def _run_query(query):
            search_result = await self._search_by_query(stage, query, availability, self.MAX_OUTPUT)
            self._log_search_result(query, search_result)
            return search_result

        _CACHE.cleanup()
        tasks = [_run_query(query) for query in queries]
        partial_results = await async_utils.gather_with_concurrency(20, *tasks)

        result: dict = {}
        for search_result in partial_results:
            self._merge_partial(result, search_result)

        self._log_search_result(", ".join(queries), result)
        dq_result = self._merge_dimensions(result)
        logger.info(f"{dq_result=}")
        return dq_result

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
    def _log_search_result(query, partial):
        if not partial or len(partial) == 0:
            return
        logger.info(f"{query=}")
        for dataset_id in partial:
            logger.info(f"\t{dataset_id=}")
            for dimension_id in partial[dataset_id]:
                logger.info(f"\t\t{dimension_id=} {partial[dataset_id][dimension_id]}")

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
    def _merge_partial(result, partial):
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
    def _merge_dimensions(dq_dataset):
        result = {}
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
    def _dimension_queries_to_dict(dimension_queries_dict):
        if not dimension_queries_dict:
            return None
        availability_dict = {}
        for dataset_id in dimension_queries_dict:
            if dataset_id not in availability_dict:
                availability_dict[dataset_id] = {}
            for dimension_query in dimension_queries_dict[dataset_id].dimensions_queries:
                dimension_id = dimension_query.dimension_id
                values = dimension_query.values
                availability_dict[dataset_id][dimension_id] = set(values)
        return availability_dict

    async def lexical_pre_match(self, query: str, highlight_field, availability, max_output):
        search_result = await self._hints_by_lexical(
            query, highlight_field, availability, max_output
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
                f"Skipped {skipped} of {len(search_result.hits.hits)} hits due to missing highlights."
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
        self, user_query: str, highlight_field, availability, max_output
    ) -> SearchResult:
        # TODO: think about using of availability
        # should = []
        # for dataset_id in availability:
        #     should.append({"term": {"dataset_id.keyword": int(dataset_id)}})

        query = {
            "bool": {
                "must": [
                    {"match": {"primary_normalized": {"query": user_query}}},
                ],
                "should": [
                    {"match": {"name_normalized": {"query": user_query, "boost": 0.3}}},
                ],
                # "filter": [{"bool": {"should": should}}],
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
            size=max_output,
        )
