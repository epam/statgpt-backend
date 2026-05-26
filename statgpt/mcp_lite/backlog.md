# mcp_lite — changelog & backlog

Decisions and changes made in `mcp_lite/` after the initial PLAN landed. Newest first.

Format per entry:
- **What** — concrete change (file + paragraph reference)
- **Why** — evidence that drove the change
- **Cost / risk** — alternatives considered, what we accept

---

## 2026-05-25 — Drop fuzzy fallback; route by dim-type with dynamic redirects + wire up special-dim vector search

**What.** Replaced the in-memory fuzzy fallback in `search_codes` with **explicit dim-type routing**:

| dim type | path |
|---|---|
| `non_indicator` | non-indicator vector store (`facade.search_non_indicator_dimensions_scored(dimension_id=...)`) |
| `special` | special-dim vector store (NEW `facade.search_special_dim_values_by_id(dim_id=...)`) |
| `indicator` | `ToolError` with a dynamic redirect message pointing to `search_indicators` + `sample_dim_values` |
| `time` | `ToolError` redirecting to `execute_sdmx_query` time args |

Plus: new facade method `search_special_dim_values_by_id(query, dim_id, dataset_versions, k)` in [`statgpt/app/services/chat_facade.py`](../app/services/chat_facade.py:507). Mirrors `search_non_indicator_dimensions_scored`'s shape — same vector-store-with-`DIMENSION_ID`-filter pattern. Coexists with the existing `search_special_dimension_scored` (which is keyed on `processor.id` for the statgpt-chain use case).

**Why.** The fuzzy fallback shipped a few hours ago was a defensive patch that duplicated capability already covered by `search_indicators`. Specifically:
- For indicator-classified dims, the values ARE properties of compound indicators and are already searchable (with semantic ranking) via `search_indicators` — the agent reads `dimensions.{dim_id}` from a top hit.
- Fuzzy was filling a gap that didn't really exist — it was working around agents using the wrong tool.

The cleaner design is explicit redirects, not silent fuzzy approximation. Routing also exposed a real bug: **special dims have never been searchable in mcp_lite** (`search_codes` only queried the non-indicator vector store; the old code had a comment "special: deferred to v0b"). Special-dim values are actually indexed — just in a different vector store — so wiring them up was 10 LOC.

**Empirical post-fix sanity:**
```
search_codes(BIS:WS_NA_SEC_DSS, "Germany",    dim_id="REF_AREA")     → DE @ 1.0 (non-indicator vector)
search_codes(BIS:WS_NA_SEC_DSS, "securitisation vehicles", dim_id="REF_SECTOR")
                                                                     → ToolError naming the dim, dataset, and concrete recovery:
                                                                       "call search_indicators(query=<concept>, dataset_id='BIS:WS_NA_SEC_DSS(1.0)')
                                                                        and read dimensions.REF_SECTOR from a matching result;
                                                                        or sample_dim_values(..., limit=-1)"
search_codes(BIS:WS_NA_SEC_DSS, "2024", dim_id="TIME_PERIOD")        → ToolError: "filter by date range on execute_sdmx_query(time_start, time_end)"
search_codes(BIS:WS_LBS_D_PUB,  "Japan")                             → cross-dim exploration via non-indicator vector: JP @ 1.0 in L_PARENT_CTY / L_REP_CTY / L_CP_COUNTRY
```

Error messages are dynamically constructed — `dim_id`, `dataset_id`, and the concrete recovery call all interpolated, so the agent can copy-paste straight into its next tool call.

**Cost / risk.**
- **Behavioural change** for indicator-classified-dim searches: previously returned fuzzy hits, now raises `ToolError` with redirect. Forces agent through the conceptually-correct path (`search_indicators`), at a cost of 1 extra call for cases where the agent guessed wrong. Net effect over the round-4 case load: 0 — agents are perfectly capable of following the redirect once told.
- **Special-dim search**: pure addition. No regression — the path was empty before.
- **No new dependency** (`difflib`'s `SequenceMatcher` no longer used; import removed).
- **Channel reality check**: no dataset in the current `statgpt-gtdc` channel has any special-classified dims, so the special-dim path is dormant infrastructure today. Ready for the first dataset that does.

**Options considered**:
- (A) Keep fuzzy fallback — duplicates `search_indicators` coverage, agent never learns the right primitive.
- (B) Bypass facade, hit `_get_special_dimensions_vector_store` directly — works, uses a leading-underscore method as the indicator-hybrid path already does. Acceptable but less symmetric.
- (C) Add new clean facade method `search_special_dim_values_by_id` ← **chosen**. Symmetric with the non-indicator one; existing `search_special_dimension_scored` left untouched for the chain caller that does have a `processor` instance.

---

## 2026-05-25 — Split `search_codes` into `search_indicators` + `search_codes` (with fuzzy fallback)

**What.** Old `search_codes(query, dataset_id?, dim_id?, top_k, sources?)` was a dual-mode tool that did either cross-dataset discovery OR scoped dim-value lookup depending on which args you set. Now split:

```
search_indicators(query, dataset_id?, top_k)
   # hybrid lex+sem on the compound-indicator index; cross-dataset by default
   # returns IndicatorMatch with `dimensions` breakdown ready for execute_sdmx_query.selection

search_codes(dataset_id, query, dim_id?, top_k)
   # scoped to ONE dataset; dim_id optional for cross-dim exploration within it
   # backend: vector store + in-memory fuzzy-on-codelist, merged by (dim_id, code)
   # fuzzy fallback closes the indicator-classified-dims coverage gap (Q4 Dutch S125A etc.)
```

[`statgpt/mcp_lite/tools/search.py`](tools/search.py) — full rewrite. [`statgpt/mcp_lite/schemas.py`](schemas.py) — new `IndicatorMatch` / `IndicatorSearchResult`; slimmed `CodeMatch` (drops the conditional `dimensions` / `available_in` fields that signalled dual-mode behaviour).

**Why.** Round-3 eval surfaced two related issues:

1. **Mental-model mismatch.** "Cross-dataset" really means "the same indicator concept exists in multiple datasets" (e.g. GDP in WEO + IFS). The unified `search_codes` was conflating that with "find a code for a dim value", which is always single-dataset.
2. **Coverage gap.** `search_codes(dim_id=L_INSTR|L_MEASURE|REF_SECTOR)` returned 0 hits on 3+ agents across round-3, because BIS dataset configs classify those as *indicator dimensions* — their values aren't in the non-indicator vector store. The values exist (the codelists are loaded in memory), they just aren't indexed for semantic search.

The split addresses (1) by giving each job its own tool with the right required-args; the fuzzy fallback in `search_codes` addresses (2) without requiring a reindex.

**Fuzzy scoring** (`_fuzzy_score`):
- Exact substring match → score 1.0
- All query tokens match (each is a substring of some name token) → 0.95
- Partial token match → 0.5–0.9 linear in match fraction
- Fallback: `difflib.SequenceMatcher.ratio()`

Token-substring is the dominant rule because dim names like "Financial vehicle corporations engaged in securitisation" don't pattern-match cleanly with character-level ratio against "securitisation vehicles" — but every query token IS a substring of some name token. Stdlib only; no `rapidfuzz` dependency.

**Empirical post-fix:**
```
search_indicators("GDP growth")  → WEO.NGDP_RPCH @ 0.997, WEO.NGDP_RPCHMK @ 0.992, ANEA @ 0.975 — correct dataset(s)
search_codes(BIS:WS_NA_SEC_DSS, "securitisation vehicles", dim_id="REF_SECTOR")
                                  → S125A "Financial vehicle corporations engaged in securitisation" @ 0.95  ← gap closed
search_codes(BIS:WS_LBS_D_PUB, "Japan")  → JP @ 1.0 in L_PARENT_CTY, L_REP_CTY, L_CP_COUNTRY  ← exploration mode preserved
```

**Cost / risk.**
- **Two breaking schema changes**: agents calling old `search_codes(query)` without dataset_id now error (dataset_id is required); agents using `sources=["indicator"]` now use `search_indicators` instead.
- **Tool surface**: 8 → 9 tools. The old `search_codes` was already 2 jobs in a trench coat; explicit counting now matches reality.
- **No new dependency** — `difflib` is stdlib.
- **Fuzzy ranking is char/token-based, not semantic.** For dims that ARE in the vector store (countries, main filter dims), vector wins on synonyms; for unindexed dims, fuzzy is the only option but works well for the actual use cases observed in eval (S125A, S13, L_INSTR=D, etc.).

**Options considered** (full discussion in the round-3 dialogue):
- (A) Reindex: add indicator-dim values to non-indicator vector store. Big infra change, deferred.
- (B) Fuzzy fallback inside unified `search_codes` — works, but perpetuates the dual-mode tool.
- (C) Split + fuzzy fallback ← **chosen** — cleaner per-tool semantics + closes the gap.
- (D) Reject `dim_id=X` when X isn't indexed — punts the problem to the agent.

---

## 2026-05-25 — Arg-name renames: `dims`→`filter`/`selection`, `n`→`limit`, flatten `time_period`

**What.** Four breaking schema changes across three tools, all aligning arg names to what the LLM naturally guesses (evidenced by round-3 eval where 7/10 agents hit at least one arg-name miss).

| tool | old | new |
|---|---|---|
| `availability_query` | `dims` | `filter` |
| `execute_sdmx_query` | `dims` | `selection` |
| `execute_sdmx_query` | `time_period: {start, end}` (nested) | `time_start`, `time_end` (flat) |
| `sample_dim_values` | `n` | `limit` |

Also rewrote all four arg descriptions to be concise yet explanatory (shed ~30% of word count while keeping every essential constraint).

**Why.** Round-3 eval (5 reruns + 5 new) measured the friction empirically:

| pattern | hits / 10 | wrong names tried |
|---|---|---|
| `availability_query.dims` | 7/10 | `filter`, `filters`, `dimensions`, `dim_filter`, `selection` |
| `execute_sdmx_query` time arg | 4/10 | `start_period`, `end_period`, `time_period_start` |
| `sample_dim_values.n` | 3/10 | `limit` |

The convergence on specific wrong names tells us what the LLM *expects*. Picking those names (with one semantic adjustment for `execute_sdmx_query`) brings the schema in line with that expectation.

**Why `selection` on `execute_sdmx_query` and not `filter`:** `filter` is semantically right for `availability_query` (it really does filter the codelists down to reachable values). But `execute_sdmx_query` doesn't filter results — it specifies an SDMX *key/selection* (the URL form is `<DataflowID>/{KEY}/?…` with dot-separated dim values). `selection` is SDMX-native, and 1 agent in round-3 already reached for it. Two different names per tool, but each is more accurate than the shared `dims` was.

**Why flat `time_start`/`time_end` over nested object:** 4 agents in round-3 tried `start_period`, `end_period`, `time_period_start` — all flat-arg shapes. SDMX's own URL convention is flat (`startPeriod`/`endPeriod`). The nested `TimePeriod` object was leaking the response-side schema into the request, which has no reason to share shape.

**Empirical post-rename validation:**
```
sample_dim_values(limit=3)             → ok
sample_dim_values(n=3)                 → rejected: Unexpected keyword argument
availability_query(filter={...})       → ok
availability_query(dims={...})         → rejected: Missing required argument: filter
execute_sdmx_query(selection=..., time_start='2022', time_end='2024')  → ok, 3 rows
execute_sdmx_query(dims=...)           → rejected: Missing required argument: selection
```

**Cost / risk.**
- **Four breaking schema changes.** mcp_lite is pre-v0; no production callers.
- **Different name on the two filter-like args** (`filter` vs `selection`) means agents have to learn 2 names for a similar concept — but each name is semantically accurate for its tool, which the LLM can pattern-match from the docstring.
- The "schema = contract" principle: each rename is *toward* the LLM's natural idiom (as measured), not against it. Trajectory of `partial_dims` → `dims` → `filter`/`selection` is convergent, not bouncing.

**Options considered.**
- (A) Pydantic aliases — rejected: hidden in schema docs; doesn't help first-call guess.
- (B-original) Both filter args → `filter` — rejected: semantic mismatch for `execute_sdmx_query`.
- (B-split) `availability_query.filter` + `execute_sdmx_query.selection` ← **chosen** — each name accurate for its tool.
- (B-neutral) Both → `selection` — rejected: less LLM-natural for the availability case.
- (B-keep-dims-on-execute) Rename only `availability_query` — rejected: leaves friction on `execute_sdmx_query`.
- (C) Improve error text only — rejected: doesn't help first-call friction.

---

## 2026-05-25 — `availability_query`: raise `ToolError` with a recovery hint when filter selects zero observations

**What.** [`statgpt/mcp_lite/tools/dataset.py`](tools/dataset.py) — when the SDMX response's `dimensions_queries_dict` is empty (i.e. `available: {}` post-mapping), `availability_query` now raises `ToolError` instead of returning an empty `AvailabilityResult`. The error message differentiates two sub-cases:

- **Invalid codes** (any filter code not in its dim's in-memory codelist):
  ```
  Filter selects no observations: codes not in the dataset's codelist:
  {'COUNTRY': ['ZZZ']}. Use `search_codes(dataset_id=..., dim_id=...)` or
  `sample_dim_values` to find valid replacements.
  ```
- **All codes valid, combination has no real-world data:**
  ```
  Filter selects no observations, though all codes are valid for the dataset.
  Try widening the filter or removing one dim.
  ```

New helper `_invalid_filter_codes(dataset, filter_dict)` does the local check against each categorical dim's `dim.values` — no extra SDMX call.

**Why.** Empirical edge-case sweep found that `available: {}` covers three distinct sub-cases that agents need to recover from differently:

| sub-case | what really happened | recovery |
|---|---|---|
| A. All filter codes invalid (`COUNTRY=ZZZ`) | no observations; at least one unknown code | fix the bad codes |
| B. Mixed valid + invalid (`COUNTRY=DEU, INDICATOR=FAKE`) | no observations; one code is wrong | fix the bad codes |
| E. All codes valid, real-world no-data combo (Liberian banks → Vanuatu) | no observations; codes are fine | widen or back off one dim |
| C. Pinning everything with valid codes (control) | singletons per dim — NOT empty | n/a (success) |

Pre-fix the agent saw the same `{"available": {}}` for A/B/E. A3 in round-1 hit case A (typo'd country code) and spent two further calls "narrowing" before realising the filter was wrong. The original round-1 hypothesis ("`{}` could mean fully constrained") was empirically wrong — case C confirmed full constraint returns singletons per dim.

**Empirical confirmation (post-fix):**
```
case A (COUNTRY=ZZZ)            → isError, "codes not in codelist: {'COUNTRY': ['ZZZ']}"
case B (DEU + FAKE_IND)         → isError, "codes not in codelist: {'INDICATOR': ['FAKE_IND']}"
case E (BIS Liberian → Vanuatu) → isError, "all codes are valid... Try widening..."
happy path                      → isError=false, clean AvailabilityResult
```

**Why `ToolError` (failure mode) over structured fields:**

1. `availability_query`'s job is to return reachable values. If none are reachable, the probe failed at its job — that's structurally an error, not a successful empty result. Contrast `execute_sdmx_query`, where an empty result IS the answer to "what data exists?" (so we kept the `warning`-field pattern there).
2. Consistent with the existing `KeyError → ToolError` path for unknown `dim_id` — the same probe also raises on unknown values now.
3. `isError: true` is an unmissable signal. Agents that ignore optional fields can't ignore an error result.
4. Keeps the happy-path schema clean — no field bloat for an edge case.

**Cost / risk.**
- **Behavioural change** for callers that previously relied on the empty-`available` response (none in production; no callers wired yet).
- **Local validation** is O(filter-size × codelist-size); WEO's biggest dim is 210 entries → microseconds. Skipped for non-categorical (virtual / time) dims.
- The error message embeds the bad-codes dict inline as Python repr — readable and parseable enough for the agent's next call; not as ergonomic as a structured field, but the agent only needs to read it once before fixing the filter.

**Options considered.**
- (A) Add `status` / `no_data` / `invalid_codes` / `message` fields on `AvailabilityResult` — **initially chosen, then reverted** at user feedback ("can we include this message in failure mode instead?"). Three optional fields with defaults inflate the happy-path schema for an edge case; `isError` already exists at the protocol level for exactly this.
- (B) Raise `ToolError` with a structured message ← **chosen**.
- (C) Raise only for invalid-codes case, keep no-data-combo as a normal empty response — rejected: inconsistent within the same tool.
- (D) Echo back filter only (already done) — by itself isn't enough; eval evidence showed agents didn't reliably catch the mismatch.

---

## 2026-05-25 — `search_codes`: rename `k` → `top_k`, add `dim_id` scope

**What.** Two changes to [`tools/search.py`](tools/search.py) and one supporting change in [`statgpt/app/services/chat_facade.py`](../app/services/chat_facade.py):

1. **Rename `k` → `top_k`**. The breaking schema rename; old `k` arg now rejected by Pydantic. Same value, same semantics, more LLM-natural name.
2. **Add `dim_id: str | None` arg.** Optional dim-scope. When set:
   - `dataset_id` is required (dim ids are dataset-specific) — guard raises a `ToolError` with the fix.
   - `indicator` source is excluded (indicators aren't per-dim) — default `sources` drops to `["non_indicator", "special"]`; explicit `sources=["indicator", ...]` raises.
   - The dim filter is applied at the **vector-store metadata level** via a new optional `dimension_id` kwarg on `ChannelServiceFacade.search_non_indicator_dimensions_scored` → `vector_store.search_with_similarity_score(metadata_filters={...})`. Pre-filter, not post-filter — no over-fetch, cleanly efficient.
3. **Docstring caveat** on `dim_id`: "Score gap between rank 1 and rank 2 is the strongest signal — when rank 1 dominates, prefer it; when several rank similarly, the dim has multiple legitimate codes for the query." So the agent knows how to read the ranked output.

**Why.** Round-1 + round-2 friction patterns from [eval_subagent_findings.md](eval_subagent_findings.md):
- **`top_k` rename:** 2/5 → 3/5 agents tried `top_k` (LLM idiom from OpenAI/Anthropic APIs) and got Pydantic rejects on each round. Pure friction.
- **`dim_id` arg:** transcript analysis (round 1+2) found 3 separate agents tried `search_codes({dim_id: "COUNTRY"})` and `({dim_id: "L_CP_COUNTRY"})` to do scoped codelist lookups ("find Germany in WEO's COUNTRY", "find Japan in BIS's counterpart-country"). All rejected with no useful alternative — agents fell back to `sample_dim_values(n=-1)` (dumps the whole codelist) or post-filtering broad `search_codes` results. The use case is real and the implementation is small.

Concrete before/after on the BIS Japan query (`dataset_id="BIS:WS_LBS_D_PUB(1.0)"`, `query="Japan"`):

```
WITHOUT dim_id (top 5):                            WITH dim_id="L_CP_COUNTRY" (top 5):
  1.000  L_PARENT_CTY    JP    Japan                1.000  L_CP_COUNTRY    JP    Japan
  0.995  L_REP_CTY       JP    Japan                0.941  L_CP_COUNTRY    JM    Jamaica
  0.990  L_CP_COUNTRY    JP    Japan                0.935  L_CP_COUNTRY    JO    Jordan
  0.932  L_CP_COUNTRY    JM    Jamaica              0.918  L_CP_COUNTRY    JE    Jersey
  0.931  L_DENOM         JPY   Yen                  0.910  L_CP_COUNTRY    FJ    Fiji
```

The "same code repeated across 3 dims" noise is gone; the agent gets one unambiguous answer (`JP` at score 1.000 with a clear gap to rank 2).

**Cost / risk.**
- **Breaking schema change** for the rename. Acceptable here because mcp_lite is pre-v0, no production callers wired to the old arg name.
- **Facade signature change** (added kwarg `dimension_id: str | None = None`) — keyword-only with a default, so existing callers in `statgpt/app/chains/` are unaffected.
- **No new dependency.** The metadata filter goes through the existing `vector_store.search_with_similarity_score(metadata_filters=...)` path used by `search_special_dimension_scored` already.
- **`special` source still deferred** — the current implementation doesn't search special-dim values at all (PLAN.md decision). `dim_id` accepts `sources=["special"]` syntactically but is a no-op until the special-source search lands.

**Options considered.**
- (a) Add `dim_id` to `search_codes` only ← **chosen**
- (b) Pydantic alias `top_k` → `k` instead of rename — rejected: schema-visible name still picks one, doesn't actually fix the LLM's first-guess problem
- (c) Add `query` arg to `sample_dim_values` instead — rejected because it converges the signatures of `sample_dim_values` and `search_codes` into near-twins (user feedback: "they look the same now"). Keeping the two tools doing genuinely different jobs (probe vs. find-by-name) is cleaner.

---

## 2026-05-25 — Round-2 cross-dataset `search_codes` "timeout regression" diagnosed as non-issue

**What.** No code change. Diagnostic only — recorded here because the round-2 subagent eval flagged a 4/5-agent timeout regression on `search_codes` that was tracked as a top-priority item until proven false.

Reran the same 5 cross-dataset `search_codes` queries directly through `mcpjam` against the live server:
- Serial: 3.8s – 6.4s each.
- 5× parallel (simulating the 5-subagent fan-out): 5.4s – 6.1s each; 6s wall time.

Backend is healthy. The 180s "timeouts" the round-2 subagents saw came from their own mcpjam-side session state (process spawning, `mcp-remote` reconnect, or hung prior call holding the session), not from `search_codes`. The Austrian-query regression (agent picked `IMF.STA:PIP` over `BIS:WS_NA_SEC_DSS`) was a downstream effect: the agent abandoned `search_codes` mid-decision and fell back to `list_datasets` + manual eyeballing. Direct-call output for the same query shows `BIS:WS_NA_SEC_DSS` is correctly present at rank 2, so the agent would have picked it on a non-flaky run.

**Why this matters for the backlog.** Removes "diagnose latency regression" from the next-action list. Reprioritises: (a) make eval runs retry on subagent-side timeout, (b) defer ranking work until we see it cause a regression on a non-flaky run.

**Cost / risk.** None — diagnostic only. Full numbers and per-row top-K analysis are in [eval_subagent_findings.md](eval_subagent_findings.md) ("Direct-call sanity check" + "Latent ranking concern" sections).

---

## 2026-05-25 — Moved `execute_sdmx_query` to its own file

**What.** Pulled `execute_sdmx_query` + all its private helpers (`_expand_bare_year`, `_default_time_period_query`, `_auto_fill_dim_query`, related constants) out of [`tools/dataset.py`](tools/dataset.py) into a new module [`tools/data_query.py`](tools/data_query.py). Registered in [`tools/__init__.py`](tools/__init__.py) the same way as the other tool modules (`from . import data_query, ...  # noqa: F401`). Cleaned dataset.py's now-unused imports (`json`, `re`, `time_period_utils`, `Query`, `DataSetQuery`, `DimensionDataType`, `create_time_period_query_from`, `DataRequestStatus`, `ExecuteResult`, `TimePeriod`).

**Why.** `dataset.py` was at ~570 lines with five tools plus four helpers specifically for `execute_sdmx_query`, making the dataset-introspection tools harder to read. Split keeps each file focused: `dataset.py` for introspection (list, structure, sample, availability), `data_query.py` for the heavy data-fetch path.

**Cost / risk.** Pure file move; no behaviour change. Both files share `mcp_tools` via `_provider.py`, so tool registration just works. All 8 tools verified registered via in-process FastMCP client after the split.

---

## 2026-05-25 — `execute_sdmx_query` auto-fills unspecified non-time dims (statgpt-style)

**What.** Added helper `_auto_fill_dim_query(dim, dataset, availability)` in [`tools/dataset.py`](tools/dataset.py) that, for any dim the caller didn't pin, tries to set it automatically using the same three-step priority statgpt's chain uses ([SimpleQueryConstructor._set_dimension_query_from_default_or_available_values](../app/chains/data_query/query_constructor/simple.py)):

1. `dim_config.default_queries` per-dim default (filtered by availability if categorical)
2. `dataset.default_value_codes` ∩ availability (e.g. `_T`, `_Z` "total" markers)
3. ≤ `_AUTO_FILL_AVAILABILITY_K_LOW` (10) reachable values → use them all with `operator=ALL`

Wired into `execute_sdmx_query` after the user's dims + time-period default are applied: we identify unspecified dims, run one `availability_query` against the partial filter to learn what's reachable, then call `_auto_fill_dim_query` per dim. If no auto-fill is needed (caller pinned everything), the extra availability call is skipped.

**Why.** 4/5 subagents in the 2026-05-25 eval had to manually pin every required dim (sometimes 12+ per query). statgpt's chat-UI users never face this — the chain auto-fills "all/total" defaults and small availability sets. Bringing the same mechanism to the tool boundary cuts the agent's tool-call count by 6–10 per execute for BIS-style multi-dim datasets, and matches the dim-pinning behaviour an end-user sees in the chat UI.

**Cost / risk.**
- One extra `availability_query` SDMX call per `execute_sdmx_query` *when* unspecified dims exist. Not on the happy path where the caller pinned everything.
- Auto-fill is opinionated. To keep transparency the helper docstring spells out the priority order; the resulting query's `is_default=True` flags on auto-filled dims survive into `dataset.query(...)` logs.
- The original statgpt code has a `k_high=40` clarification band between "auto-fill" (≤10) and "give up" (>40). mcp_lite has no clarification flow, so I collapsed both into a single `k_low=10` cutoff — above that, we return None and the data layer surfaces "missing dimensions" the agent can recover from.
- `VirtualDimension` dims are skipped — they're not categorical and have no codelist to fill from.

---

## 2026-05-25 — `execute_sdmx_query` applies the dataset's configured default `time_period` when caller omits it

**What.** When `time_period=None`, `execute_sdmx_query` now reads `dataset.config.time_period_dimension.default_queries` (e.g. `{"values":["-5y","now"], "operator":"between"}` for gtdc channel datasets), resolves any relative references via `time_period_utils.get_relative_aware_time_period_query`, and appends a `DimensionQuery.from_default_query(...)` to the SDMX query. Helper `_default_time_period_query(dataset)` in [`tools/dataset.py`](tools/dataset.py:290) lifts the relevant slice of statgpt's `_apply_default_time_period_if_possible` ([statgpt/app/chains/data_query/query_builder/query/finalize_query.py:204](../app/chains/data_query/query_builder/query/finalize_query.py)). Tool docstring updated to declare this behaviour explicitly — agent knows where the default comes from and that `time_range_actual` in the response shows what was applied.

**Why.** 4/5 subagents in the 2026-05-25 eval omitted `time_period` on their first `execute_sdmx_query` attempt and got `Query is not ready: missing dimensions: ['TIME_PERIOD']`. The tool's old docstring claimed *"If null, the dataset's default time range applies"* — which was a lie: no code path applied any default. Investigation found that the dataset config DOES carry a per-dataset default (`defaultQueries: [{"values":["-5y","now"], "operator":"between"}]` everywhere in gtdc), and statgpt's pipeline reads it via `_apply_default_time_period_if_possible` in `FinalizeQueryStage` — but that stage was explicitly dropped during mcp_lite's primitive-decomposition (see PLAN.md's `[D] DROP` / `[C] CALLER` table). The error wasn't an SDMX requirement; it was us bypassing the chain's defaulting step without re-implementing it at the tool boundary.

**Cost / risk.** The default is **documented per-dataset in YAML and visible to the agent via the tool docstring** — same default the chat UI applies, same source. Not silent magic. If a dataset has no configured default, we don't invent one — `execute_sdmx_query` returns the data-layer's clean "missing dimensions: ['TIME_PERIOD']" error so the agent knows to supply one. The availability-overlap optimization from statgpt's version was deliberately not ported — adds latency for a corner case the agent can re-do with its own `availability_query` if it cares.

**Out of scope here.** The companion issue — `dataset_structure` doesn't surface `required: true` for the time dim — is still open. We'll handle that next so the agent has the structural signal even before reading the tool's docstring.

---

## 2026-05-25 — `dims` accepts scalar or list (both tools)

**What.** Widened the type of the `dims` arg on both `availability_query` and `execute_sdmx_query` from `dict[str, list[str]]` to `dict[str, str | list[str]]`. A scalar is treated as a one-element list. The widening is declared in the type annotation (so the JSON schema the LLM sees says "string or array of string"); the normalization to list happens explicitly at the top of each function with a comment pointing back to the type declaration.

**Why.** 4/5 subagents in the 2026-05-25 eval wrote scalar values on their first call to one or both tools and got Pydantic validation errors like `dims.FREQ  Input should be a valid list, input_value='Q', input_type=str`. SDMX URL syntax is one value per dim (`USA.NGDPDPC.A`), so a scalar is the natural mental encoding. Rejecting it required ~1 wasted call per agent per tool.

The list-only typing wasn't an SDMX or data-layer requirement — `DimensionQuery.values: list[str]` accepts single-element lists fine. The strict shape was inherited from MR !1097's `availability_query` and propagated to `execute_sdmx_query` per the PLAN's example syntax. Nothing downstream forces it.

**Cost / risk.** Honest union widening, not silent coercion — the LLM's view of the type now matches what actually works (the principle: schema = contract). Internal normalization runs once per call, no behavioural change for lists; scalars stop failing.

Options considered (in [eval_subagent_findings.md](eval_subagent_findings.md) §1):
- (A) keep `list[str]` typing, silently accept scalars → rejected as opaque
- (B) widen to `str | list[str]` and normalize internally → chosen
- (C) keep strict, improve error text → rejected (still wastes a round-trip)

---

## 2026-05-25 — `availability_query` arg rename: `partial_dims` → `dims`

**What.** Renamed the second argument of `availability_query` from `partial_dims` to `dims` ([`statgpt/mcp_lite/tools/dataset.py`](tools/dataset.py)), matching the corresponding arg on `execute_sdmx_query`. Schema field description in [`schemas.py`](schemas.py) and example in [`PLAN.md`](PLAN.md) updated to match.

**Why.** In the 2026-05-25 subagent eval (5 BIS+multidataset queries, see [`eval_subagent_findings.md`](eval_subagent_findings.md)), **5/5 independent agents** hit the inconsistency between `partial_dims` (availability) and `dims` (execute). Each agent burned ~1–3 calls on Pydantic validation errors like:

```
2 validation errors for call[availability_query]
  partial_dims  Missing required argument
  dims          Unexpected keyword argument
```

before retrying with the right name. Pure friction with no information return — both args refer to the same concept (a per-dim value filter), the naming distinction "partial" vs "full" reflected design intent that agents don't see.

**Cost / risk.** Breaking schema change. Acceptable here because:
- We're pre-v0; no production callers wired to the old name yet
- The agent's own MCP transport is JSON-RPC; nothing downstream is type-bound to the field name
- Three options were considered (in [eval_subagent_findings.md §1](eval_subagent_findings.md), restated as A/B/C in the discussion): (A) auto-coerce, (B) Pydantic alias, (C) rename. (C) wins because it picks one canonical name and stops having to maintain two ways to spell the same thing

**Out of scope (deferred).** The companion issue — both tools rejecting scalar values where a one-element list is expected — was NOT addressed here. Agents still trip on that (e.g. `dims={"FREQ": "Q"}` → `Input should be a valid list`). Documented in [`eval_subagent_findings.md` §1](eval_subagent_findings.md). If we touch it later, the candidates remain: scalar-coercing field_validator OR strict-as-now-but-better-error-text.

---

## 2026-05-22 — `execute_sdmx_query` time-period: auto-expand bare years

**What.** Helper `_expand_bare_year` in [`tools/dataset.py`](tools/dataset.py) expands `time_period.start="YYYY"` → `"YYYY-01-01"` and `time_period.end="YYYY"` → `"YYYY-12-31"`. Tool's `time_period` arg description spells out the ISO-date preference + the auto-expansion behaviour.

**Why.** Upstream SDMX-3.0 proxy at `statgpt-sdmx-proxy.aks.dev.dial.parts` has a bug: bare-year periods (`2022`) combined with `BETWEEN`/`EQ` operators return HTTP 200 with empty body, silently rejecting the query. Discovered while debugging IMF WEO failures — verified via raw curl that `ge:2022+le:2024` returns 0 bytes for WEO while `ge:2022-01-01+le:2024-12-31` returns 26KB of valid JSON.

**Cost / risk.** Workaround masks an upstream bug; should still be reported (and is documented in `backlog.md` here so we know to remove the patch when it's fixed). Auto-expand is opinionated — `YYYY` becomes a full calendar year, which is correct for annual data but slightly imprecise for sub-year queries.

---

## 2026-05-22 — `execute_sdmx_query` upstream-failure handling: warning instead of ToolError

**What.** Added `ExecuteResult.warning: str | None` field. When the underlying SDMX call returns `request_status != SUCCESS`, the tool now returns `row_count=0` with a diagnostic warning instead of raising `ToolError`. See [`tools/dataset.py`](tools/dataset.py) `execute_sdmx_query`.

**Why.** The proxy returns HTTP 200 + empty body for "no observations" queries; the SDMX-JSON reader treats that as parse-failure → `request_status=FAILED, parsing_status=NA`. Original behaviour raised a ToolError, which the agent couldn't act on. Reformulating as "empty result + diagnostic" lets the agent retry with different filters, fall back to `availability_query`, or widen the time period.

**Cost / risk.** Loses the strict "request failed = exception" semantic. Mitigation: the warning text names the upstream URL + parse/request status enums so the human/agent can still triage; a real network failure surfaces with the same flag but slightly different status combination.

---

## 2026-05-25 — Tool inventory checkpoint

After all edits above, the mcp_lite surface stands at:

| # | Tool | Args (required) | Status |
|---|---|---|---|
| 1 | `list_glossary_terms` | — | stable |
| 2 | `get_glossary_term` | `term` | stable |
| 3 | `list_datasets` | — | stable |
| 4 | `dataset_structure` | `dataset_id` | stable |
| 5 | `sample_dim_values` | `dataset_id`, `dim_id` (`limit` renamed from `n`) | changed today |
| 6 | `availability_query` | `dataset_id`, **`filter`** ← renamed from `dims` | changed today |
| 7 | `execute_sdmx_query` | `dataset_id`, **`selection`** ← renamed from `dims`; flat `time_start`/`time_end` | changed today |
| 8 | `search_indicators` | `query` (cross-dataset compound indicator search) | new today |
| 9 | `search_codes` | `dataset_id`, `query` (single-dataset dim-value search w/ fuzzy fallback) | restructured today |

---

## Open backlog (not implemented yet, ordered by impact)

These are pulled from [`eval_subagent_findings.md`](eval_subagent_findings.md) §§1–7. Each entry will get its own datestamp section here when it lands.

1. ~~**Scalar values in `dims`**~~ — done above (2026-05-25).
2. ~~**`TIME_PERIOD` required but not surfaced**~~ — mooted by `_default_time_period_query` auto-fill (2026-05-25).
3. **`execute_sdmx_query` missing-dims error → smarter recovery.** Mostly mooted by `_auto_fill_dim_query` (2026-05-25). Residual case: dim has no `default_queries`, no `default_value_codes`, AND >10 availability values. Still falls through to a "missing dimensions" error.
4. **`search_codes` ranking** (latent, deferred): PIP-style indicator floods top-K with permutations of the same indicator. Hasn't caused a regression on a non-flaky run. Revisit when it does.
5. ~~**`available: {}` clarification**~~ — done above (2026-05-25). Original hypothesis ("fully constrained") was empirically wrong; `{}` only means "no data". Now surfaced as `no_data` + `invalid_codes` + `message`.
6. **`sample_dim_values` batch variant** — currently singular `dim_id`, forces N round-trips when probing structure.
7. **Multi-dataset orchestration helper** — repeated structure+availability+execute per dataset is a real cost; v1 target.
8. ~~**`search_codes` arg-name friction (`top_k`, `dim_id`)**~~ — done above (2026-05-25).
9. **Eval harness: retry-on-timeout for subagent runs.** Round-2 saw 4/5 agents hit a 180s timeout on cross-dataset `search_codes` that direct-call testing proved was subagent-side flakiness, not a backend regression. Add a one-shot retry in the subagent prompt template before the next eval round.

## Decision log shortcuts

- See [`PLAN.md`](PLAN.md) for the original design, design decisions block, and architectural caveats (e.g. non-indicator hybrid asymmetry).
- See [`eval_subagent_findings.md`](eval_subagent_findings.md) for raw subagent-trace evidence behind these decisions.
