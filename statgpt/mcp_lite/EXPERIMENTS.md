# mcp_lite — broad-discovery report

`mcp_lite` exposes a StatGPT channel as a small set of **low-level, no-LLM SDMX
primitives** over MCP. Instead of the production app's coded search pipeline
(LLM normalise → hybrid retrieve → LLM rerank → finished selection), mcp_lite
hands the raw primitives to an **agent** that drives discovery, code resolution,
availability checks, and fetch itself.

This report covers: the tool surface, how mcp_lite differs from the production
search pipeline, the enhancements we made to the tools, and the evaluation of an
mcp_lite-driven agent against the production baseline on a 23-case multi-dataset
suite.

Detailed run-by-run history, configs, and operational notes live in the
**Appendix**.

---

## 1. MCP tool surface

Nine data primitives (plus an eval-only artifact writer). Each tool's own
description is the authoritative spec; one-liners:

| tool | purpose |
|------|---------|
| `list_datasets` | what datasets are in the channel |
| `dataset_structure` | dimensions of one dataset (id, type, codelist size) |
| `sample_dim_values` | peek / list the values of a non-indicator dim |
| `search_indicators` | **cross-dataset concept search** → `datasets[]` groups; each match carries a ready `dimensions` selection |
| `search_codes` | atomic dim-value codes (countries, sectors, instruments) within one dataset |
| `availability_query` | which dim values are **reachable** under a partial filter |
| `execute_sdmx_query` | fetch observations for a fully-resolved key |
| `list_glossary_terms` / `get_glossary_term` | channel-specific vocabulary |
| `write_data_query_artifact` | eval harness only — records the agent's final selection |

The agent's typical loop: `search_indicators` (discover candidate datasets) →
`search_codes` / `sample_dim_values` (resolve entity codes) → `availability_query`
(verify the combination exists) → `execute_sdmx_query` (fetch).

---

## 2. Architecture: production search vs mcp_lite

Both share the **same underlying hybrid retrieval** (ElasticSearch BM25 ∥
pgvector cosine, convex-combined). They differ in everything wrapped around it —
the production app does the reasoning in code; mcp_lite delegates it to the agent.

```
 PRODUCTION (statgpt-app, HybridSearcher)        mcp_lite (search_indicators + agent)
 ----------------------------------------        ------------------------------------
   user query                                      user query
       |                                               |
       v                                               v
   [LLM] normalise query                          search_indicators(query, top_k)
   (strip NER + time, split subjects)             (raw query, no LLM normalise)
       |                                               |
       v                                               v
   ES lexical pre-match → pick alpha              hybrid retrieval (BM25 ∥ pgvector,
       |                                          fixed alpha = 0.9)
       v                                               |
   hybrid retrieval (BM25 ∥ pgvector)                  v
       |                                          group by dataset_id -> datasets[]
       v                                          (NO rerank, NO availability filter)
   availability filter (drop unreachable)              |
       |                                               v
       v                                          ((  AGENT decides which groups   ))
   [LLM] rerank each candidate 0-3                ((  to keep, resolves codes,      ))
       |                                          ((  verifies, fetches            ))
       v                                               |
   per-dataset top-1 selection                         v
       |                                          search_codes / sample_dim_values
       v                                               v
   dataset_queries: dict[uuid, DimQuery]          availability_query
   (multi-dataset by construction)                     v
                                                   execute_sdmx_query
```

| step | production (HybridSearcher) | mcp_lite (`search_indicators` + agent) |
|------|------------------------------|-----------------------------------------|
| preprocessing | LLM normalises query (strips NER + time, splits subjects) | none — raw query straight to retrieval |
| retrieval | BM25 ∥ pgvector, **dynamic** α | BM25 ∥ pgvector, **fixed** α = 0.9 |
| availability filter | inline — unreachable indicators dropped | not in search; agent calls `availability_query` itself |
| rerank | dedicated LLM scoring 0–3 in batches | none — pure similarity, no LLM in search |
| selection unit | per-dataset top-1 by rerank | tool returns `datasets[]`; **agent** decides how many groups to keep |
| dim resolution | same rerank LLM, from candidate metadata | separate primitives (`search_codes`, `sample_dim_values`) |
| output | finished `dict[uuid, DimQuery]` | grouped matches; agent assembles selection downstream |

**Why it matters:** the production app's two LLM stages (normalise + rerank) and
availability filter do work that, in mcp_lite, the *agent* must do via prompting
+ tool calls. That's exactly why tool-description shape, argument names, output
structure, and agent methodology (skill / "research thoroughly") move the
mcp_lite numbers so much — the intelligence lives in the agent loop, not the
pipeline.

### 2.1 The hybrid retrieval, step by step

Both sides combine the **same two signals** over the same indicator index:
**ES BM25** (lexical — sharp on exact indicator names) and **pgvector cosine**
(semantic — catches paraphrases / synonyms), fused by a convex combination

```
score = α · sem_norm + (1 − α) · lex_norm
```

Everything *around* that one line is what differs.

**Production — `HybridSearcher` (LLM-in-the-loop, dynamic α):**

1. **Lexical pre-match → terminology hints.** A BM25 query with ES highlighting
   (`<em>` spans) is parsed token-by-token to extract phrases that *exactly*
   match an indicator's `primary_normalized` name ("good candidates") vs partial
   hits ("candidates") — i.e. known channel terminology in the query.
2. **LLM normalise.** An LLM rewrites the query: strips named entities (countries)
   and time periods, but is *forbidden* to drop or split the good-candidate
   phrases. ("What was India's GDP in 2020?" → `gdp`.)
3. **LLM split subjects.** A second LLM splits the normalised query into
   independent sub-queries (one concept each), searched in parallel.
4. **Plan α per sub-query.** `_query_planner` re-runs the lexical pre-match on the
   sub-query and picks α from the lexical signal:
   - **no** lexical hit → **α = 0.999** (fall back to ~pure semantic), double the
     candidate budgets;
   - strong *and broad* lexical hit (`primaries > max_candidates`, i.e. > 32) →
     **α = 0.8** (lean more on BM25), double budgets;
   - otherwise → **α = 0.9** (default blend).
5. **Retrieve + normalise.** BM25 (≤128) ∥ pgvector (≤64); each min-max normalised
   (sem from a theoretical floor of −1, lex from 0) before the α-blend.
6. **Availability filter.** Candidates whose dataset/dim-values aren't in the
   pre-computed availability set are dropped *here*, before any LLM rerank cost.
7. **Diversify + LLM rerank.** Survivors are bucketed per dataset and round-robined
   (so one dataset can't crowd out the rest), then scored by a **relevance LLM in
   batches of 32, 0–3**. Per dataset the max-scoring candidate(s) above a
   threshold are kept (`multi_/single_dataset_score_threshold`).
8. **Output:** a *finished* `dict[dataset_id → DimensionQuery]` —
   availability-checked, reranked, dim-pinned. No agent reasoning required.

**mcp_lite — `search_indicators` (no LLM, fixed α):**

1. **Raw query in.** No pre-match, no normalise, no subject split — the user's
   string (or the agent's paraphrase) goes straight to retrieval.
2. **Retrieve.** BM25 ∥ pgvector, each pulling `max(4·top_k, 50)` candidates;
   lex normalised as `score / lex_max`, sem as `(cos + 1) / (sem_max + 1)`.
3. **Combine at fixed α = 0.9** (`0.9·sem + 0.1·lex`) — semantic-leaning, matching
   the production *default* blend, but never re-planned per query.
4. **No availability filter, no rerank.** Top-`top_k` by blended score, bucketed
   into `datasets[]` groups, returned as-is.
5. **Agent does the rest** — picks which groups to keep, resolves codes via
   `search_codes`, checks `availability_query`, then `execute_sdmx_query`.

**Net effect.** Production runs two LLM calls (normalise, split) + an availability
pass + a batched rerank LLM *inside* retrieval, and adapts α to each sub-query.
mcp_lite ships the raw blended top-k and pushes all of that judgement — query
framing, which datasets matter, availability, dim-pinning — out to the agent
loop. That is why the agent-side levers (verbatim-first query, mandatory broaden,
the skill) move mcp_lite's numbers so much: they stand in for production's
normalise + rerank + diversify stages.

### 2.2 Resolving atomic codes — `search_codes`

`search_indicators` resolves the **compound indicator dims** (the *what* — already
pinned in each match's `dimensions`). The **atomic slicing dims** — country,
currency, sector, instrument, counterpart — are a different problem, handled by
`search_codes`. In production these never get a dedicated search: the rerank LLM
picks them out of candidate metadata. mcp_lite exposes them as an explicit
primitive the agent drives.

It differs from `search_indicators` in three ways: it is **semantic-only** (pure
pgvector cosine — no BM25, no α blend), **scoped to one dataset** (codes are
dataset-specific: `USA` ≠ `US`), and it fans across **two vector stores** — the
non-indicator-dim store and a per-dim **special-dim** store. There is no lexical
side because dim-value *labels* are short atomic strings ("Germany", "US dollar",
"Banks, total") where embedding similarity (synonyms, abbreviations, translations)
carries essentially all the signal.

```
 search_codes(dataset_id, query, dim_id?, top_k)
        |
        v
   classify dim_id  (only when caller pins one)
     ├─ indicator → ToolError: "use search_indicators, read dimensions.<id>"
     ├─ time      → ToolError: "filter via time_start / time_end"
     └─ non_indicator | special | (dim_id null) → proceed
        |
        +─────────────────────────────┬──────────────────────────────+
        v                             v
   NON-INDICATOR store          SPECIAL-DIM stores  (one per special dim)
   pgvector cosine,             pgvector cosine, keyed by dim_id,
   k = 2·top_k                  k = 2·top_k each
   (runs if dim_id null         (runs if dim_id null → all special dims,
    or dim is non_indicator)     or dim_id is that special dim)
        |                             |
        v                             v
   normalise (cos+1)/(max+1)    normalise (cos+1)/(max+1)
   per store                    per store          ← same semantic map as §2.1
        |                             |
        +──────────────┬──────────────+
                       v
        keep only rows for THIS dataset's entity_id
                       v
        dedup by (dim_id, code), keep max score
                       v
        sort by score desc → take top_k
                       v
        matches[]: { source, dim_id, code, name, score }
```

1. **Classify first (when `dim_id` is pinned).** An indicator dim is redirected to
   `search_indicators`; the time dim to `time_start/time_end`; a non-categorical
   dim is rejected — each with a concrete recovery hint, *before* any vector call.
2. **Fan out.** With `dim_id` **null**, both paths run — every non-indicator dim
   *and* every special dim — which is how you discover **which dim holds a
   concept** ("Japan" may live in `L_CP_COUNTRY`, `L_REP_CTY`, or `L_PARENT_CTY`).
   With `dim_id` set, only the matching path/dim runs.
3. **Score + normalise.** Each store pulls `2·top_k`; scores are min-maxed with the
   **same semantic map as `search_indicators`** — `(cos + 1)/(max + 1)`, floor −1,
   per-query top as ceiling. (No lexical term, so no α and no convex combination.)
4. **Merge + cut.** Results from both stores are filtered to the dataset, deduped
   by `(dim_id, code)` keeping the max score, sorted, truncated to `top_k`.

**Reading the result:** the **score gap between rank 1 and rank 2** is the signal —
a dominant rank 1 means one obvious code; several near-ties mean the dim genuinely
has multiple legitimate codes for the query (e.g. "banks" → several sector codes).

**How it differs from production.** The retrieval engine is *the same* — mcp_lite
calls the exact same facade method (`search_non_indicator_dimensions_scored`) over
the same semantic vector store; both sides are pure cosine here (production never
runs BM25 on these dims either). What differs is, again, the wrapper around it:

| step | production | mcp_lite (`search_codes`) |
|------|------------|----------------------------|
| what gets searched | an **LLM NER step** extracts entities from the query (countries, sectors, …); the normaliser has already *stripped* them out of the indicator query | the **agent** decides what to look up and passes the raw concept ("Germany", "EUR") |
| query per entity | one semantic search **per extracted entity**, seeded by NER | one call per concept the agent chooses; `dim_id=null` fans across all dims at once |
| filtering | "Strong Non-Indicators" + **availability filter** + split-by-dataset, all coded | none — raw scored matches; the agent reads the rank-1-vs-2 gap and picks |
| who maps entity→dim | the coded pipeline (NER type → dim), no model judgement at call time | the agent, from the returned `(dim_id, code)` rows |

So production and mcp_lite split the *same* indicator-vs-non-indicator boundary the
same way (compound dims → hybrid indicator search; atomic dims → semantic dim-value
search). The difference is **who drives the atomic side**: production extracts the
entities with an NER LLM and resolves them in code (availability-filtered), whereas
mcp_lite hands the agent a raw, single-dataset, scored lookup and lets it choose the
search terms and the winning codes — the same "intelligence-in-the-agent-loop"
trade as §2.1.

---

## 3. Enhancements

Three classes of change, all driven by observed agent friction in eval traces.

### 3.1 Tool argument names — LLM-natural and self-documenting

Early evals showed agents repeatedly guessing the "natural" argument name and
hitting Pydantic rejects (wrong-arg-name friction: **14 misfires per 30 calls**).
We renamed args toward the LLM's natural guess / SDMX-native vocabulary:

NOTE: check with sdmx standard!!
NOTE: we can add named entites to description
| tool | before | after | friction before fix |
|------|--------|-------|---------------------|
| `search_indicators` | `k` | `top_k` | 2–3/5 agents |
| `availability_query` | `dims` | `filter` | 7/10 agents missed |
| `execute_sdmx_query` (key) | `dims` | `selection` (SDMX-native; distinct from availability's `filter`) | — |
| `execute_sdmx_query` (time) | nested `time_period: {start, end}` | flat `time_start`, `time_end` | 4/10 agents missed |
| `sample_dim_values` (size) | `n` | `limit` | 3/10 agents missed |

(`availability_query`'s key arg had churned earlier too: `dimension_queries` →
`partial_dims` → `dims` → `filter`.) Result: wrong-arg friction dropped from
**14/30 → 1/30** across an eval batch, and arg descriptions were rewritten tight
+ explanatory. `selection` vs `filter` is a deliberate split — both are
`dim_id -> code|list`, but "selection" reads as the SDMX key you fetch while
"filter" reads as the partial constraint you probe reachability with.

### 3.2 `search_indicators` output model — flat list → grouped by dataset

Multi-dataset coverage is the whole game on broad questions, so we changed the
output from a flat score-sorted list (coverage *implicit* — the agent had to
notice `dataset_id` varied) to **groups bucketed by dataset** (coverage
*structural* — the agent must read `datasets[]` to consume the result at all).

**Before** — flat `matches[]`, `dataset_id` repeated per row:
```json
{ "query": "CPI", "matches": [
  {"dataset_id": "IMF.STA:NSDP(7.0.0)", "name": "Consumer prices …", "score": 0.999, "dimensions": {…}},
  {"dataset_id": "IMF.STA:CPI(5.0.0)",  "name": "CPI, Communication …", "score": 0.998, "dimensions": {…}},
  "… 8 more, all IMF.STA:CPI — same dataset dominates the list"
]}
```

**After** — `datasets[]` groups sorted by `best_score`, `dataset_id` lifted out:
```json
{ "query": "CPI", "n_total_matches": 10, "datasets": [
  {"dataset_id": "IMF.STA:NSDP(7.0.0)", "best_score": 0.999, "matches": [ … ]},
  {"dataset_id": "IMF.STA:CPI(5.0.0)",  "best_score": 0.998, "matches": [ … ]}
]}
```

Each `matches[i].dimensions` is a ready selection for `execute_sdmx_query`. The
docstring also states plainly that *multi-dataset answers are the norm* and a
high score on group 2+ is a different lens, not a duplicate.

### 3.3 Error handling & actionable notifications

Empty/failed responses used to be opaque — the agent couldn't tell *why* nothing
came back, so it silently dropped valid datasets. We made the failure paths
**typed and actionable** on both query tools.

**(a) `availability_query` — distinguish bad codes from empty-but-valid.**
A zero-result filter now raises one of two specific `ToolError`s instead of a
bare empty/failure:
```
# some pinned code isn't in the dataset's codelist
Filter selects no observations: codes not in the dataset's codelist: ['B1GQ_V_USD'].
Use `search_codes(dataset_id=…, dim_id=…)` or `sample_dim_values` to find valid replacements.

# every code is valid, but the combination has no data
Filter selects no observations, though all codes are valid for the dataset.
Try widening the filter or removing one dim.
```
The agent now knows whether to **fix a code** (first case) or **relax the
combination** (second) — rather than concluding "this dataset is empty." Eval
traces showed agents explicitly recovering from the second message.

**(b) `execute_sdmx_query` — empty-result pin diagnosis.** A
`search_indicators` match gives indicator-dim pins (e.g. `INDICATOR=B1GQ_V_USD`);
`availability_query` passes (that pair is reachable *somewhere*); but the full
joint key returns 0 rows.

Before:
```json
{"row_count": 0, "warning": null}
```
→ agent concludes "FSIC has no India GDP" and drops a real truth dataset.

Now — it names the culprit pin and suggests prefix-ranked reachable alternatives:
```
0 rows returned. Pin mismatch detected — these values have no data under your
anchor filter: INDICATOR=['B1GQ_V_USD'] → not reachable;
try one of ['B1GQ_V_XDC', 'AQ12_CFSI_PT', …]. Retry with a reachable value.
```
→ agent swaps `USD → XDC`, refetches, gets 4 rows. (`B1GQ_V_XDC` ranks first as
it shares the longest prefix with the bad pin.) The probe distinguishes a
**single-pin mismatch** from a **joint mismatch** (each pin reachable alone,
combination empty) and words the warning accordingly; it runs only on
`row_count == 0` — zero cost on the happy path. The culprit dim and suggestions
come from **live `availability_query` re-probes** at diagnosis time, not a canned
string — see the three-level escalation and code line-refs in
[Appendix D](#d-execute_sdmx_query-empty-result-diagnostic--full-design).

**Still open — availability vs upstream-failure.** Both messages above assume the
empty result is *real*. They can't yet distinguish a genuine empty from an
upstream **auth/transport failure** (a 401 episode under parallel load made an
agent drop valid datasets it simply couldn't verify). Planned fix: surface a
typed error on 401/non-2xx so the agent retries or keeps the candidate instead of
reading it as "no data."

---

## 4. Evaluation — 23-case multi-dataset suite

23 broad questions ("What is the CPI of Brazil?", "trade balance of the US?",
…), each with a ground-truth set of every channel dataset that materially answers
it (11 original macro concepts + 12 added across BIS effective-exchange-rate,
IMF NSDP, IMF CPI). Scored by the **statgpt-eval pipeline** against cleaned
targets (unjustified dimension pins removed, factual mismatches fixed).

### 4.1 Accuracy vs production baseline

**Dataset-level** (did it pick the right datasets) and **dimension-value level**
(did it recover the exact dim pins — the strict framework metric) shown together:

| run | model | DS Rec % | DS Prec % | Recall % | Indic R % | NonInd R % | Prec Soft % | Prec Hard % |
|-----|-------|---------:|----------:|---------:|----------:|-----------:|------------:|------------:|
| production baseline | gpt-4.1 (DIAL) | 94.6 | 80.3 | 87.3 | 82.4 | 95.5 | 84.0 | 53.8 |
| **mcp sonnet — RT + skill** | claude-sonnet-4.6 | **96.4** | 87.1 | **90.3** | 85.4 | **97.3** | 92.2 | 66.3 |
| mcp sonnet — RT, no skill | claude-sonnet-4.6 | 83.9 | 90.4 | 83.9 | 80.2 | 89.1 | 89.9 | 70.5 |
| mcp sonnet — skill, no RT | claude-sonnet-4.6 | 87.5 | 90.7 | 82.3 | 79.5 | 87.8 | 92.7 | 70.8 |
| **mcp sonnet — skill, effort=high, no RT** | claude-sonnet-4.6 | 95.2 | 91.2 | 90.9 | **89.4** | 94.1 | **98.7** | **75.9** |
| mcp opus — skill, no RT | claude-opus-4.8 | 76.8 | **95.6** | 77.1 | 78.0 | 76.4 | **94.8** | 71.1 |
| **mcp opus — skill, effort=high, no RT** | claude-opus-4.8 | 87.2 | 94.9 | 82.0 | 81.1 | 84.3 | 96.8 | **81.9** |

*Metrics:* **DS Rec / DS Prec** = dataset-level recall/precision (right datasets
picked; ignores dim-value correctness). **Recall** = dimension-value pins
recovered — strict, hence lower than DS Rec since it also checks the pins inside
each dataset; **Indic/NonInd R** split it by indicator vs non-indicator dims.
**Prec Soft / Hard** = partial-credit vs exact-match precision (Hard penalises any
over-pinned dim). "RT" = "Research thoroughly." suffix added to user query; "skill" = the
`using-statgpt-mcp-lite` methodology skill.


### 4.2 Cost (per query, avg over 23 cases)

| config | avg wall-time | avg tokens | avg tool calls |
|--------|--------------:|-----------:|---------------:|
| production baseline (gpt-4.1) | 74s | **13.6k** | n/a (coded pipeline) |
| sonnet RT + skill | 124s | 56.6k | 16.3 |
| sonnet skill | 135s | 50.1k | 17.5 |
| sonnet RT, no-skill | 103s | 29.6k | 12.1 |
| sonnet skill, effort=high | 89s | 49.7k | 13.5 |
| opus skill | 108s | 57.0k | 11.3 |
| opus skill, effort=high | 74s | 52.3k | 11.7 |

### 4.3 Single-family depth check — BIS data-query (41 queries)

A separate eval on **41 narrower BIS-family queries**. These are mostly **single-target** (one right dataset + a specific dim
key), so they test *depth* — exact dim-pin resolution — rather than multi-dataset
*breadth*. Re-run sonnet + skill (no RT) after the skill/tool-description changes.

| run | n | Recall % | Indic R % | NonInd R % | Prec Soft % | Prec Hard % | 1+ unexp ds |
|-----|--:|---------:|----------:|-----------:|------------:|------------:|------------:|
| **mcp_lite (sonnet + skill)** | 41 | **84.5** | **96.3** | **68.8** | **98.1** | **69.9** | 17% (7/41) |
| statgpt-app (gpt-4.1) | 41 | 55.8 | 54.6 | 57.1 | 86.8 | 32.5 | 49% (20/41) |

**Artifact-corrected (honest comparison).** Both runs omit the same proven
single-valued/auto-filled total codes (`COUNTERPART_SECTOR:S1`, `CUST_BREAKDOWN:_T`
— data-identical, §F.1). Counting those as covered **symmetrically**:

| run | NonInd R: scored → corrected | All-dim Recall: scored → corrected |
|-----|:----------------------------:|:----------------------------------:|
| **mcp_lite (sonnet + skill)** | 68.8 → **90.9** | 84.5 → **94.3** |
| statgpt-app (gpt-4.1) | 57.1 → **65.4** | 55.8 → **59.6** |

The correction **widens** mcp_lite's lead (NonInd gap +11.7 → **+25.5pp**), because
nearly all mcp_lite's non-indicator misses were the harmless total convention,
whereas most of the baseline's are *real* (`L_CP_COUNTRY` missing the named country
`DE`/`GB`/`JP`/`US`, `UNIT_MEASURE=EUR`, `FREQ=Q` — wrong/missing slices that change
the data). See §F.1.

**Cost (per query, avg over 41 cases):**

| config | avg wall-time | avg tokens | avg tool calls | avg cost |
|--------|--------------:|-----------:|---------------:|---------:|
| mcp_lite (sonnet + skill) | 103s | 55.1k | 13.6 | — |
| statgpt-app (gpt-4.1) | 62s | **13.1k** | n/a (coded pipeline) | $0.029 |


> **Scored recall understates real recall here.** 26/41 cases scored < 1.0, but
> **~20 are a scoring-convention artifact, not wrong data**: the agent omits
> redundant "total" structural codes (`COUNTERPART_SECTOR=S1`, `CUST_BREAKDOWN=_T`)
> that are single-valued under the rest of the key and that `execute_sdmx_query`
> **auto-fills** anyway — verified 25/25 data-identical (51 rows with vs without
> the pins). That gap is the bulk of the 68.8% non-indicator recall.
> **Data-equivalent recall is ~95%+**; only ~5 cases are real agent errors (one
> wrong-dataset pick, four single-dim country/variant mispins). Full breakdown +
> per-case IDs in **Appendix F**.

Run dirs: mcp_lite `eval_runs/20260530.claude.mcplite.gtdc.bis41.sonnet-skill/jobs`
(scored offline against the `statgpt-gtdc` fixture, report
`data/evals/…/eval.20260531-095241.…xlsx`); baseline
`~/Downloads/20260416-142239.gtdc.data-query-bis`.
---

# Appendix

## A. Metric definitions

- **Recall** — fraction of target dimension-value pins recovered across selected datasets (framework scoring). The in-doc "dataset-presence recall" (Appendix B) is coarser: did the agent pick the right dataset URNs, ignoring dim-value correctness.
- **Indic R / NonInd R** — recall restricted to indicator-dim pins vs non-indicator dims (COUNTRY, FREQUENCY, UNIT, …).
- **Prec Soft / Prec Hard** — partial-credit vs strict exact-match precision (hard penalises any extra/over-pinned dim).
- **avg wall-time** — per-query subagent duration (`response_time` patched into artifacts from completion notifications).
- **avg tokens** — total tokens/query (DIAL `usage` for the baseline; subagent harness counter for mcp_lite).

## B. Evaluation history

### B.1 The agent-steering series (11-test set, framework-scored on cleaned targets)

How the mcp_lite agent improved as we changed tool descriptions and agent
methodology. Truth-set total dropped 37 → 29 datasets after dim-pruning, so these
are not comparable to the pre-cleanup numbers.

| run | n | Recall % | Indic R % | NonInd R % | Prec Soft % | Prec Hard % |
|-----|--:|---------:|----------:|-----------:|------------:|------------:|
| R1 baseline (gpt-4.1) | 11 | 70.0 | 62.3 | 85.8 | 78.0 | 40.7 |
| R2 opus, no hint | 11 | 47.6 | 44.8 | 52.6 | 92.4 | 70.5 |
| R3a opus + hint | 11 | 57.2 | 54.0 | 62.7 | 92.4 | 83.3 |
| R3b sonnet + hint | 11 | 60.0 | 55.4 | 67.3 | 92.9 | 85.0 |
| R3c haiku + hint | 11 | 39.3 | 37.0 | 43.5 | 92.5 | 72.7 |
| R4 sonnet grouped+big-hint | 11 | 69.2 | 63.3 | 79.4 | 92.0 | 82.7 |
| **R5 R4 + research-thorough** | 11 | **78.7** | 74.9 | 86.5 | 89.0 | 64.4 |
| BIS WS_EER set (mcp_lite) | 42 | 82.4 | 91.0 | 71.3 | 97.8 | 72.4 |

R2 (flat output, no hint) → R4 (grouped output + multi-dataset hint) → R5
(+ "research thoroughly") tracks the §3.2 output change and the agent-prompting
levers. Test-cleanup later lifted every mcp_lite run 10–19pp.

### B.2 Headline series — dataset-presence scoring (29-truth, 11-test)

Coarser in-doc metric (dataset URN presence only) used while iterating; kept for
the cost/time figures captured per run.

| setup | model | recall | precision | avg time | avg tok |
|-------|-------|--------|-----------|----------|---------|
| R1 statgpt-app | gpt-4.1 | 82.8% | 80.0% | 86s | 12.5k |
| R2 mcp_lite | opus-4.7 | 37.9% | 100% | 106s | - |
| R3 mcp_lite | opus-4.7 | 51.7% | 100% | 104s | - |
| R3 mcp_lite | sonnet-4.6 | 55.2% | 94.1% | 103s | - |
| R3 mcp_lite | haiku-4.5 | 34.5% | 90.9% | 117s | - |
| R4 native | sonnet-4.6 | 65.5% | 90.5% | 68s | 17.4k |
| R5 native +RT | sonnet-4.6 | 82.8% | 82.8% | 87s | 24.3k |
| R6 native +skill | sonnet-4.6 | 69.0% | 90.9% | 74s | 31.4k |
| R7 verbatim-first | sonnet-4.6 | 75.9% | 88.0% | 132s | 53k |
| R8 verbatim+broaden | sonnet-4.6 | 79.3% | 88.5% | ~100s | 49k |
| R8 verbatim+broaden | opus-4.8 | 79.3% | 92.0% | ~90s | 59k |

### B.3 md23 dataset-presence scoring (56-truth, 23-test)

Coarse dataset-URN-presence metric on the full 23-case set (vs the framework
scoring in §4.1, which also checks dimension-value correctness — hence lower
framework recall, e.g. 90.3 vs 96.4 for the best config):

| config | model | recall (of 56) | precision |
|--------|-------|---------------:|----------:|
| mcp sonnet RT+skill | sonnet-4.6 | 96.4% | 87.1% |
| statgpt-app (data-42) | gpt-4.1 | 94.6% | 80.3% |
| mcp sonnet skill | sonnet-4.6 | 87.5% | 90.7% |
| mcp sonnet RT no-skill | sonnet-4.6 | 83.9% | 90.4% |
| mcp opus skill (clean) | opus-4.8 | 76.8% | 95.6% |

### B.4 Apples-to-apples on the 15 gtdc cases (partial baseline data-41)

The earlier data-41 baseline only ran the 15 gtdc-folder cases. Scored with the
mcp_lite configs on that same 15-case / 37-truth subset (dataset-presence):
mcp sonnet RT+skill **94.6%/85.4%** vs statgpt-app **81.1%/73.2%** — mcp_lite
ahead on both. (Superseded by the full-23 data-42 baseline in §4.)

## C. Run configurations

- **R1 / baseline** — production LangChain pipeline `statgpt/app/chains/data_query/`; gpt-4.1 via DIAL. Dirs: `data-40` (11-test), `~/Downloads/data-41` (15 gtdc), `~/Downloads/data-42` (full 23).
- **R2** — mcp_lite, flat `search_indicators` output, no hint, mcpjam transport, opus-4.7.
- **R3** — flat output + multi-dataset hint, all 3 Claude tiers, mcpjam.
- **R4** — native MCP (project `.mcp.json`), grouped output + big hint, `mcp-data-query` agent (MCP-tools-only, no filesystem → no ground-truth leak), sonnet, verbatim query.
- **R5** — R4 + "Research thoroughly." suffix.
- **R6** — R4 + `using-statgpt-mcp-lite` skill.
- **R7** — skill rewritten to "verbatim user question first" (no top_k override).
- **R8** — skill makes broadening mandatory (verbatim + `top_k=100` methodology-paraphrase follow-up).
- **md23 configs** — run dirs `eval_runs/20260529.claude.mcplite.gtdc.broad-discovery.md23.{sonnet-rt,sonnet-skill,sonnet-rt-noskill,opus-skill}/jobs`; the RT-no-skill config uses `mcp-data-query-noskill` (Skill tool removed — see D.3). xlsx reports in statgpt-new `data/evals/…md23.*/eval.*.xlsx`.

## D. `execute_sdmx_query` empty-result diagnostic — full design

Implemented in [tools/data_query.py](tools/data_query.py): `_diagnose_empty_result`
([L208-301](tools/data_query.py#L208-L301)) + `_smart_sample`
([L172-205](tools/data_query.py#L172-L205)), invoked from the execute call site
([L439-444](tools/data_query.py#L439-L444)). Runs **only** when `row_count == 0`
(zero cost on the happy path); fires in both the upstream-empty-200 and the
genuinely-empty-result-set branches.

**Nothing here is a canned message.** The warning *scaffolding* ("0 rows
returned. Pin mismatch detected …") is static, but the culprit dim, the offending
value, and the suggested alternatives are all derived from **live
`availability_query` re-probes** run at diagnosis time. The diagnostic is a
three-level escalation:

**Setup.** Partition the pinned dims: **anchors** = neither indicator nor special
(COUNTRY, REF_AREA, TIME_PERIOD); **relaxable** = indicator ∪ special dims. Bail
out (return `None`, no warning) if either set is empty.

1. **Single-pin probe** ([L231-258](tools/data_query.py#L231-L258)). Re-run
   `availability_query` with the **anchors only**. For each relaxable pin, compare
   its pinned value(s) to the reachable set returned by that probe; if
   `pinned & reachable` is empty the value is individually unreachable under the
   anchor filter → emit `Pin mismatch detected` naming that `dim=value` plus
   `_smart_sample` suggestions. (This is the common case: e.g. `INDICATOR=B1GQ_V_USD`
   when the dataset only carries `…_XDC` for that country.)
2. **Joint-pin probe** ([L264-292](tools/data_query.py#L264-L292)). If every pin
   is individually reachable, drop **exactly one** relaxable pin at a time, re-run
   `availability_query` on the rest, and check whether the dropped pin's value is
   reachable *given the others*. A value reachable alone but not under the joint of
   the rest is joint-incompatible → `Joint-pin mismatch` warning. Isolates which
   single dim breaks the combination.
3. **Fallback** ([L294-299](tools/data_query.py#L294-L299)). Both probes pass →
   "each pin individually reachable & pairwise-compatible, but the full
   combination has no observations; widen `time_period` or drop a pin." The only
   purely-static branch — there is nothing left to localise.

`_smart_sample` ranks suggestions by **longest shared prefix** with the bad pin:
for each pinned value it tries progressively shorter prefixes, collecting reachable
values that start with that prefix before padding the rest alphabetically (so
`B1GQ_V_USD` surfaces `B1GQ_V_XDC` / `B1GQ_R_USD` first, not alphabetical `AQ12_*`
neighbours), capped at 20. The whole function is wrapped in a `try/except` that
returns `None` on any error — a failed diagnosis never breaks the fetch response.

**Measured impact (R6 strong + diag, 11-test):** no recall lift over R6 strong
(72.4% both) — because the strengthened-skill agents weren't *attempting* the
borderline datasets, so they never hit the empty-fetch path. The diagnostic helps
an agent that tries-and-fails, not one that doesn't try; the RT lever (which
pushes attempts) is what makes it pay off.

## E. Operational notes & caveats

- **ElasticSearch OOM under parallel load.** 11–23 concurrent agents pegged ES (`-Xmx2g`/`mem_limit 3g`) and it was OOM-killed; `search_indicators` silently degraded to semantic-only (BM25 half gone), dropping lexical-match datasets (NSDP, BIS-LBS). Fix: VM 8→12GB, ES `-Xmx3g`/`mem_limit 5g` ([docker-compose.yml](../../docker-compose.yml)). **Before trusting a run, verify ES `:9200` + DIAL-core `:8080/health` are up and `search_indicators` returns a lexical-heavy dataset (e.g. NSDP for "unemployment rate").**
- **opus 401-degraded vs clean rerun.** The first md23 opus run hit upstream 401s on `availability_query`/`execute` and scored 82.1%/90.2% (inflated — kept unverifiable candidates). The clean rerun scored **76.8%/95.6%**. The 401 run is archived at `…md23.opus-skill/jobs_401degraded/`.
- **Skill auto-invocation.** The `mcp-data-query` agent lists `Skill` in its allowlist, and `using-statgpt-mcp-lite` auto-invokes on every statistics question (23/23 agents, even with no instruction). A true skill-free baseline needs an agent **without** the Skill tool — `.claude/agents/mcp-data-query-noskill.md`.
- **Two scorings, don't conflate.** Dataset-presence (Appendix B) vs framework per-dimension (§4). Framework recall is lower because it also scores dimension-value correctness, not just dataset presence.
- **Timing/tokens were patched post-hoc** into md23 artifacts from session completion-notifications (`response_time`, `total_tokens`, `agent_wall_time_ms`, `agent_tool_uses`) so the eval pipeline surfaces them.

## F. Failure-mode analysis — scored misses vs real misses

Neither suite's sub-100% recall is mostly real. Both carry a **scoring-convention
artifact**, and they are *different* artifacts — investigated by probing the live
channel and by re-scoring code-only.

**Root cause of the MD23 artifact: `Term` equality is id + name.** The framework's
`Term.__eq__` ([statgpt_eval/schemas/test_case.py:67-75](../../../statgpt-new/statgpt_eval/schemas/test_case.py))
requires **both** the code *and* the display label to match (by design — "to
detect codelist name changes"). So a correct code with a differently-worded label
scores as a phantom **FN + FP**.

| suite | artifact | shape | detected by | scored → adjusted recall |
|-------|----------|-------|-------------|--------------------------|
| BIS (§F.1)  | omitted auto-filled total codes (`S1`/`_T`) | FN-only | availability probe (data-identical) | 84.5% → **~95%** |
| MD23 (§F.2) | correct code, different label | FN+FP | re-score code-only | 90.3% → **94.5%** |

### F.1 BIS — omitted auto-filled total codes

26 of 41 BIS cases scored recall < 1.0 (§4.3), but most "misses" are a
**scoring-convention artifact, not wrong data.** Investigated by probing the live
channel under each agent's own selection.

**The dominant pattern.** BIS false-negatives are almost all **non-indicator**
(49 of 56 FN), concentrated on two "total / catch-all" structural codes the agent
left *unpinned*: `COUNTERPART_SECTOR=S1` (Total economy, missed 20×) and
`CUST_BREAKDOWN=_T` (Total, missed 18×) — **38 of 56 FN (68%)**. Over-pinning is
~0, so these aren't wrong codes; the dim was simply omitted.

**Why omission is harmless — proven.** For every WS_NA_SEC_DSS case that "missed"
`S1`/`_T`, an `availability_query` under the agent's *own* filter shows the dim is
**single-valued**: reachable `COUNTERPART_SECTOR = [S1]`, `CUST_BREAKDOWN = [_T]`.
A direct row-count check (Austrian financial-corps case) returned **51 rows with
the pins, 51 without** — byte-identical. Across all 25 testable instances: **25/25
artifact, 0 real.**

**Mechanism.** `execute_sdmx_query` **auto-fills** unpinned dims
([data_query.py:386-405](tools/data_query.py#L386-L405) → `_auto_fill_dim_query`):
per-dim default → dataset-wide default codes (`_T`/`_Z` totals) → low-cardinality
take-all. So the SDMX key that *actually runs* includes `S1`/`_T` even when the
agent omits them. The recall metric scores the agent's recorded
`dimension_id_to_name` (which omits them), **not** the resolved query — hence the
phantom miss. The data the agent reasoned over was complete.

**Breakdown of the 26 sub-100% cases:**

| class | n | cases |
|-------|--:|-------|
| **pure `S1`/`_T` artifact** (data-identical) | 17 | `c81f30b4`, `b5c68409`, `c7a1b2b2`, `875fb384`, `31dedbb4`, `2332caca`, `648c5c44`, `e4f1d19f`, `e0998969`, `306e9e6d`, `4cf1d576`, `00b4fc2e`, `f22cea96`, `ff0498e0`, `6a977b00`, `236053ec`, `88f5a726` |
| **artifact + one real extra** | 4 | `becf09f1` (`REF_SECTOR` S12T vs S122), `8711a685` (`CURRENCY_DENOM` _T vs XDC; FREQ), `ea66bf5e` (FREQ=Q), `05a60b05` (FREQ=A) |
| **real failure** | 5 | `9d5db6b9` (wrong dataset → 0.0), `3781b7ec` (missed IMF.STA:EER variant), `bb80d342` (`L_CP_COUNTRY` 5J vs US), `07d3f556` (`L_PARENT_CTY` 5J vs CH), `d77950e8` (`L_CURR_TYPE=F`) |

**Implication.** BIS's **data-equivalent recall is ~95%+**, not the scored 84.5% —
the non-indicator-recall gap (68.8%) is almost entirely the redundant-total
convention. Three ways to close it, cheapest first: (1) record the *resolved*
execute key (which already contains the auto-filled `S1`/`_T`) in the artifact
instead of the agent's hand-passed dims; (2) a skill line to pin total codes
explicitly; (3) a scoring-convention change (an omitted dim whose reachable set is
a single value, or contains the truth value, counts as covered). Only the ~5 real
failures reflect actual agent error — 4 are single-dim country/variant mispins,
1 is a wrong-dataset pick.

**Symmetric correction vs the baseline (honest comparison).** The statgpt-app
baseline omits the *same* `S1`/`_T` totals, so the correction must be applied to
both. Counting those omissions as covered for each run:

| run | NonInd R: scored → corrected | All-dim Recall: scored → corrected |
|-----|:----------------------------:|:----------------------------------:|
| mcp_lite (sonnet + skill) | 68.8 → **90.9** | 84.5 → **94.3** |
| statgpt-app (gpt-4.1) | 57.1 → **65.4** | 55.8 → **59.6** |

mcp_lite jumps +22.1pp NonInd (almost all its NI misses were the total
convention); the baseline only +8.3pp — because the rest of *its* NI misses are
**real**: `L_CP_COUNTRY` missing the named counterpart country (`DE`/`GB`/`JP`/`US`),
`UNIT_MEASURE=EUR`, `REF_AREA`, `FREQ=Q`. The correction therefore **widens**
mcp_lite's lead (NonInd +11.7 → +25.5pp; all-dim +28.7 → +34.7pp) rather than
narrowing it. *(Correction uses the two codes proven single-valued across all 25
mcp_lite WS_NA_SEC cases; the baseline is assumed to share that dataset structural
property — re-probing under its own pins would only strengthen the gap.)*

### F.2 MD23 — correct code, wrong label

11 of 23 MD23 (sonnet RT+skill) cases scored < 1.0. Re-scoring **by code only**
(ignoring the `Term` name component) lifts mean recall **90.3% → 94.5% (+4.2pp)** —
that 4.2pp is the name-mismatch artifact above. Unlike BIS, this shows up as the
*same* code in both FN and FP with differently-worded labels. (BIS re-scored
code-only is unchanged, +0.0pp — its labels matched; its artifact is the §F.1
auto-fill kind instead.)

Applied to **all five §4.1 runs** (the code-only re-score reproduces each run's
published scored value exactly before correction — validation that the parse is
faithful):

| run | Recall s→c | Indic R s→c | NonInd R |
|-----|:----------:|:-----------:|:--------:|
| production baseline (gpt-4.1) | 87.3 → 91.3 | 82.4 → 89.2 | 95.5 |
| mcp sonnet — RT + skill | 90.3 → 94.5 | 85.4 → 92.5 | 97.3 |
| mcp sonnet — RT, no skill | 83.9 → 87.8 | 80.2 → 86.9 | 89.1 |
| mcp sonnet — skill, no RT | 82.3 → 86.8 | 79.5 → 87.1 | 87.8 |
| mcp opus — skill | 77.1 → 80.6 | 78.0 → 84.2 | 76.4 |

**NonInd recall is unchanged for every run** — the artifact is purely an
indicator-dim label phenomenon, and it lifts all runs ~uniformly, so it does not
change the ranking or the mcp-vs-baseline gap. (Precision is also nudged up by the
same FP→TP relabelling, not recomputed here.)

**Cases the name-only mismatch inflated:**

| id | scored | code-only | note |
|----|-------:|----------:|------|
| `99076693` inflation Argentina | 0.50 | **0.92** | `PCPIEPCH`/`PCPIPCH`/`CPI`/`YOY` all correct codes, verbose names |
| `5a08cf85` price levels Uruguay | 0.86 | **1.00** | pure name mismatch |
| `5f69257c` aggregate CPI Italy  | 0.88 | **1.00** | pure name mismatch (`PCPI`) |
| `77f3e530` prices YoY Kiribati  | 0.88 | **1.00** | pure name mismatch (`_T`) |
| `5c644283` bank credit S.Africa | 0.82 | 0.91 | part name, part real |
| `4ab50a69` govt debt Italy      | 0.75 | 0.81 | part name, part real |

**Genuine residual misses (code-only still < 1.0) — ~8 cases, all real:**

| id | code-only | real failure |
|----|----------:|--------------|
| `09cca3dc` GDP India        | 0.73 | missed **FSIC** dataset (GDP in domestic currency) |
| `4ab50a69` govt debt Italy  | 0.81 | missed **IIP** dataset (portfolio-investment debt lens) |
| `34630a9c` CPI Brazil       | 0.80 | picked the **rate** (`PCPIPCH`/`YOY`), truth wanted the **index** (`PCPI`/`IX`) |
| `ce8018e6` trade balance US | 0.85 | `BOP` `G1` merchandise vs truth `G`/`GS`; missed `UNIT=USD` |
| `99ce8346` Norway NEER      | 0.86 | `EER` `AEW` vs truth `ACW` weighting variant |
| `ea714a94` Mexico FX rate   | 0.86 | `ER` `XDC_EUR` vs truth `USD_XDC`; missed `EER_BASKET=N` |
| `5c644283` bank credit S.Africa | 0.91 | `FAS` `OUTL_COMBANK` vs `OUTL_COMBANK_A` (annual variant) |
| `99076693` inflation Argentina | 0.92 | truth wanted `POP`+`YOY` transformations; agent got only `YOY` |

The residual real failures are **2 whole-dataset misses + ~6 indicator-variant
disambiguations**, most on genuinely **underspecified questions** ("CPI" = index or
rate? "exchange rate" = vs USD or EUR? NEER weighting scheme?) — real, but several
are defensible reads. Dataset-level discovery is near-perfect; the gap is
intra-dataset variant choice.

**Net of both artifacts, both suites sit at ~94–95% data-equivalent recall.** Fix
for the MD23 artifact: score dim values by **code**, keeping name-equality as a
separate, non-recall-penalising "label drift" warning.


## G. What "Research thoroughly" (RT) changes mechanically

Isolated by comparing **RT+skill vs skill-no-RT** on the same 23 cases (skill held
constant). RT's +8pp recall is **not** mostly from more concept searches: the 6
cases where RT issued an extra `search_indicators` gained only **+3pp**, while the
17 **equal-search** cases gained **+9.7pp**. RT's real signature is *running the
whole loop more thoroughly*:

- **broader inclusion** — selects **+0.43 datasets/query and never fewer** (9 cases
  more, 0 fewer), catching the second-lens dataset skill-alone drops;
- **deeper resolution** — more `search_codes` + `availability_query` (recall gain
  correlates +0.39 / +0.49), pinning dims more accurately (Jamaica reserves: 4
  `search_codes` vs 1 → +67pp);
- slightly more `search_indicators` on hard cases (+0.26 avg) — the **weakest**
  contributor;
- tail risk: occasionally over-explores into a wrong variant (CPI Brazil −0.2).

**Reproducibility caveat.** All eval numbers are **single run per config (n=23)**
with a stochastic agent — per-case deltas carry run-to-run variance; only the
aggregates and directional signals (e.g. RT never selects fewer datasets) are
trustworthy. Clean attribution / CIs would need **3–5 runs per config**. Subagents
also ran **thinking-off** at unrecorded effort (see §E).

FOLLOWUP<
1. evals not consistent with claude desktop (artifacts created BEFORE final answer, so they might differ).
   PARTIAL FINDING (A/B on bank-credit-SA, sonnet, skill, thinking-off, vary only prompt):
   eval-framing "enumerate candidates" → ~4.5 datasets (stable); desktop-framing "answer the user"
   → ~1.5 (noisy). Direction reproducible across 4 reps each. CONFOUNDED: IMF data-fetch was DOWN,
   so desktop pivoted to BIS-only (the only fetchable dataset) — magnitude inflated by the outage,
   not pure framing. Also: real Desktop runs thinking-ON (these were thinking-off). TODO: re-run
   clean once IMF execute recovers, before concluding.
2. ✅ RT mechanics — see Appendix G (broader inclusion + deeper resolution, not more searches)
3.user facing demo
