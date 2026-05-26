# StatGPT MCP-Lite — Plan (no-LLM MCP)

**Issue:** [#694 — no-llm statgpt mcp](https://gitlab.deltixhub.com/Deltix/migapp/talk-to-your-data/statgpt-eval/-/issues/694)
**Related:** [MR !1097 — testcase generation MCP](https://gitlab.deltixhub.com/Deltix/migapp/talk-to-your-data/statgpt-eval/-/merge_requests/1097), [MR !1142 — full agentic data query](https://gitlab.deltixhub.com/Deltix/migapp/talk-to-your-data/statgpt-eval/-/merge_requests/1142), [`data_query_pipeline_backlog.md`](data_query_pipeline_backlog.md)

## Goal & non-goals

**Goal:** prove an external LLM (Claude Code / supreme agent) can drive the existing statgpt search + availability primitives well enough to be competitive with the in-tool LLM pipeline, **with zero LLM calls inside the MCP** (embeddings allowed).

**Non-goals (v0):** auth on the wire, multi-tenant safety, latency tuning, schema stability, polished errors, full client coverage. Few-day hypothesis test.

**Success signal:** Claude-Code-via-MCP single-dataset recall ≥ current backend pipeline on a small eval slice (BIS, ~40 cases — same set MR !1142 used to get 0.99 recall).

---

## Where it lives

```
statgpt-backend/statgpt/
  admin/           ← existing admin portal + admin MCP
  app/             ← existing agentic DIAL app + agentic MCP
  common/          ← shared services / data layer / SDMX  ← we import from here
  mcp_lite/        ← NEW — peer of app/ and admin/
    app.py            FastMCP server, mounted at /mcp-lite/{channel}/
    deps.py           channel-from-URL → ChannelServiceFacade resolver
    schemas.py        request / response shapes
    tools/
      __init__.py
      data_query.py   all tools, single file for v0; no channel arg
```

Forked from `admin/mcp/` for the auth model, with `app/mcp/`'s URL-bound channel pattern (minus DIAL creds). Reasons:
- `admin/mcp/` uses `SystemUserAuthContext()` (service principal) — no per-request DIAL creds.
- Stateless HTTP, mounted at a flat path, no `deployment_id` in the URL.
- Per-request `Depends(get_session_context_manager)` for DB; no `ChannelServiceFacade` machinery.

That matches #694 exactly. `app/mcp/`'s auth coupling is the baggage we don't want — but channel scoping is **kept**, just done differently (see below).

---

## Channel model

Each channel = a separate business client (BIS, IMF, Swiss Re, derzhstat, gastat-demo, ...). Per-client, distinct:

- **Visible datasets** — only the subset configured for that client.
- **Per-dataset annotations** — dimension aliases, codelist enrichment, named-entity types (lives in `scripts/config/clients/<client>/datasets/*.yaml`).
- **Glossary** — `glossaries/*.csv` per client.

**Channel is a structural access boundary, not a tool argument.** Putting it on every tool would mean the *caller* enforces the boundary — a leak the moment the agent has a bug or hallucinates a channel name. The server has to enforce it. So we **mirror `app/mcp/`'s URL-binding pattern**, just without DIAL auth:

```
URL:  /mcp-lite/{channel}/...
              ↑
       channel extracted from path on every request → ChannelServiceFacade
       resolved once → injected into every tool as scoped context
```

What this gives us:

- Each MCP connection is bound to exactly one channel for its lifetime. A `bis` client literally cannot call into `imf` without changing the endpoint URL.
- Tools take **no `channel` arg** — they receive a pre-scoped `ChannelServiceFacade` via dependency injection, the same way `app/mcp/`'s tools do.
- No DIAL credentials. The channel is a *scoping* primitive, not an *authentication* primitive. The MCP server runs with a service-principal auth context (same as `admin/mcp/`).
- Multi-channel comparison from one agent requires connecting to two MCP endpoints. That's a feature, not a bug — clean blast radius.

What we don't take from `app/mcp/`: the "tools dynamically wired per channel from `ChannelConfig.tools`" part. mcp_lite always exposes the same 8 tools regardless of channel — channel only scopes *what each tool sees*, not *which tools exist*. Implementation-wise this means `ChannelToolProvider` becomes a thinner provider that builds the same tool list, just bound to a different facade per request.

---

## The split: current `data_query` → MCP-lite tools

### Current pipeline (today)

One tool call to `data_query` triggers a 3-stage chain with ~7+ LLM calls and ~32s mean wall-clock.

```
                 supreme agent
                       │
                       ▼
        ┌──────────────────────────────┐
        │         data_query           │   ONE tool call
        │  ~32s · ~7+ LLM calls inside │   black box to the agent
        └──────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │ Stage 1 — Search Prep       │   norm → dset-sel → (NER ‖ time)
        │ Stage 2 — Dimension Search  │   non-ind ‖ ind (hybrid) ‖ special
        │ Stage 3 — Finalize Query    │   build → avail → route
        └─────────────────────────────┘
```

### Decomposition: every stage gets one of four fates

```
       Stage / sub-step                Fate          Becomes
       ───────────────────────────────────────────────────────────────────
  S1a  Normalization                   [D]  DROP    (agent writes intent directly)
  S1b  Dataset selection               [C]  CALLER  (agent picks; passed as arg)
  S1c  NER                             [C]  CALLER  (agent passes entity hints)
  S1d  Time period extraction          [C]  CALLER  (agent passes parsed period)
  S1e  Country entity filter           [D]  DROP    (collapsed into availability)

  S2a  Non-indicator search            [T]  TOOL    search_codes (source=non_indicator)
  S2b  Hybrid: lexical_pre_match       [T]  TOOL    (inside search_codes)
  S2c  Hybrid: normalize_input         [D]  DROP    (LLM, not needed)
  S2d  Hybrid: separate_subjects       [D]  DROP    (LLM, agent decomposes itself)
  S2e  Hybrid: _hybrid_candidates      [T]  TOOL    search_codes (source=indicator) body
       └─ _lexical (ES)                                    │
       └─ _semantic_raw (pgvector)                         │   embedding call OK
       └─ _hybrid_combine                                  │
  S2f  Hybrid: relevance batch (LLM)   [C]  CALLER  (agent filters returned candidates)
  S2g  Special dimensions selection    [C]  CALLER  (treated as plain dim values)

  S3a  Build dataset queries           [T]  TOOL    (folded into execute_sdmx_query)
  S3b  Dataset query availability      [T]  TOOL    availability_query
  S3c  Time period defaults / routing  [D]  DROP    (caller validates result)
  S3d  Execute SDMX                    [T]  TOOL    execute_sdmx_query (returns URL + data)

       Glossary                        [R]  REUSE   list_glossary_terms,
                                                    get_glossary_term

       Legend:
         [T] new MCP-lite tool (wraps existing service)
         [R] reuse: tool already non-LLM in backend, expose
         [C] caller responsibility (agent passes as arg)
         [D] drop entirely
```

### New surface

```
                  supreme agent / Claude Code
                          │
        ┌─────────────────┼─────────────────────────────────────────────────────────┐
        │                 │                                                         │
        ▼                 ▼                                                         ▼
   list_datasets     dataset_structure                                        search_codes (merged: indicator + non-indicator + special)
   list_glossary     sample_dim_values (n=-1 for full codelists)              availability_query
   get_glossary_term                                                          execute_sdmx_query


                       (8 tools, all stateless, no LLM)
```

---

## Tool inventory

All tools stateless, no LLM calls inside. Signatures use Python-style type hints; example payloads are sketches, not final schemas.

> **No `channel` arg on any tool.** Channel is bound by URL (`/mcp-lite/{channel}/...`) and injected into each tool as a pre-scoped `ChannelServiceFacade`. See [Channel model](#channel-model) above. All examples below assume the agent has already connected to `/mcp-lite/bis/`.

### Reuse — already non-LLM, exposed somewhere in backend

**`list_glossary_terms()`**
Wraps `GlossaryOfTermsService.list_terms` (powers `AvailableTermsTool`). Returns the channel's glossary.

```
→ list_glossary_terms()
← [{"term":"CPI","short":"Consumer Price Index"},
   {"term":"GDP","short":"Gross Domestic Product"}, ...]
```

**`get_glossary_term(term: str)`**
Wraps `GlossaryOfTermsService.get_definition` (powers `TermDefinitionsTool`).

```
→ get_glossary_term(term="CPI")
← {"term":"CPI",
   "definition":"Consumer Price Index measures the change in prices ...",
   "source":"BIS glossary"}
```

### Lift — exists in MR !1097 (statgpt-eval side), move to backend

> #1097 says *"Integrate MCP into statgpt-backend and remove MCP from statgpt-eval"* as a next step. This MCP **is that integration**.

**`list_datasets()`**
Datasets visible to the channel. Doesn't work behind SDMX proxy per MR !1097 — accepted limit.

```
→ list_datasets()
← [{"id":"BIS_LBS", "name":"Locational banking statistics", "agency":"BIS"},
   {"id":"BIS_CPI", "name":"Consumer prices",               "agency":"BIS"}, ...]
```

**`dataset_structure(dataset_id: str)`**
Dimensions, dim types, codelist sizes, applying the channel's per-dataset annotations / aliases. Large codelists are summarized (size only); call `sample_dim_values` or `search_codes` to inspect contents.

```
→ dataset_structure(dataset_id="BIS_LBS")
← {"id":"BIS_LBS",
   "dims":[
     {"id":"INDICATOR",   "type":"indicator",     "codelist_size":42},
     {"id":"REF_AREA",    "type":"non_indicator", "codelist_size":250},
     {"id":"L_POSITION",  "type":"non_indicator", "codelist_size":4},
     {"id":"INSTR_ASSET", "type":"non_indicator", "codelist_size":12},
     {"id":"FREQ",        "type":"special",       "codelist_size":3},
     {"id":"TIME_PERIOD", "type":"time"}
   ]}
```

**`sample_dim_values(dataset_id: str, dim_id: str, n: int = 20)`**
Random / top-N values for a single dimension. Use to discover what a dim contains when its purpose isn't obvious from the name.

```
→ sample_dim_values(dataset_id="BIS_LBS", dim_id="L_POSITION", n=10)
← [{"code":"C","name":"Total claims"},
   {"code":"L","name":"Total liabilities"},
   {"code":"N","name":"Net (claims − liabilities)"},
   {"code":"A","name":"All positions"}]
```

**`availability_query(dataset_id: str, dims: dict[str, list[str]])`**
Reachable dim values given a partial filter. The agent's core search primitive — call repeatedly to narrow.

```
→ availability_query(
    dataset_id="BIS_LBS",
    dims={"INDICATOR":["CBS_LBS"], "REF_AREA":["DE"]}
  )
← {"ok": true,
   "available":{
     "L_POSITION":  ["C","L","N"],
     "INSTR_ASSET": ["F3","F4"],
     "FREQ":        ["Q"]},
   "time_range":["2010-Q1","2025-Q1"]}
```

### New — merged search

> **Design decision (post-v0 refinement)**: the original plan had two search tools — `search_codes_by_name` (lex over non-indicator codelists) and `search_indicators` (hybrid over indicators). Collapsed into a single `search_codes` for v0. Reasons:
>
> 1. MR !1097's `search_codes_by_name` already merged all three search surfaces (indicator + non-indicator + special), tagging each match with a `source` field. The "two tools" framing was a misread of the plan.
> 2. Once both are **hybrid (lex + sem)**, the only difference between the two is *which vector store / ES index* is queried — that's a parameter, not a separate tool.
> 3. The agent doesn't always know upfront whether a name is an indicator or a code; one search-everything entry point with a `source` tag lets it post-filter rather than guess.
> 4. Per advice received: **indicator search should expose which dataset each match comes from** (so the agent can pivot across datasets A/B/C) and **indicator results should include the dimension breakdown** so the agent can take a partial sub-series — e.g. hybrid returns "GDP, per capita, current prices" but the agent keeps just `INDICATOR=GDP, UNIT=PERSON` and wildcards `PRICE_TYPE`.

**`search_codes(query: str, dataset_id: str | None = None, k: int = 20, sources: list[str] | None = None)`**

Hybrid lex + semantic search across indicator, non-indicator, and special-dim vector stores. Wraps `ChannelServiceFacade.search_*_scored` methods; lex side via Elasticsearch. **No LLM rerank.** Caller's LLM filters.

| arg | purpose |
|---|---|
| `query` | Free-text. Same `query` is sent to all enabled `sources`. |
| `dataset_id` | If set, scope to one dataset. If null, search **every** dataset in the channel; each result is tagged with its `dataset_id`. |
| `k` | Top-K per source (default 20). |
| `sources` | Optional whitelist: any subset of `["indicator", "non_indicator", "special"]`. Default = all three. |

Returns a flat list of matches, sorted by score:

```
→ search_codes(query="gdp per capita")
← [
    {"source":"indicator", "dataset_id":"IMF.STA:WEO(1.0)",
     "code":"NGDPRPC_USD_PERCAP",
     "name":"GDP, per capita, current prices",
     "score":0.91,
     "dimensions":{"INDICATOR":"GDP", "UNIT":"PERSON", "PRICE_TYPE":"CURRENT_PRICES"}},

    {"source":"non_indicator", "dataset_id":"BIS_LBS",
     "dim_id":"REF_AREA",
     "code":"DE", "name":"Germany",
     "score":0.87},
    ...
  ]
```

- **`source: indicator`** results include `dimensions` — the dim values pinned by the indicator. Agent can drop pins to wildcard them in `execute_sdmx_query`.
- **`source: non_indicator` / `special`** results are flat: `dim_id`, `code`, `name`.
- All results carry `dataset_id` and `score`. Agent sorts/filters by score; no LLM rerank inside the MCP.
- **Score is per-source max-relative in [0, 1].** Indicators get `α·sem + (1−α)·lex` where each side is already max-normalized; non-indicators get `(score+1) / (max+1)` over the result set. The top hit within each source is ~1.0. Useful for ranking *within a source*; do **not** compare absolute values across queries. The merged sort therefore interleaves sources by "rank within group," not by absolute relevance — statgpt's existing data_query pipeline avoids this problem by keeping the sources in separate chains and letting an LLM decide which to use; we accept this caveat because v0 explicitly forbids in-MCP LLMs.

> **Design decision — non-indicator deduplication across datasets.** The backend's non-indicator vector store (`collections."AvailableDimensions_{channel_id}"`) is **shared content storage**: each unique `(dim_id, code, name)` lives as ONE document with ONE embedding, plus a separate mapping row per dataset that contains it. For the gtdc channel today: 1,115 unique documents → 4,590 mapping rows (~4× duplication; "Germany" alone has 17 mappings). A naive cross-dataset search returns the same value N times — verified empirically with `"GDP per capita"` where `"Advanced Economies"` filled 4 of the top 8 slots.
>
> Behaviour:
> - **`dataset_id` arg set** (single-dataset search): only one mapping per document matches the version_id filter, no duplication, no dedup needed.
> - **`dataset_id` null** (cross-dataset): the tool dedupes non-indicator results by `(dim_id, code, name)`, keeps the top-scoring mapping as the canonical `dataset_id`, and lists every other dataset the same value exists in under a new optional field `available_in: list[str]`. The agent gets one row per real value plus a pointer to which datasets carry it.
>
> Schema effect: `CodeMatch.available_in: list[str] | None`. Null when there's only one occurrence (single-dataset, or value unique to one dataset cross-dataset).
>
> Why not dedup indicators too? Indicator docs *are* per-dataset — `(dataset_id, indicator_composite_id)` is unique because the dim breakdown encodes the dataset's specific dim ids. Cross-dataset "GDP" matches in WEO vs IMTS are genuinely different indicator documents with different `series` structures, so they should stay separate.
>
> Trade-off accepted: dedup happens **after** the underlying top-K search returns. If the top-K is dominated by duplicates of one value, the dedupped result has fewer unique entries than `k` requested. v0b can over-fetch (e.g. `k * 4`) to compensate if eval shows this matters.

> **Asymmetry — hybrid is only available for the indicator source.** The current StatGPT backend has an Elasticsearch index (`indicators_index`) for indicators but **no ES index for non-indicator or special dim values** — those live only in pgvector. Concrete consequences:
>
> | source | lex (ES) | sem (pgvector) | v0 behaviour |
> |---|---|---|---|
> | `indicator` | ✅ `indicators_index` | ✅ indicator vector store | **hybrid** (lex + sem, convex combination) — re-uses HybridSearcher's `_lexical` + `_semantic_raw` + `_hybrid_combination` pieces; skips its LLM rerank |
> | `non_indicator` | ❌ — | ✅ non-indicator dims store | **semantic only** (`facade.search_non_indicator_dimensions_scored`) |
> | `special` | ❌ — | ✅ special dims store | **semantic only** |
>
> Mitigation in v0: the agent can still narrow non-indicator hits with cheap follow-up calls — e.g. semantic returns `DEU Germany` plus near neighbours (Austria, Switzerland) for "germany"; the agent confirms by checking the codelist (`sample_dim_values`) or chaining `availability_query`. Recall on common entities (countries, large codelists) is typically fine even without lex.
>
> **Possible next steps to make non-indicator hybrid:**
> 1. **Index non-indicator dim values into a new ES collection** during channel onboarding (analogous to the indicator indexer). Then `search_codes` adds a lex side per source. Best long-term fix; ~1 week of work to wire indexer + reindex existing channels.
> 2. **Add an in-process fuzzy match layer** (`rapidfuzz` over `dimension.values`) as a per-request lex pass. No new infra, runs in <50 ms even for 1k-value codelists. Cheap stopgap; ~1 day. Caveat: enumerates the full codelist per query — fine for ~thousand-value dims, breaks down for ICD-style (14k+) codelists.
> 3. **Rely on the existing semantic plus aggressive caller-side rerank** — caller's LLM sees top-K semantic hits and disambiguates. Zero backend changes. Cost: an LLM call per search; not in the spirit of "no LLM inside MCP" but it's *outside* the MCP.
>
> Recommend (2) as the v0b stopgap if non-indicator semantic recall turns out to be poor in eval. (1) is the right answer once mcp_lite proves out as a hypothesis.

**`execute_sdmx_query(dataset_id: str, dims: dict[str, list[str]], time_period: {start, end} | None = None)`**
Compose, sanity-check (availability), and run the SDMX query in one step. Returns resolved URL alongside data so caller retains inspectability. Build-without-execute deliberately not exposed in v0 — `availability_query` covers the dry-run case.

```
→ execute_sdmx_query(
    dataset_id="BIS_CPI",
    dims={"INDICATOR":["CPI_HEADLINE_YOY"], "REF_AREA":["DE"], "FREQ":["Q"]},
    time_period={"start":"2010", "end":"2025"}
  )
← {"query_url":"https://stats.bis.org/api/v2/data/BIS_CPI/Q.DE.CPI_HEADLINE_YOY?startPeriod=2010&endPeriod=2025",
   "row_count":61,
   "time_range_actual":["2010-Q1","2025-Q1"],
   "data":[
     {"TIME_PERIOD":"2010-Q1","value":1.2},
     {"TIME_PERIOD":"2010-Q2","value":1.1}, ...]}
```

### Dropped — caller does it or it's gone

- Normalization (LLM)
- Dataset selection (LLM) → `dataset_id` is a required arg on `dataset_structure` / `sample_dim_values` / `availability_query` / `execute_sdmx_query`, and optional on `search_codes` (null → cross-dataset)
- NER (LLM) → caller passes entity hints (dim_id → values)
- Time-period extraction (LLM) → caller passes `{start, end}` or SDMX time string
- Indicator relevance rerank (LLM) → caller filters `search_codes` output
- `separate_subjects` / `normalize_input` inside HybridSearcher (LLM)
- Special-dimensions LLM selection → exposed as another `source` in `search_codes`
- `build_sdmx_query` as a standalone tool → folded into `execute_sdmx_query` (returns URL too)
- `search_codes_by_name` + `search_indicators` as separate tools → merged into a single `search_codes` (see Design decision above)

---

## Typical agent loops

### Single-dataset query — "inflation in Germany since 2010"

```
  agent connects to:  /mcp-lite/bis/    ← channel bound for this session

agent ── list_datasets() ──────────────────────────────────────►
      ◄────────────────── [{id:"BIS_CPI", name:"CPI"}, ...]

agent ── dataset_structure(dataset_id="BIS_CPI") ──────────────►
      ◄────────────────── {dims: [INDICATOR, REF_AREA, FREQ, ...]}

agent ── search_codes("inflation cpi", dataset_id="BIS_CPI") ──►
      ◄────────────────── [{source:"indicator", code:"CPI_HEADLINE",
                            dimensions:{INDICATOR:"CPI_HEADLINE"}, score:.91}, ...]

agent ── search_codes("germany", dataset_id="BIS_CPI") ─────────►
      ◄────────────────── [{source:"non_indicator", dim_id:"REF_AREA",
                            code:"DE", name:"Germany", score:.97}]

agent ── availability_query(BIS_CPI,                           ►
              {INDICATOR:["CPI_HEADLINE"], REF_AREA:["DE"]})
      ◄────────────────── {ok:true, time_range:["2010","2025"]}

agent ── execute_sdmx_query(BIS_CPI,                           ►
              {INDICATOR:["CPI_HEADLINE"], REF_AREA:["DE"],
               TIME_PERIOD:{start:"2010", end:"2025"}})
      ◄────────────────── {data: [...]}
```

Six tool calls, no LLM inside MCP. If this loop consistently needs more than ~10 turns or repeated `availability_query` thrashing, that's a signal we're missing a primitive (likely `resolve_entity` or a combined `find_dim_values_for_query`).

### Comparison to MR !1142 baseline

```
                                 Tool calls   LLM inside MCP   Recall (BIS)
  current backend pipeline           1            ~7             0.71 (single)
  MR !1142 (claude-code MCP)         ~10           0             0.99 (single)
  mcp_lite (this plan)               ~6–10        0             target ≥ baseline
```

MR !1142 hit 0.99 recall with similar tool count and zero internal LLM. We're betting we can hit comparable numbers with a slimmer surface backed by the same hybrid index.

---

## File scaffold (concrete, two-file fork)

`statgpt-backend/statgpt/mcp_lite/app.py` (forked from `admin/mcp/app.py` + URL-bound channel from `app/mcp/`, no DIAL auth):

```python
from fastmcp import FastMCP
from statgpt.mcp_lite.tools import mcp_tools
from statgpt.mcp_lite.deps import get_channel_facade  # extracts {channel} from path

mcp = FastMCP(providers=[mcp_tools])

# Mounted at /mcp-lite/{channel}/ — channel resolved per request and turned
# into a ChannelServiceFacade that tools receive via Depends(get_channel_facade).
http_app = mcp.http_app(
    path="/{channel}/",
    transport="streamable-http",
    stateless_http=True,
)
```

`statgpt-backend/statgpt/mcp_lite/deps.py` (channel resolution):

```python
from fastapi import Request, HTTPException
from statgpt.app.services import ChannelServiceFacade

async def get_channel_facade(request: Request) -> ChannelServiceFacade:
    channel = request.path_params["channel"]
    try:
        return await ChannelServiceFacade.get_channel(channel)
    except ChannelNotFound:
        raise HTTPException(404, f"Unknown channel: {channel}")
```

`statgpt-backend/statgpt/mcp_lite/tools/data_query.py` (sketch):

```python
from fastmcp import LocalProvider, Depends
from statgpt.common.services import DataSetService, GlossaryOfTermsService
from statgpt.app.services import HybridSearcher  # cross-import; lift later if kept

mcp_tools = LocalProvider()

@mcp_tools.tool
async def list_datasets(
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # ← from URL
    session: AsyncSession = Depends(get_session_context_manager),
):
    ...

@mcp_tools.tool
async def search_codes(
    query: str,
    dataset_id: str | None = None,        # null ⇒ search all datasets in channel
    k: int = 20,
    sources: list[str] | None = None,     # subset of {indicator, non_indicator, special}
    facade: ChannelServiceFacade = Depends(get_channel_facade),
    session: AsyncSession = Depends(get_session_context_manager),
):
    # Hybrid lex + sem per source:
    #   indicator      → ES (indicators_index) + pgvector (indicator vector store)
    #   non_indicator  → ES (matching_index)   + pgvector (non-indicator dims store)
    #   special        → pgvector (special dims store)
    # No LLM rerank. Results carry: source, dataset_id, code, name, score,
    #   plus `dimensions` (for indicators) or `dim_id` (for non-indicator / special).
    ...

# ... 6 more tools, all flat — facade injected via dependency, never a tool arg
```

Mount on the same FastAPI host as `admin/mcp` (different path: `/mcp-lite`) or stand up a tiny separate FastAPI app in `mcp_lite/main.py` — pick whichever is faster to wire on day 1.



## Risks (known, accepted for v0)

1. **`search_codes` without LLM rerank** may be noisy. If recall is poor, the first lever is *external* rerank — the caller's LLM does it; we don't put a model inside the MCP.
2. **Large codelists** (KVED has 1k+ values): `dataset_structure` payload can blow up. Paginate dim values; expose `get_codelist(dataset_id, dim_id, page)` if needed.
3. **`list_datasets` doesn't work with SDMX proxy** (per MR !1097 note). Document, move on.
4. **No NER means free-text country/KVED resolution depends on `search_codes`.** If "germany" → `DE` fails frequently in eval, add `resolve_entity` as the next tool.
5. **Cross-package import** (`mcp_lite` → `app.services.HybridSearcher`). Pragmatic now; if `mcp_lite` survives the experiment, the cleanup is to lift `HybridSearcher` into `common/services/`.
6. **Cross-dataset `search_codes`** (when `dataset_id=null`) returns matches from many datasets; result payload grows. v0 caps per-source at `k`; if payloads are still too large, lower the default or chunk per dataset.

---

## Open questions to confirm with Ales before day 1

1. **Mount point.** Inside backend FastAPI host (next to `admin/mcp`) or new process?
2. **Channel wiring.** v0 binds channel via URL (`/mcp-lite/{channel}/`) and injects a scoped facade per request — same shape as `app/mcp/` minus DIAL auth. Confirm this is the model Ales had in mind for #694.
3. **`list_datasets` vs SDMX proxy.** Acceptable limitation, or do we need a workaround?
