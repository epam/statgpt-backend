# mcp_lite eval report — phase 2 handoff

Self-contained brief for a Claude Code instance running in the **statgpt-eval**
repo. Phase 1 (eval runs) is done in `statgpt/mcp_lite/eval_runs/`; phase 2 is
**report generation, cross-config comparison, and trend tracking**.

Read this end-to-end before touching code. Every path is absolute or repo-rooted
so it works regardless of where the eval repo is cloned.

---

## What's already been done (phase 1)

Six full eval runs on 11 broad-concept testcases, plus a baseline. All results
live under [`statgpt/mcp_lite/eval_runs/`](eval_runs/). Summary in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

| # | run dir | setup | model | recall | precision | avg time | avg tok |
|---|---|---|---|---:|---:|---:|---:|
| R1 | `data-40/evals/20260526-112654.gtdc.broad-queries/` | statgpt-app baseline | gpt-4.1 | 77% | 90% | 85.6s | 12,461 |
| R2 | `20260526.claude.mcplite.gtdc.broad-discovery/` | mcp_lite flat, no hint | opus-4.7 | — | — | — | — |
| R3a | `20260526.…search-indicators-hint-final/` | flat + hint | opus-4.7 | 43% | 100% | 104.3s | 44,237 |
| R3b | `…search-indicators-hint-final.sonnet/` | flat + hint | sonnet-4.6 | 49% | 100% | 103.1s | 34,166 |
| R3c | `…search-indicators-hint-final.haiku/` | flat + hint | haiku-4.5 | 31% | 100% | 117.4s | 35,984 |
| R4 | `20260527.…grouped-big-hint-sonnet-11q/` | grouped + big hint | sonnet-4.6 | 63% | 100% | 112.0s | 39,754 |
| **R5** | `20260527.…grouped-big-hint-sonnet-rt-11q/` | R4 + "Research thoroughly." | sonnet-4.6 | **80%** | **100%** | 134.2s | 48,559 |

R5 beats the gpt-4.1 baseline on both recall (80 vs 77) and precision (100 vs 90)
at ~9× the token cost.

---

## What you need to build (phase 2)

The user is at the **report generation** step. Concretely, three deliverables, in
priority order:

1. **Per-run summary report** — given a run dir, produce a single
   self-contained markdown/HTML file with:
   - The headline metrics (recall, precision, picks, hits, avg time, avg tokens, avg cost/q).
   - Per-test-case table: query, expected datasets, picked datasets, hit/miss, time, tokens.
   - For misses: what the agent picked instead, what the agent missed, why
     (was the missed dataset in the top-K of `search_indicators`? what rank?).

2. **Cross-run comparison report** — given N run dirs, produce a side-by-side
   matrix:
   - One row per testcase.
   - One column group per run with picks/hits/time/tokens.
   - Highlight regressions (recall drop relative to the best-prior run).
   - Optional aggregate header table (`EXPERIMENTS.md` headline-table style).

3. **Hand-off-friendly trend chart** — recall/precision over time across the run
   series (R1→R2→R3→R4→R5). Markdown table is fine; an SVG/PNG would be nicer.
   No new build tooling — use whatever stack the eval repo already has.

If a `phase2` skill / scaffold already exists in the eval repo, follow it
verbatim. If not, default to **markdown reports + python script that produces
them**; the user prefers explicit code paths over hidden orchestration.

---

## Data layout

Each run dir has this shape:

```
20260527.…grouped-big-hint-sonnet-rt-11q/
├── jobs/
│   ├── 1dc6d65bff374773938e8f.json         ← one per test case
│   ├── 70ae292b2a904f78868e62.json
│   └── …  (11 files total for broad-discovery runs)
└── (no other top-level files)
```

The statgpt-app baseline (`data-40/`) has a different shape because it was
produced by the standalone eval framework:

```
data-40/evals/20260526-112654.gtdc.broad-queries/
├── eval.20260526-112654.gtdc.broad-queries.xlsx   ← human-readable summary
├── eval.log
└── jobs/
    ├── p5-YThJETy21zF9zbGV9Tw.json
    ├── job_stats.jsonl                            ← aggregate per-job metrics
    └── …  (11 files)
```

Always cross-reference the directory naming convention (full filename gives you
the date, channel, eval-type, config-suffix, and model).

---

## Job-file schema (mcp_lite runs)

Each `jobs/*.json` has top-level keys `test_case` and `responses`. Anatomy:

```python
{
  "test_case": {
    "id": "50774b6d-acdc-4379-8d65-956463ee39eb",         # UUID for the case
    "name": "<same as id, the eval framework dedups by it>",
    "tags": [],
    "conversation": [                                      # one user turn for these runs
      {"role": "user",
       "content": "What are the foreign reserves of China? Research thoroughly.",
       "target": None}
    ]
  },
  "responses": [
    {
      "data": {                                            # DIAL chat-completion payload
        "id": "...",
        "choices": [{
          "message": {
            "content": "",                                 # text answer (often empty for tool-only runs)
            "custom_content": {
              "state": {                                   # DIAL-specific debug envelope
                "show_debug_stages": False,
                "tool_messages": [
                  {"type": "ai", "tool_calls": [{"name": "Query_Data", "args": {"query": "…"}}]},
                  {"type": "tool",
                   "tool_call_id": "claude_data_query_0",
                   "status": "success",
                   "content": "success",                   # or an error string on failure
                   "custom_content": {
                     "state": {                            # ← the agent's tool-message dump
                       "type": "DATA_QUERY",
                       "datasets_selection_response": {
                         "dataset_ids": ["IMF.STA:BOP(21.0.0)",
                                         "IMF.STA:IIP(13.0.0)"]      # ← **THE PICKS** (mcp_lite)
                       },
                       "indexed_datasets_id_map": {…},
                       "dataset_queries": {…},                       # only set in baseline (R1)
                       "normalized_query": …,
                       "named_entities_response": …,
                       …
                     }
                   }}
                ],
                …
              }
            }
          }
        }]
      },
      "status_code": 200,
      "start_time_utc": "2026-05-27T07:14:46.838Z",
      "end_time_utc":   "2026-05-27T07:19:33.282Z",
      "response_time": 286.44,                              # wall-clock seconds
      "exception_info": None,
      "failed_attempts": 0
    }
  ]
}
```

Picks live in **two different places** depending on run type:
- **mcp_lite runs** (R2–R5): `data.choices[0].message.custom_content.state.tool_messages[i].custom_content.state.datasets_selection_response.dataset_ids` — a flat `list[str]`.
- **statgpt-app baseline** (R1): `data.choices[0].message.custom_content.state.tool_messages[i].custom_content.state.dataset_queries` — a `dict[uuid, dataset_query]`. The keys are the dataset UUIDs; map them to source ids via `indexed_datasets_id_map` (which lives at the same level).

The `mcp_lite/broad_testcases/patch_artifact.py` script post-edits each artifact
to merge in subagent harness metrics (wall-time, total tokens, per-tool counts +
durations) that the agent itself doesn't see. **Use whatever metrics the
artifacts already carry**; don't try to recompute them from logs.

---

## Ground truth — testcases (canonical location)

**Live in the statgpt-eval repo**, not here. Path:

```
/Users/bahdan_kapionkin/Documents/Deltix/statgpt-new/data/test_cases/conversational/gtdc/data_query/multi_dataset/*.yaml
```

11 yaml files (one per case). Sibling files under
`statgpt/mcp_lite/broad_testcases/*.yaml` exist in *this* repo but are an
**older, divergent copy** with a different schema — **ignore them**. Always
read from the canonical eval-repo path above.

### Canonical schema (eval-repo)

```yaml
id: c10ef09f-37ca-4754-8714-c951db3bee23
name: unemployment_japan
tags: [multidataset]
comments: |
  Broad query with two distinct interpretations exposed by the channel:
  - WEO.LUR — unemployment RATE (percent of labor force, annual …)
  - NSDP.LU_PE — unemployment COUNT (persons, monthly) …
conversation:
- role: user
  content: 'What is the unemployment rate in Japan?'
  target:
    is_out_of_scope: false
    tool_calls:
    - - tool_type: DATA_QUERY
        args: {query: 'unemployment rate in Japan'}
    datasets_selection:
      datasets: []
    indicator_selection:                              # ← THE TRUTH LIST
    - dataset_id: IMF.RES:WEO
      dimensions:
      - {dimension_name: COUNTRY,   values: [{id: JPN, name: Japan}]}
      - {dimension_name: INDICATOR, values: [{id: LUR, name: 'Unemployment rate'}]}
      - {dimension_name: FREQUENCY, values: [{id: A,   name: Annual}]}
    - dataset_id: IMF.STA:NSDP
      dimensions:
      - …
```

The truth list lives at **`conversation[].target.indicator_selection[].dataset_id`**.
Each entry there is one dataset that materially answers the question; the
`dimensions` block is the canonical pin set for that dataset.

### Truth-set totals (canonical schema)

Re-counted from the eval-repo yamls on `2026-05-27`:

| count | meaning |
|---|---|
| **37** | total `indicator_selection` entries across all 11 cases — **this is the recall denominator** |
| 37 | unique `(case, dataset_id)` pairs (every mention is unique per case in this schema) |
| 18 | unique `dataset_id`s globally |

Per case: `bank_credit_south_africa=4, cpi_brazil=3, current_account_france=4,
exchange_rate_mexico=4, foreign_reserves_china=2, gdp_india=4, gov_debt_italy=5,
inflation_argentina=4, population_germany=1, trade_balance_us=4, unemployment_japan=2`.

**Note the discrepancy with EXPERIMENTS.md.** The recall denominator there is
declared as 35; with the canonical truth set, the correct denominator is **37**.
That puts R5 at 28/37 = **75.7%** (not 80%) and the baseline at 27/37 = **73.0%**
(not 77%) — recomputed from the same `hits` counts. R5 still beats baseline,
just by less than EXPERIMENTS.md claims. When you publish phase-2 reports, use
37 and surface the mismatch; don't silently rewrite EXPERIMENTS.md.

### Important: dataset ids in truth-set are unversioned

Truth-set entries use **bare ids** like `IMF.RES:WEO` (no `(9.0.0)` version
suffix). Job artifacts and `search_indicators` return **versioned ids** like
`IMF.RES:WEO(9.0.0)`. Strip the parenthesised version suffix on the artifact
side before comparing, or your hits count will be 0 across the board.

### Mapping jobs → testcases

The testcase yaml carries its own UUID at the top (`id: c10ef09f-…`). The eval
framework may or may not preserve that UUID in the `job.test_case.id` field; in
the runs we have it appears to **regenerate UUIDs per run**, so don't rely on
UUID equality.

Match by **query string** instead. Take `job.test_case.conversation[0].content`,
strip the optional `"!skip_data_query_summarization "` prefix and `" Research
thoroughly."` suffix, and look it up against `conversation[0].content` in the
canonical yamls. That's deterministic across all 6 runs.

---

## Metric definitions (use these exactly)

| metric | definition |
|---|---|
| **picks** | distinct dataset ids the agent selected for this case (from `datasets_selection_response.dataset_ids`) |
| **hits** | picks that appear in the case's `candidate_datasets[].dataset_id` |
| **recall** | total_hits across all 11 cases / 35 (the truth-set size) |
| **precision** | total_hits / total_picks |
| **avg_time** | mean `response_time` (seconds) across the 11 cases of a run |
| **avg_tokens** | mean total token count; mcp_lite reads from harness metrics merged in by patch_artifact, baseline reads from DIAL `usage_per_model` |
| **avg_cost/q** | tokens × per-million price; baseline uses gpt-4.1 ($2/$8 in/out, 80/20 split); claude tiers use opus $15/$75, sonnet $3/$15, haiku $1/$5 with the same 80/20 split |

The 80/20 input/output split is a **rough estimate** — it's accurate to the
order of magnitude, not to the cent. Don't quote dollars to 4 decimal places.

---

## Server-side timing data (per-tool latency)

`statgpt/mcp_lite/timing.py` is a FastMCP middleware that logs every `tools/call`
to `/tmp/mcp_lite_timing.log` in UTC. Format:

```
2026-05-26 10:36:14,430  search_indicators   duration_ms= 6863.5  ok=true
```

The `patch_artifact.py` script joins these timing entries with the artifact's
wall-clock window (using UTC timestamps) and merges per-tool counts + durations
into the artifact. If you need raw timing data that wasn't merged, parse the log
yourself with the regex in `patch_artifact.py:46`.

Don't rely on the timing log being present — it's a developer-laptop artifact,
not a production data source. Most eval-run jobs only have what `patch_artifact`
already merged.

---

## Phase-2 tool-call dump (per-job) — the harder bit

For the per-run report's miss-analysis, you need to know **what `search_indicators`
returned** for each case — specifically whether the missed dataset was even in
the top-K. That data isn't stored in the artifact (it's lost between the
subagent and DIAL). Options:

1. **Cheapest**: rerun `search_indicators(query=<the testcase query>, top_k=20)`
   against a live mcp_lite server, capture the JSON, attach to the report.
   Requires a running server at `http://localhost:5000/api/v1/statgpt-gtdc/mcp-lite/`
   (or wherever the eval repo's deployment points). The query is deterministic
   modulo embedding-service variance, so this is reproducible enough for
   post-hoc analysis.

2. **Most thorough**: re-run the subagent with stdout capture to file. Slow but
   gives you the full tool-call sequence. Probably not worth it unless we hit a
   case where rerun-search gives different results.

3. **Best of both**: skip the miss-analysis for v1 of the report, just show
   pick/miss without rank-in-search context. Ship the report, decide if rank
   data is needed by reading it.

Pick (3) for the first iteration. Add (1) only if the user pushes back on the
report being shallow.

---

## What's intentionally NOT in scope for phase 2

- **No new eval runs**. Phase 1 is closed (R1–R5 are the canonical comparison
  set). If a new run is needed, the user will say so explicitly.
- **No re-curation of testcases**. The yamls in the canonical eval-repo path
  (`…/data/test_cases/conversational/gtdc/data_query/multi_dataset/`) are the
  truth. If they look wrong, flag it; don't patch them silently.
- **No model-pricing renegotiation**. The numbers in `EXPERIMENTS.md` use the
  pricing table above; use the same. Re-pricing has its own conversation later.
- **No regressions on the existing CLI / Make targets**. Whatever script you add
  should be additive (`make eval_report` is fine; rewriting `make eval` is not).

---

## Files to read first, in order

1. **This file** — you're here.
2. [`EXPERIMENTS.md`](EXPERIMENTS.md) — the phase-1 results narrative, including
   the headline comparison table and per-run setup descriptions. Read every word
   of the "Headline comparison" and "Takeaways" sections. Note the truth-set
   denominator there (35) is stale; use 37.
3. **One canonical ground-truth file** (eval repo, not here):
   `/Users/bahdan_kapionkin/Documents/Deltix/statgpt-new/data/test_cases/conversational/gtdc/data_query/multi_dataset/unemployment_japan.yaml`
4. [`broad_testcases/patch_artifact.py`](broad_testcases/patch_artifact.py) — to
   see which fields get merged into the artifacts post-hoc. (The yamls in this
   directory are an older copy — ignore them as truth source.)
5. **One job file from R5** to confirm the schema:
   `eval_runs/20260527.…grouped-big-hint-sonnet-rt-11q/jobs/1dc6d65bff374773938e8f.json`.
   This is the highest-quality run; use it as the schema reference.
6. **One job file from R1 baseline** to see the `dataset_queries` shape:
   `eval_runs/data-40/evals/20260526-112654.gtdc.broad-queries/jobs/p5-YThJETy21zF9zbGV9Tw.json`.

---

## Useful invariants

- All 11-query runs are **comparable 1:1 by testcase id**, but the IDs are
  per-run UUIDs, so always **match by query string** (after stripping the
  `!skip_data_query_summarization ` prefix and ` Research thoroughly.` suffix).
- The canonical testcases are in the **eval repo**, at
  `/Users/bahdan_kapionkin/Documents/Deltix/statgpt-new/data/test_cases/conversational/gtdc/data_query/multi_dataset/`.
  Ignore the divergent copies under `statgpt/mcp_lite/broad_testcases/*.yaml`
  in this repo.
- Truth set sum (canonical) = **37**, not 35 (EXPERIMENTS.md is stale) and not
  40 (the old divergent copies). Always recount from the eval-repo yamls
  before publishing numbers.
- Dataset ids in truth-set entries are **unversioned** (`IMF.RES:WEO`); jobs
  use **versioned** ids (`IMF.RES:WEO(9.0.0)`). Strip the `(…)` suffix when
  comparing.
- The mcp_lite server URL is `http://localhost:5000/api/v1/<channel>/mcp-lite/`
  where `<channel>` is `statgpt-gtdc` for these runs. The MCP server runs inside
  the statgpt-backend app; it's not a standalone process.

---

## When in doubt

- If a metric you're computing diverges from `EXPERIMENTS.md`'s numbers by more
  than 1 dataset / 1 percentage point, **stop and ask**. Either the doc is
  stale (possible; tell the user) or your parsing is off (also possible). Don't
  silently file a report that contradicts the canonical numbers.
- If a job file has `failed_attempts > 0` or `status_code != 200`, **call it out
  in the report**; don't quietly include it in averages. The 11-query runs are
  expected to have all-200; surface anything that isn't.
- If a testcase YAML's `candidate_datasets` doesn't match what your parser
  found, **never edit the yaml**. Flag the mismatch in the report and let the
  user decide.

This brief is intentionally complete enough that you should not need to read
this conversation's history. If you find yourself needing context that's only
in chat scrollback, the brief is wrong — say so and I'll patch it.
