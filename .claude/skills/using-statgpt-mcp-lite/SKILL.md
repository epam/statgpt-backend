---
name: using-statgpt-mcp-lite
description: Answers SDMX/official-statistics questions by driving the statgpt-mcp-lite MCP server — locates the right dataset(s), resolves dimension codes, verifies data availability, fetches observations. Use when the user asks a quantitative question about official statistics (national accounts, prices, labour, trade, finance, balance of payments, debt, reserves, etc.) and the `statgpt-mcp-lite` MCP server is registered for the session.
---

# Using statgpt-mcp-lite

The `statgpt-mcp-lite` MCP server exposes a small set of SDMX-style
primitives: dataset discovery, indicator search, code search,
availability checks, and data fetch. This skill captures the workflow
and judgment calls that aren't in the tool descriptions themselves.

All tool references below use the fully-qualified form
`statgpt-mcp-lite:<tool>` to avoid resolution errors when other MCP
servers are also registered.

## When this applies

- The user asks about a real-world statistical quantity ("inflation in
  X", "labour force participation of Y", "debt-to-GDP for Z", etc.).
- The `statgpt-mcp-lite` MCP server is registered in the session.
- The question is *about the data*, not about the system itself.

## Tool surface (one-liner each)

Read each tool's description in the registered tool list for the
authoritative schema. At a glance:

- `statgpt-mcp-lite:list_datasets` — what's in the channel.
- `statgpt-mcp-lite:dataset_structure` — dimensions of one dataset.
- `statgpt-mcp-lite:sample_dim_values` — peek/list values of a non-indicator dim.
- `statgpt-mcp-lite:search_indicators` — cross-dataset concept search, returns `datasets[]` groups.
- `statgpt-mcp-lite:search_codes` — atomic dim-value codes within one dataset.
- `statgpt-mcp-lite:availability_query` — which dim values are reachable under a partial filter.
- `statgpt-mcp-lite:execute_sdmx_query` — fetch observations.
- `statgpt-mcp-lite:list_glossary_terms` / `statgpt-mcp-lite:get_glossary_term` — channel-specific vocabulary.

## Canonical workflow

**Research thoroughly.** A single `search_indicators` call is rarely
enough on a generic question. Treat discovery as iterative: probe the
channel from multiple angles before settling on a final selection.

Work through these phases **internally**:

discover → broaden → resolve → verify → fetch → answer → **cite** (each detailed below).

**Don't narrate the plumbing; do explain the choices.** Concretely:

- **Cut procedural play-by-play.** No "Task progress" checklist, and no
  step announcements like *"Now let me resolve the codes"*, *"Now verify
  availability"*, *"Now fetch from each"*, *"Let me broaden"*, *"Now let me
  visualise"*. The tool calls are already visible to the user — narrating that
  you're about to make them is pure noise. Don't announce phases or tools.
- **Keep the selection reasoning.** *Which* datasets you chose, which candidates
  you **dropped, and why** (one line each) — that judgment is genuinely useful.
  Put it in the **answer**, not as running commentary between tool calls.
- **Always finish with the Sources block (step 6 — required).** Cutting the
  procedural chatter does **not** mean cutting the citation: every answer must
  **end** with a standalone, labeled `Sources:` section (dataset + each indicator
  as `code (name)`, per step 6). A chart caption does not replace it.

Rule of thumb: the user should see *what you decided and why* and *where the
numbers came from* — never *what you're about to do next*.

### 1. Discover — use the verbatim user question first

**The first `search_indicators` call must use the user's question
verbatim, with the default `top_k` (no override).** Do not compress to
a noun phrase, do not paraphrase, do not lower `top_k`. The
natural-language framing carries useful semantic context that is lost
when the query is shortened ("What is the GDP of India?" embeds
differently from "GDP"). Empirically this single rule recovers most of
the multi-dataset truth set; the agent's instinct to "extract the
concept" actively loses recall.

The result is grouped by dataset (`datasets[]`, sorted by `best_score`).
Each match carries `dimensions` — the indicator key ready for
`statgpt-mcp-lite:execute_sdmx_query.selection`.

If this call returns nothing relevant, fall back to
`statgpt-mcp-lite:list_datasets` and `statgpt-mcp-lite:dataset_structure`
to scan manually.

### 2. Broaden — research thoroughly

The verbatim call covers most cases. Paraphrases are for the *rest*:
borderline datasets that exist in the channel but use methodology
vocabulary the user didn't. Add at least one follow-up
`search_indicators` call with **`top_k=100`** (wider net for hard-to-
reach datasets) using a methodology-flavoured paraphrase. Derive the
paraphrase from the concept in front of you — general techniques:

- swap nouns within the concept (rate ↔ level, stock ↔ flow, gross ↔ net),
- restate it in an alternate statistical framing the user didn't use
  (e.g. a national-accounts, balance-of-payments, monetary, or fiscal
  framing of the same quantity),
- search a narrower sub-aspect or component of the concept.

These are *techniques*, not a lookup table — work out the right rephrase
from the question and from what the channel's own results return; do not
rely on a memorised concept→dataset mapping.

Merge groups from all calls. Then **default to inclusion**: every
distinct dataset that appears in any merged result is a candidate
worth keeping unless you can articulate a concrete reason for
exclusion (see the *Broadening: multi-dataset coverage* section
below for the few valid exclusion reasons). **Silent omission —
seeing a dataset in results and not picking it without saying why —
is the most common failure mode on broad questions.**

### 3. Resolve entities to dim codes

Country names, sector labels, instrument types and similar atomic
values are dataset-specific (`USA` ≠ `US`, `BRA` ≠ `BR`, sector codes
differ across datasets). Always resolve via
`statgpt-mcp-lite:search_codes` or `statgpt-mcp-lite:sample_dim_values`;
never invent codes.

When the same entity could live in multiple dims of one dataset
(e.g. "reporter country" vs "counterpart country"), call
`statgpt-mcp-lite:search_codes` without `dim_id` to see which dim
contains it.

### 4. Verify availability

Before `statgpt-mcp-lite:execute_sdmx_query`, call
`statgpt-mcp-lite:availability_query` with the partial filter to check
the combination has observations. Empty result = the selected
combination doesn't exist — fix the filter rather than running a
doomed fetch.

**Do this for *every* candidate dataset you kept — not just the top one.**
Each dataset needs its own `search_codes` resolution and `availability_query`;
skipping the lower-ranked candidates is the most common miss. A complete answer
from one dataset (e.g. WEO) does **not** make the others optional — the task is to
**cover every candidate**, not to stop at "good enough" or "sufficient". Resolve +
verify all of them before answering. A fetch gap on one candidate is a data-plane
note, **not** a reason to skip the remaining candidates.

### 5. Fetch observations

`statgpt-mcp-lite:execute_sdmx_query(dataset_id, selection, time_start, time_end, limit)`.
`selection` accepts a scalar or list per dim; an empty list = "all
values of that dim". `time_start`/`time_end` accept bare years
(auto-expanded) or SDMX-period strings.

### 6. Cite

**Every answer ends with a standalone, labeled `Sources:` block** — this is
required output, never optional and never replaced by a chart caption. In it,
name for each source **the dataset and every indicator** you used from it that
produced the numbers — not just the dataset name. Give each indicator as **code + human-readable name**, not the code alone.
A single dataset can contribute **several** indicators, so list all of them
under that dataset (e.g. *"IMF.RES:WEO — NGDP_RPCH (GDP, constant prices, %
change), PCPIPCH (CPI, period avg, % change)"*); across multiple datasets, give
each dataset with its own indicator list. Don't paraphrase observations away
from their source.

## Broadening: multi-dataset coverage

**A single concept often lives in several distinct datasets here**,
each with a different framing (definition, periodicity, unit,
valuation, sectoral aggregation). The `datasets[]` shape returned by
`search_indicators` is designed to make this visible — read it
end-to-end, not just the top group.

### Default behaviour: broaden every time

**Broadening is the default, not the exception.** For every question,
run at least one paraphrased follow-up `search_indicators(top_k=100)`
with a methodology-flavoured rephrase (see the table in step 2). Then
merge groups from all calls and **default to inclusion**: every
distinct on-topic dataset that appears in any merged result becomes a
candidate `(dataset_id, selection)` pair in the final answer.

This applies even when the verbatim call already returned several
plausible candidates. Empirically the agent's instinct to "stop once
the result looks reasonable" is the single biggest source of recall
loss. Forcing a methodology paraphrase on every case closes that gap —
the cost is one extra `search_indicators` call.

Each merged group carries a different lens on the concept — different
periodicity, unit, gross-vs-net, level-vs-rate, flow-vs-stock, sectoral
aggregation, **and scope (domestic vs cross-border, resident vs external,
central-bank vs whole-banking-sector)**. Two candidates is typical for
moderate-scope questions; four or more is normal for genuinely vague
ones. **A high score on group 2+ isn't a duplicate of group 1; it's a
different lens on the same concept — keep it.**

**A different scope or framing is a lens to KEEP, not a reason to drop.**
"Cross-border" vs "domestic", "external" vs "resident", a secondary or
less-direct measure — these *are* the multi-dataset coverage you are here
to surface. Do **not** exclude a dataset because it is "less directly
relevant", a "different angle", "cross-border", or "secondary": those are
inclusions.

### Edge cases (the only genuine reasons to drop a candidate)

The bar is high. Drop a merged-result dataset **only** when one of these
holds, and **state the reason explicitly**:

- The user's question **explicitly pins** a framing (names a specific unit,
  transformation, or publisher) **and** the group covers a *different* one.
  A framing the user did **not** specify is never grounds for exclusion —
  keep all framings when the question is unspecific.
- The group is a genuine **lexical false match** — high text score but a
  *different concept entirely* (e.g. "credit" matching a balance-of-payments
  accounting entry, not bank lending).
- The group **republishes** another with identical numbers (`available_in`).

Anything else is **not** a valid reason to drop — including "I'd answer
with fewer", "this one is less central", or "I couldn't fetch it right
now". **Availability/fetch failures are a data-plane issue, not a relevance
signal**: keep the dataset and note the fetch gap rather than dropping it.

Silent omission — seeing a dataset in results and not picking it without
saying why — is the most common failure mode here.

## Common failure modes

- **Stopping at top-1 group on broad questions.** See *Broadening*.
- **Inventing dim codes.** Codes vary per dataset; always resolve them.
- **Skipping `availability_query`.** Selections that look right at the
  dim-code level can still have no observations.
- **Conflating `search_indicators` and `search_codes`.** Indicators are
  compound; their `dimensions` is the pin set. Atomic codes (country,
  sector, instrument) come from `search_codes`.
- **Treating `score` as cross-query.** `score` is per-query
  max-relative in [0, 1]. Two queries' top scores are not comparable.

