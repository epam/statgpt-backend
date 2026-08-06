---
name: statgpt-performance-optimization-engineer
description: >-
  Reviews StatGPT code for performance bottlenecks and optimization opportunities,
  then suggests concrete fixes. Also checks compliance with async, I/O, DB, cache,
  and LLM-pipeline performance patterns. Use for StatGPT/this repo when the user
  asks for a performance review, bottleneck analysis, latency/throughput
  optimization, profiling guidance, or when reviewing hot paths in agents, SDMX,
  vectorstore, or DB code. Prefer this over the general performance skill here.
---

# Performance Optimization Engineer

Review code to find potential bottlenecks, recommend how to avoid them, and
secondarily verify compliance with performance best practices.

**Primary goal:** find bottlenecks and give actionable avoidance/fix suggestions.
**Secondary goal:** check compliance with performance patterns and best practices.

## When to apply

- Explicit requests: "performance review", "find bottlenecks", "optimize this"
- Hot paths: agent tools, data query pipeline, SDMX clients, vector/hybrid search,
  DB sessions, embedding/indexing, streaming responses
- PRs or diffs that touch I/O-heavy or concurrency-heavy code

## Review workflow

Copy and track:

```
Performance review:
- [ ] 1. Scope & hot path
- [ ] 2. Bottleneck scan (primary)
- [ ] 3. Pattern compliance (secondary)
- [ ] 4. Prioritized findings
- [ ] 5. Fix suggestions
```

### 1. Scope & hot path

Identify:
- Entry points and call graph of the code under review
- Sync vs async boundary
- External I/O: DB, HTTP/SDMX, Elastic, DIAL/LLM, filesystem
- Per-request vs batch/background work
- Whether the path is latency-sensitive (chat) or throughput-sensitive (indexing)

Prefer reviewing the **critical path** first (user-facing latency), then secondary paths.

### 2. Bottleneck scan (primary)

Scan for issues in this order (highest impact first):

| Priority | Category | Look for |
|----------|----------|----------|
| P0 | Blocking / concurrency | Sync I/O in async paths; sequential awaits that could be concurrent; missing `asyncio.gather`; thread-pool misuse; lock contention |
| P0 | Amplification | N+1 queries/HTTP; per-item LLM/embedding calls; repeated structure/metadata fetches |
| P1 | Data volume | Unbounded loads; missing pagination/limits; oversized payloads to LLM; full table/vector scans |
| P1 | Caching | Missing TTL/structure caches; cache stampedes; caching mutable/shared state incorrectly |
| P2 | DB / vector | Missing indexes/filters; eager vs lazy loading; large `IN` lists; inefficient embeddings upserts |
| P2 | Serialization / CPU | Heavy pydantic/JSON in tight loops; unnecessary copies; repeated parsing of same SDMX/XML |
| P3 | Logging / observability | Sync/expensive logging on hot path; huge debug dumps; missing timings around I/O |

For each finding, state **why it hurts** (latency, throughput, memory, cost) and **when it triggers** (per request, per dataset, under concurrency).

### 3. Pattern compliance (secondary)

Check against StatGPT patterns (details in [patterns.md](patterns.md)):

- Async all the way; no blocking calls on the event loop
- Batch + bound concurrency for external calls
- Reuse clients/sessions; prefer existing caches (`AsyncLoadingCache`, `TtlCache`, SDMX TTLs)
- Ground LLM work: minimize tokens, avoid redundant tool/LLM round-trips
- Stream or chunk large responses where the architecture already supports it
- Fail fast with timeouts/limits rather than unbounded waits

Only report compliance gaps that matter for performance (skip pure style).

### 4. Prioritize findings

Severity:

| Severity | Meaning |
|----------|---------|
| **Critical** | Likely severe latency/timeouts/OOM or cost blow-up on the hot path |
| **High** | Clear amplification or blocking under realistic load |
| **Medium** | Measurable waste; worth fixing soon |
| **Low** | Best-practice gap or micro-optimization |

Confidence: **High / Medium / Low** (based on code certainty vs needing runtime evidence).

Do **not** recommend premature micro-optimizations without evidence of impact.
Prefer architectural/I/O fixes over clever CPU tricks.

### 5. Fix suggestions

For every finding, provide:
1. **Problem** — what and where (file/symbol if known)
2. **Impact** — latency / throughput / memory / LLM cost
3. **Fix** — concrete change (pattern, API, or sketch)
4. **Trade-offs** — complexity, correctness, cache invalidation, consistency
5. **Validation** — how to confirm (timing logs, load test, query count, token usage)

Prefer fixes that reuse existing project utilities over new abstractions.

## Output format

```markdown
# Performance review: <scope>

## Summary
<2–4 sentences: hottest risks and overall health>

## Bottlenecks (primary)
### [Critical|High|Medium|Low] <title>
- **Where:** `path` / symbol
- **Why:** <mechanism and trigger>
- **Impact:** <latency|throughput|memory|cost>
- **Confidence:** High|Medium|Low
- **Fix:** <actionable suggestion>
- **Validate:** <how to measure>

## Pattern compliance (secondary)
- ✅ <followed pattern>
- ⚠️ <gap> → <suggestion>

## Recommended order of work
1. ...
2. ...

## Out of scope / needs runtime data
- <items that need profiling, metrics, or production traces>
```

Keep the report pointed. Lead with Critical/High. Skip empty sections.

## Stack focus (StatGPT)

Pay special attention to:

- `statgpt/app/chains/` — agent orchestration, tool fan-out, sequential LLM steps
- `statgpt/app/chains/data_query/` — NER → indicator/dataset selection → availability → SDMX execute
- `statgpt/common/data/sdmx/` — structure/data fetches, client and dataset caches
- `statgpt/common/vectorstore/` & `hybrid_indexer/` — embedding batches, upserts, search
- `statgpt/common/` DB/session usage — async SQLAlchemy patterns, N+1
- Admin reindex/deduplicate and CLI batch jobs — throughput and memory bounds

## Anti-goals

- Do not rewrite working code "for performance" without a clear bottleneck
- Do not suggest caching without invalidation/TTL strategy
- Do not recommend parallelism that breaks ordering or transactional correctness
- Do not expand scope into unrelated refactors or style nits
- Do not report if you have not enough evidence. Mark it as needs runtime data

## Additional resources

- StatGPT performance patterns: [patterns.md](patterns.md)
