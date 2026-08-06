# StatGPT Performance Patterns

Reference for the Performance Optimization Engineer skill. Read when checking
pattern compliance or drafting concrete fixes.

## Async & concurrency

**Do**
- Keep I/O async end-to-end (`async`/`await`, asyncpg, aiohttp).
- Parallelize independent awaits with `asyncio.gather` (or bounded semaphores).
- Offload truly blocking CPU/sync libraries via `asyncio.to_thread` only when necessary and bounded.

**Avoid**
- Sync HTTP/DB/file I/O inside async request handlers or agent tools.
- Unbounded `create_task` storms without backpressure.
- Holding DB sessions/transactions across long LLM or network waits.

## Amplification (N+1)

**Do**
- Batch DB reads/writes and vector upserts (`batch_size` settings where available).
- Fetch SDMX structures/metadata once per flow and reuse via client/dataset caches.
- Collapse per-item LLM/embedding calls into batched APIs when the provider supports it.

**Avoid**
- Loop → query / HTTP / embed one-by-one on request path.
- Re-loading the same dataflow/codelist repeatedly in one request.

## Caching

Prefer existing utilities:
- `AsyncLoadingCache`, `TtlCache` in `statgpt/common/utils/`
- SDMX settings: `client_cache_ttl`, `dataset_cache_ttl`, `cache_dir`
- Auth token caches where already established

**Do**
- Cache expensive, mostly-immutable artifacts (structures, tokens, embeddings config).
- Set explicit TTLs; document invalidation for admin/content updates.

**Avoid**
- Caching user-specific or rapidly changing query results without a clear key/TTL.
- Global mutable caches without concurrency safety.

## Database & vectorstore

**Do**
- Filter early; project only needed columns/fields.
- Use appropriate eager loading (`selectinload` / `joinedload`) to prevent ORM N+1.
- Bound list sizes for `IN` clauses and embedding batches.
- Prefer indexed filters for vector/hybrid search constraints.

**Avoid**
- Loading entire tables or all embeddings into memory for “simple” checks.
- Long-lived sessions; lazy loads after session close (async surprise queries).

## SDMX / external HTTP

**Do**
- Reuse clients; honor configured TTLs and `use_cache=True` where loaders support it.
- Availability checks before large data queries when the pipeline already does so.
- Timeouts on outbound calls; fail fast to the agent with a clear error.

**Avoid**
- Wide open queries without dimension constraints.
- Parsing large XML/JSON payloads repeatedly for the same structure.

## LLM / agent pipelines

**Do**
- Minimize sequential LLM hops on the critical path; parallelize independent tool prep.
- Keep prompts and tool payloads tight; avoid dumping full datasets into context.
- Ground responses from query results rather than re-asking the model for numbers.
- Prefer deterministic/local steps (NER, search) before expensive LLM reasoning.

**Avoid**
- Redundant tool calls that re-fetch the same data in one turn.
- Unbounded candidate lists passed into selection chains—batch/truncate intentionally
  (`CandidatesSelectionBatchedChainFactory` and similar patterns).

## Indexing & batch jobs (CLI / admin)

**Do**
- Process in batches with configurable `batch_size`.
- Bound concurrency for embed/index workers.
- Log progress and per-batch timings; make runs resumable where possible.

**Avoid**
- Loading a full channel/dataset into memory before processing.
- Mixing interactive request-path code with heavy reindex logic without isolation.

## Observability for validation

When suggesting validation, prefer:
- Timed spans around SDMX, DB, vector search, and LLM calls
- Counts: queries per request, HTTP calls per request, tokens per turn
- Before/after comparison on a representative query or reindex subset

Runtime proof beats speculative micro-optimizations.
