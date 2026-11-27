# Design

## Context

See proposal.md - Why. The interpreter version is set in five independent places, and nothing keeps them in sync:

```
pyproject.toml       requires-python, [tool.black] target-version
docker/*.Dockerfile  FROM python:3.11-alpine (builder + server), hard-coded lib/python3.11/ cleanup paths
.github/workflows    pr.yml, release.yml (python-version), test.yml (input default)
Makefile             PYTHON ?= python3  ->  poetry env use $(PYTHON)
README.md            install instructions
```

Two locked dependencies don't install on 3.13. `numpy` is pinned `<2.0`, and 1.26 publishes no cp313 wheels; on Alpine it would need a source build. `asyncpg` is pinned `<0.30`, and 0.29 doesn't build against 3.13's C API. Every other locked package already ships 3.13 wheels, including `musllinux` ones for the Alpine images.

## Goals / Non-Goals

**Goals:**
- One supported interpreter (3.13) everywhere: locally, in CI, and in the images.
- The smallest dependency change that installs on 3.13, so any regression points to the interpreter, not to unrelated upgrades.
- The services behave exactly as before.

**Non-Goals:**
- Converting `t.TypeAlias` aliases to `type` statements (see Decisions).
- Switching to 3.12/3.13 APIs that don't remove code: `os.walk` → `Path.walk` (our callers do string path arithmetic), or `asyncio.get_event_loop()` → `get_running_loop()` in coroutines, where the current call already works.
- Fixing deprecations unrelated to the interpreter, e.g. pydantic's `FieldValidationInfo` used in `statgpt/common/schemas/data_query_tool.py`.
- Supporting several Python versions at once (no CI matrix).
- Free-threaded (`3.13t`) builds or the experimental JIT.

## Decisions

### Target 3.13, pinned to one minor version (`>=3.13,<3.14`)
Pinning to one minor version, as the project did for 3.11, keeps local, CI and image environments the same, so the lock file is resolved and tested against exactly one interpreter.
- *Alternative: 3.12.* A smaller step, but less benefit and a support window that is two years shorter. We would have to repeat this work sooner.
- *Alternative: 3.14.* Too new: not every dependency publishes `musllinux` wheels for it yet, and the Alpine images would have to build from source.
- *Alternative: `>=3.13` with no upper bound.* A future minor version could resolve dependencies that were never tested, and the Docker and CI pins would drift from what Poetry accepts.

### Bump only the dependencies that block 3.13
`numpy` becomes `>=2.1.0,<3.0.0` (2.1 is the first release with 3.13 wheels), and `asyncpg` becomes `>=0.30.0,<0.31.0`. Poetry then resolves `numpy` 2.5.5 and `asyncpg` 0.30.0. `pandas` (2.3.3), `pyarrow` and `sdmx1` stay on their current versions, which already support both numpy 2 and Python 3.13. The old comment justifying the numpy 1.x pin (a CVE false positive) is replaced with the reason for the new floor.
- *Alternative: refresh the whole lock (`poetry update`).* That mixes dozens of unrelated upgrades into a runtime migration and makes regressions hard to attribute.
- *Alternative: `asyncpg` up to 0.32.* We get nothing from it now. Later releases can come through the usual dependency PRs.

numpy 2.0 removed aliases like `np.NaN`, `np.float_` and `np.product`. The code only uses `np.ndarray`, `np.floor`, `np.log10`, `np.array` and `np.float32`, which all still exist, so no code change is needed.

### Change the Makefile default to `python3.13`
Every `make` target depends on `init_venv`, which runs `poetry env use $(PYTHON)`. With `PYTHON ?= python3`, any machine where `python3` is a different version fails as soon as `requires-python` is strict. Naming the exact interpreter matches the README instructions. `PYTHON=...` can still override it.

### Fix the two Python 3.12 breakages where they occur
- **PEP 701 (f-string tokenization):** flake8 now sees the expressions inside f-strings, so `{i+1}` triggers E226. The fix is to reformat it as `{i + 1}`. Ignoring E226 globally would hide real cases.
- **`runtime_checkable` protocols use `inspect.getattr_static` since 3.12:** a bare `Mock()` creates attributes dynamically, so it no longer satisfies `ChoiceI` when pydantic checks `isinstance` on `ChainState.choice`. `Mock(spec=ChoiceI)` defines the protocol members statically. Production objects (`Choice`, `NullChoice`) define the methods on their class and are not affected. Making the field `Any` would weaken validation just to suit a test.

### Replace `utils.batched` with `itertools.batched`
Our helper reimplemented `itertools.batched` (added in 3.12), and its docstring said to switch once we could. One difference matters: the stdlib version yields tuples, while ours yielded lists.
- `CandidatesSelectionBatchedChainFactory` passes each batch to the inner selection chains, which type candidates as `list[...]`, check them for truthiness, index `[0]` and hand them to `candidates_to_llm_string`. It wraps each batch in `list(batch)`, so they get exactly what they got before.
- `PgVectorStore.add_documents` only iterates and zips its batches, so it takes the tuples directly. Its annotation becomes `tuple[Document, ...]`.

The helper, its re-export from `statgpt.common.utils` and its tests are removed. Tests of the stdlib function would add nothing.

### Move type parameters to PEP 695 syntax
Every module-level `TypeVar`/`ParamSpec` plus `Generic[...]` base becomes an inline type parameter list: `class DataSet[DataSetConfigType: DataSetConfig, ...](BaseEntity, ABC)`, `def background_task[**Param, RetType](...)`, `type AsyncHandler[**P, T] = ...`.
- **Names stay the same**, so class and function bodies don't change. Only the `_` prefix is dropped (`_T` → `T`), because the parameter is no longer a module global.
- **Each class declares its own parameters.** A module TypeVar that several classes reused (`CategoryType` in the attribute and dimension modules, `T` in the authorizers and caches) becomes a separate parameter on each class.
- **Forward-referenced bounds need no quotes.** PEP 695 evaluates bounds lazily, so `DataSourceHandlerType: DataSourceHandler` works even though `DataSourceHandler` is imported only under `TYPE_CHECKING`.
- **TypeVars that did nothing are dropped instead of converted:**
  - `prefilter.create_chain` used its TypeVar once, as a parameter type, so it takes `type[SingleFilterLLMOutput]`.
  - The `catch_and_log_async` wrapper returned an unbound `T`, so it returns `Any`.
  - `admin/services/dataset.py` imported another module's TypeVar only to annotate a return value, so it uses `base.DataSetConfig`.
- **`OfflineDataSet` passes its type parameters on to `DataSet`** instead of inheriting from a bare `DataSet`, now that it declares them inline.
- *Alternative: leave it for a separate cleanup.* Review asked for it in this change. The conversion is mechanical, and mypy checks it.

### Keep `t.TypeAlias` aliases
A `type X = ...` statement creates a `TypeAliasType`, which pydantic writes as a `$defs` entry plus a `$ref` instead of inlining it. We verified this. Aliases like `DatasetAvailabilityQueriesType` are fields of models used as LLM structured-output schemas, so converting them would change the schemas OpenAI strict mode validates (see #769). Converting only the aliases that pydantic never sees would leave two styles side by side for no real gain.

The one exception is the generic alias `AsyncHandler` in `statgpt/app/mcp/decorators.py`. It was a bare `Callable[P, Coroutine[Any, Any, T]]` built from module-level `P`/`T`, and with those removed a `type AsyncHandler[**P, T] = ...` statement is the only way to keep it generic. pydantic never sees it.

### Drop the image workarounds that 3.13 makes unnecessary
On 3.11, the venv Poetry creates was seeded with old `setuptools` and `wheel`, and the base image installed both into the system `site-packages`. Two image steps dealt with that, and on 3.13 they are no longer needed:
- **The builder's `.venv/bin/pip install "setuptools==84.2.1" "wheel==0.46.2"` is removed**, along with its CVE list (CVE-2026-23949, CVE-2026-24049, CVE-2026-59890, GHSA-xmj3-9gf7-h52w). Since 3.12, `venv`/`virtualenv` seed only pip. `setuptools` is a direct main dependency already locked at 84.2.1, which fixes all four advisories (two are in copies vendored inside setuptools). Its `pyproject.toml` floor is raised from `>=84.1.0` to `>=84.2.1` and commented as a security floor, so the fix no longer depends on the lock alone. Nothing pulls in `wheel`, so the step would only have added a package the service doesn't use.
- **The runtime cleanup no longer mentions `setuptools`/`wheel`.** The 3.12+ base images install only pip, and `ensurepip` bundles only the pip wheel, so `/usr/local/bin/wheel` is dropped from the `rm` and the comment now names pip only. The rest of the step stays: it removes pip, `ensurepip` and the system `site-packages`, which clears pip's CVEs and those of its vendored deps.
- **The `krb5-libs` pin and its CVE list are removed.** The official image computes its runtime packages with `scanelf` from what the compiled modules link. On 3.11, the `nis` module linked `libnsl` → `libtirpc`, and `libtirpc` depends on `libgssapi_krb5` from `krb5-libs`. Python 3.13 removed `nis` (PEP 594), so `krb5-libs` is no longer in the base image, and `apk add --upgrade "krb5-libs>=..."` would *install* it (plus `krb5-conf`, `libverto`, `keyutils-libs`) rather than patch it. Nothing else needs it: no code uses Kerberos, and the `psycopg-binary` wheels bundle their own libraries.
- **The other `apk` upgrades stay.** `libuuid` is still linked by `_uuid`, and musl, libexpat, sqlite, zlib and OpenSSL by the interpreter and its modules. These packages come from the Alpine release under the image, not from the Python version, and `python:3.11-alpine` and `python:3.13-alpine` both use Alpine 3.24. Whether each pin is still needed is a question for the Trivy scan, not for this migration.

## Risks / Trade-offs

- [numpy 2 changes promotion and scalar repr (NEP 50)] → numpy is used only in the value rounding in `statgpt/common/data/base/dataset.py` and the embedding arrays in `pg_vector_store.py`. Their unit tests pass. Integration tests and a Review deployment check real data paths before release.
- [Alpine `musllinux` wheels missing for a dependency] → `poetry install` resolves the cp313 musllinux wheels from the lock, and a source build would show up as a failing image build in CI. The image build is a rollout task.
- [asyncpg 0.30 behavior change] → 0.30 mainly adds 3.13 support and drops 3.8. The integration tests against the `vectordb-test` container cover the DB layer.
- [Developers keep stale 3.11 venvs] → `poetry check --lock` and `poetry env use` fail with a clear version error. The PR description and README give the reset (`make remove_venv && make install_dev`).
- [PEP 695 changes how pydantic generic models (`ListResponse`, the langchain and fastmcp tool bases) are built at runtime] → we snapshotted the chat and admin OpenAPI specs, the `ListResponse[...]` schema and the MRO and fields of all 45 tool classes before and after. They are identical.
- [PEP 695 infers variance, where the old TypeVars were invariant] → that can only loosen type checks. mypy passes over all 505 source files.
- [Helm chart or other deploy config hard-codes a `python3.11` path] → checked as a rollout task. The images keep the same entrypoints, ports and env, so normally nothing changes there.

## Migration Plan

1. Merge to `development`. CI builds and scans both images on 3.13.
2. Deploy to a Review environment and smoke-test the chat (data query, MCP) and the admin API (channel import, discovery upload).
3. Developers recreate their venvs with `make remove_venv && make install_dev`.
4. Rollback: revert the PR. The images and lock file go back to 3.11 together. No data or schema migration is involved.
