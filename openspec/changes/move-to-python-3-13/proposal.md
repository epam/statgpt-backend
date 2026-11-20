# Proposal

## Why

StatGPT is pinned to Python 3.11 (`requires-python = ">=3.11,<3.12"`). 3.11 only gets security fixes, and its support ends in October 2027. Python 3.13 is faster, uses less memory, has newer typing features, and is supported until October 2029. Moving now, while the dependency set is current and the move only takes two dependency bumps, is cheaper than doing it under a deadline. Tracked in issue #826.

## What Changes

- The project requires Python `>=3.13,<3.14`. Local virtualenvs, CI, and the chat and admin images all run on 3.13.
- The chat and admin images are built on `python:3.13-alpine`. The step that removes pip and setuptools from the runtime image points at the `python3.13` paths.
- The `pr.yml`, `release.yml` and `test.yml` workflows use Python 3.13.
- `numpy` moves from 1.26 to 2.x, because 1.26 has no Python 3.13 wheels. `asyncpg` moves from 0.29 to 0.30, the first release that supports 3.13. No other locked dependency changes version. Four backport packages drop out because 3.13 doesn't need them.
- The Makefile's default interpreter becomes `python3.13`, so `make` targets build the venv with the version the project accepts.
- The README setup steps name Python 3.13.
- Two Python 3.12 language changes need small fixes, with no behavior change:
  - flake8 now checks the expressions inside f-strings (PEP 701).
  - A plain `Mock()` no longer passes `isinstance()` against a `@runtime_checkable` protocol.
- The code uses Python 3.12 features where they replace code we wrote ourselves:
  - `itertools.batched` replaces our `utils.batched` helper.
  - Type parameters use PEP 695 syntax (`class Foo[T: Bound]`, `def f[**P, R]`) instead of module-level `TypeVar`/`ParamSpec` and `Generic[...]` bases.
  - `TypeAlias` annotations stay, because a `type` statement changes the JSON schema pydantic generates.
- **BREAKING** (developers only): existing 3.11 virtualenvs have to be recreated, and installing from source on 3.11 or 3.12 is refused. Deployed services see no change; they keep the same images, APIs and settings.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. This change is about the runtime and toolchain and doesn't change system behavior: every API, tool and setting works as before, so no spec requirement changes. The change sets `skip_specs: true`.

## Impact

- Packaging: `pyproject.toml` (`requires-python`, black `target-version`, `numpy` and `asyncpg` constraints), `poetry.lock`.
- Images: `docker/chat.Dockerfile`, `docker/admin.Dockerfile`.
- CI: `.github/workflows/pr.yml`, `.github/workflows/release.yml`, `.github/workflows/test.yml`.
- Tooling and docs: `Makefile`, `README.md`.
- Code:
  - `statgpt/app/chains/data_query/query_constructor/composite.py` (lint only).
  - `itertools.batched`: `statgpt/common/utils/misc.py` and `statgpt/common/utils/__init__.py` (helper removed), `statgpt/app/chains/candidates_selection_batched.py`, `statgpt/common/vectorstore/pg_vector_store/pg_vector_store.py`.
  - PEP 695 type parameters, in about 20 modules under `statgpt/app` (MCP decorators and tools, chain tools, file RAG prefilter, onboarding appenders), `statgpt/admin/services` (`background_tasks.py`, `dataset.py`) and `statgpt/common` (auth, config, caches, utils, schemas, the data and SDMX base classes, Generic RAG client).
- Tests: `tests/unit/app/chains/test_explorer_link_surfaces.py` (mock only), `tests/unit/test_misc_utils.py` (tests of the removed `batched` helper deleted).
- Downstream: [statgpt-helm](https://github.com/epam/statgpt-helm) consumes the images, not the interpreter, so it needs no change unless it hard-codes a `python3.11` path. That gets checked during the rollout.
