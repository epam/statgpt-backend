# Tasks

## 1. Packaging and dependencies

- [x] 1.1 In `pyproject.toml`, set `requires-python = ">=3.13,<3.14"` and `[tool.black] target-version = ["py313"]`; verify `grep -n -E 'requires-python|target-version' pyproject.toml` shows only 3.13 values
- [x] 1.2 Raise the `numpy` constraint to `>=2.1.0,<3.0.0` (replacing the old 1.x pin comment with the 3.13 reason) and `asyncpg` to `>=0.30.0,<0.31.0`, and raise the `setuptools` floor to `>=84.2.1` with a security-floor comment; verify by grepping the three lines in `pyproject.toml`
- [x] 1.3 Point the Poetry env at a 3.13 interpreter and run `poetry lock`; verify `poetry check --lock` passes and the `poetry.lock` diff changes only `numpy` and `asyncpg` versions plus removed 3.11-only backports (`async-timeout`, `backports-tarfile`, `importlib-metadata`, `zipp`)
- [x] 1.4 Run `poetry install -E cli -E mcp --with dev`; verify it finishes without source builds and `.venv/bin/python --version` reports 3.13

## 2. Images, CI and tooling

- [x] 2.1 In `docker/chat.Dockerfile` and `docker/admin.Dockerfile`, switch both stages to `python:3.13-alpine` and update the `lib/python3.11/` cleanup paths to `lib/python3.13/`, remove the builder's `setuptools`/`wheel` reinstall and the `/usr/local/bin/wheel` cleanup path (3.13 ships neither), remove the `krb5-libs` pin and its CVE comment (3.13 has no `nis` module, so the base image no longer contains `krb5-libs`), and keep the other `apk` upgrades; verify `grep -n '3\.11' docker/*.Dockerfile` returns nothing
- [x] 2.2 Set `python-version: "3.13"` in `.github/workflows/pr.yml` and `.github/workflows/release.yml`, and the `python-version` input default in `.github/workflows/test.yml`; verify `grep -rn '3\.11' .github/workflows` returns nothing
- [x] 2.3 Change the Makefile default to `PYTHON ?= python3.13`; verify `make lint` creates or reuses the 3.13 venv without passing `PYTHON` on a machine where `python3` is a different version
- [x] 2.4 Update the Python install steps in `README.md` (Homebrew, Chocolatey, PATH check, pyenv) to 3.13; verify `grep -rn -E 'python ?3\.11|py311' --exclude=poetry.lock --exclude-dir=.venv --exclude-dir=.git .` returns nothing

## 3. Python 3.12+ compatibility fixes

- [x] 3.1 Fix the PEP 701 flake8 E226 finding in `statgpt/app/chains/data_query/query_constructor/composite.py` (`{i+1}` -> `{i + 1}`); verify `poetry run flake8 statgpt scripts tests` passes
- [x] 3.2 Use `Mock(spec=ChoiceI)` for the `choice` input in `tests/unit/app/chains/test_explorer_link_surfaces.py`, so it satisfies the `runtime_checkable` protocol under `inspect.getattr_static`; verify the four tests in that file pass
- [x] 3.3 Scan `statgpt`, `scripts` and `tests` for stdlib APIs deprecated or removed in 3.12/3.13 (`datetime.utcnow`/`utcfromtimestamp`, `distutils`, PEP 594 modules, `asyncio.coroutine`, `ast.Num`/`ast.Str`, `pkg_resources`); verify the grep finds no uses that need a change

## 4. Adopt Python 3.12+ features

- [x] 4.1 Replace `utils.batched` with `itertools.batched` in `candidates_selection_batched.py` (wrapping each batch in `list(batch)`) and `pg_vector_store.py` (annotation `tuple[Document, ...]`); remove the helper, its `statgpt.common.utils` export and `TestBatched`; verify `grep -rn -E 'utils\.batched|import batched|    batched,' statgpt tests scripts` returns nothing and the unit tests pass
- [x] 4.2 Convert every module-level `TypeVar`/`ParamSpec` and `Generic[...]` base to PEP 695 type parameters, keeping parameter names; verify `grep -rn -E 'TypeVar\(|ParamSpec\(|Generic\[' --include='*.py' statgpt scripts tests` returns nothing and `make mypy` passes
- [x] 4.3 Drop the TypeVars that did nothing: `prefilter.create_chain` takes `type[SingleFilterLLMOutput]`, the `catch_and_log_async` wrapper returns `Any`, and `admin/services/dataset.py` annotates `base.DataSetConfig`; verify with `make mypy`
- [x] 4.4 Leave the `t.TypeAlias` aliases as they are, and before and after 4.1–4.3 snapshot the chat and admin OpenAPI specs, the `ListResponse[...]` JSON schema and the MRO and fields of every `StatGptTool`/`StatGptMcpTool` subclass; verify the two snapshots are identical

## 5. Integration check

- [x] 5.1 Run `make lint` and `make test_unit` on Python 3.13; verify both pass (flake8, black, isort, autoflake, mypy, import check; all unit tests)
- [x] 5.2 Smoke-test the entry points with `PYDANTIC_V2=True`: import `statgpt.app.app` and run `python -m statgpt.cli help`; verify both start without errors and report numpy 2.x and asyncpg 0.30
- [ ] 5.3 On PR #827, verify that CI builds both images on `python:3.13-alpine`, the Trivy scan passes, and the integration tests (`make test_integration` against the test DB containers) pass
- [ ] 5.4 Check [statgpt-helm](https://github.com/epam/statgpt-helm) for hard-coded `python3.11` paths or interpreter assumptions; verify a search of the chart finds none, or open a follow-up issue there

## Workflow follow-up

- Deploy PR #827 to a Review environment and smoke-test the chat (data query, MCP) and the admin API (channel import, discovery upload) before merging.
- Archive the change with `openspec archive move-to-python-3-13` once the PR is approved. There are no delta specs to sync, because the change sets `skip_specs: true`.
