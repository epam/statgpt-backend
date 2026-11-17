# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**StatGPT** is an AI-driven Talk-To-Your-Data platform that enables users to interact with official statistics data using natural language. It leverages LLMs to provide relevant data from statistical databases through conversational interfaces.

## Project Structure

```
statgpt/
├── app/       # Chat backend (DIAL application): agent, tools, MCP
├── admin/     # Admin backend (standalone FastAPI): API, background jobs, migrations, MCP
├── common/    # Shared by app and admin: DB models, SDMX data access, vector store, indexing
└── cli/       # Interactive CLI that talks to the admin API
tests/
├── unit/          # No external dependencies
└── integration/   # Requires test DB containers
```

Entry points:
- `statgpt/app/chains/supreme_agent.py` - main agent orchestrator; tools live in `statgpt/app/chains/`
- `statgpt/admin/alembic/` - database migrations
- `statgpt/common/models/` - SQLAlchemy models
- `statgpt/common/data/` - data source layer: shared abstractions and different SDMX implementations

## Request Paths

How requests reach the backends in this repository:

1. **Admins:** StatGPT Admin Frontend / StatGPT CLI -> StatGPT Admin Backend (`statgpt/admin`)
2. **Chat users:** DIAL Chat (including DIAL Chat Overlay) / StatGPT Portal Frontend -> DIAL Core -> StatGPT Chat Backend (`statgpt/app`)
3. **MCP users** (e.g. ChatGPT, Claude Desktop): third-party agent -> DIAL Core -> StatGPT Chat Backend (`statgpt/app`)

## Development Workflow

1. Before writing or changing code, read and follow `CODE_STYLE.md`; follow existing patterns in neighboring files.
2. Before changing existing behavior, check the relevant specs in `openspec/specs/` (the source of truth, managed with [OpenSpec](https://github.com/Fission-AI/OpenSpec)) and keep them in sync with the implementation.
3. When changing models, follow [Database Changes](#database-changes).
4. Run `make format` and make sure `make lint` passes.
5. Run the tests relevant to the change.
6. The repository is on GitHub; the main branch is `development`. Branch from it and open PRs against it.

## Commands

```bash
make format              # Format code (autoflake, black, isort)
make lint                # Run all linters (flake8, black, isort, autoflake, mypy)
make test_unit           # Unit tests
make test_integration    # Integration tests; needs the test DB from docker-compose.test.yml and TEST_DATABASE_* env vars (see README.md "Run Tests")
```

Other targets (migrations, localization, venv, CLI) are in the `Makefile`; see `README.md` and `statgpt/cli/README.md`.

## Database Changes

1. Change the models in `statgpt/common/models/models.py`.
2. Generate a migration: `make db_autogenerate MESSAGE="..."` or directly `alembic -c alembic.ini revision --autogenerate -m "..."`. If alembic fails, stop and report; do not write migrations by hand.
3. Review and fix the generated migration; autogenerate is not reliable.
4. Set `ALEMBIC_TARGET_VERSION` in `statgpt/common/config/versions.py` to the new revision; the health check fails on a mismatch.

Migrations are shipped in and applied from the admin Docker image (`docker/admin.Dockerfile`).

## Environment Setup

Local setup is in `README.md`; environment variables are documented in the `Environment variables` section of each component's README (`statgpt/{common,app,admin,cli}/README.md`).

## Related Repositories

| Repository | What it is | Check it when |
|---|---|---|
| [StatGPT Admin Frontend](https://github.com/epam/statgpt-admin-frontend) | Admin UI (React/Next.js) | changing admin API endpoints or schemas |
| [StatGPT Portal Frontend](https://github.com/epam/statgpt-portal-frontend) | User portal UI library | changing chat response format, attachments or custom content |
| [StatGPT Helm](https://github.com/epam/statgpt-helm) | Kubernetes deployment charts | adding, renaming or removing environment variables or services |
| [DIAL Deep Research](https://github.com/epam/ai-dial-deep-research-backend) | DIAL deep research app behind the Deep Research tool | changing the Deep Research tool or what it sends |
| [DIAL Core](https://github.com/epam/ai-dial-core) | DIAL platform gateway that StatGPT runs behind | debugging routing, auth, rate limits or how DIAL Core passes app errors |
| [DIAL Chat](https://github.com/epam/ai-dial-chat) | DIAL chat UI | changing how responses, stages or attachments render
