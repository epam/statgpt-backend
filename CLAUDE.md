# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StatGPT** is an AI-driven Talk-To-Your-Data platform that enables users to interact with official statistics data using natural language. It leverages LLMs to provide relevant data from statistical databases through conversational interfaces.

### Key Capabilities
- Natural language querying of SDMX datasets
- Wide indicator search (semantic + keyword + LLM reasoning)
- Data grounding with hallucination prevention
- Multi-language support for queries and responses
- Automated data visualization with Plotly
- Glossary of terms for consistent terminology

## Version Control

GitHub is used for version control.

Current main branch: `development`

## Commands

### Code Quality
```bash
make format              # Format code (autoflake, black, isort)
make lint                # Run all linters (flake8, black, isort, autoflake, mypy)
make mypy                # Run only mypy type checking
```

### Testing
```bash
make test                # Run all tests (unit + integration)
make test_unit           # Run unit tests only
make test_integration    # Run integration tests (requires test DB containers)
make test_db_migrate     # Run migrations on test database
```

### Database Operations
```bash
make db_migrate          # Apply alembic migrations
make db_downgrade        # Rollback last migration
make db_autogenerate MESSAGE="Your migration message"  # Generate new migration
```

### Localization
```bash
make extract_messages    # Extract translatable strings from formatters
make update_messages     # Update .po files from template
make compile_messages    # Compile .po files to .mo files
make locales             # Shorthand for compile_messages
```

### Virtual Environment
```bash
make install_dev         # Install dev dependencies
make install_all         # Install dev + experiments dependencies
make remove_venv         # Remove and recreate venv
```

### CLI (Interactive Command-Line Interface)
```bash
poetry install --with cli   # Install CLI dependencies
statgpt                     # Start interactive CLI REPL
python -m statgpt.cli       # Alternative: run as module
```

CLI uses separate dependency group and `STATGPT_CLI_*` prefixed environment variables.
See `statgpt/cli/README.md` for full documentation.

**Available Commands:**
| Command | Description |
|---------|-------------|
| `auth login` | Authenticate with the admin API |
| `auth logout` | Clear cached authentication token |
| `auth status` | Show current authentication status |
| `channel list` | List all available channels |
| `channel import` | Import channel from zip archive |
| `channel status` | Show dataset preprocessing status |
| `channel deduplicate` | Deduplicate embeddings for a channel |
| `channel reindex` | Reindex dataset embeddings for a channel |
| `content init` | Initialize channels, data sources, datasets, glossaries |
| `settings` | Show current CLI settings |

## Architecture Overview

### Project Structure

```
statgpt/
├── app/                 # Chat Backend (DIAL Application)
│   ├── application/     # App factory and DIAL app setup
│   ├── chains/          # LangChain orchestration and agent tools
│   │   ├── data_query/  # SDMX data query tool
│   │   ├── file_rags/   # Publications RAG tool
│   │   ├── web_search/  # Web search tool
│   │   ├── datasets_meta/       # Available datasets tool
│   │   ├── glossary_tools.py    # Glossary tools
│   │   └── supreme_agent.py     # Main agent orchestrator
│   ├── schemas/         # Pydantic models
│   ├── services/        # Business logic
│   ├── settings/        # Pydantic Settings configuration
│   └── utils/           # Utilities (formatters with i18n)
├── admin/               # Admin Backend (FastAPI standalone)
│   ├── routers/         # API routes (channels, datasets, data sources, glossary)
│   ├── services/        # Admin business logic
│   ├── auth/            # OIDC authentication
│   ├── alembic/         # Database migrations
│   └── settings/        # Admin configuration
├── common/              # Shared code
│   ├── models/          # SQLAlchemy database models
│   ├── data/            # Data access layer and SDMX handling
│   │   ├── sdmx/        # SDMX protocol implementation (v2.1)
│   │   └── quanthub/    # QuantHub SDMX provider
│   ├── vectorstore/     # PGVector storage implementation
│   ├── hybrid_indexer/  # Vector indexing for semantic search
│   ├── services/        # Shared services
│   └── schemas/         # Shared Pydantic models
└── cli/                 # Interactive command-line interface
    ├── __init__.py      # Package init
    ├── __main__.py      # Entry point
    ├── repl.py          # Interactive REPL with prompt_toolkit
    ├── completer.py     # Tab completion
    ├── commands/        # Command implementations
    │   ├── base.py      # Command/CommandGroup base classes
    │   ├── auth.py      # auth login/logout/status
    │   ├── channel.py   # channel list/import/status/deduplicate/reindex
    │   ├── content.py   # content init
    │   └── settings.py  # settings command
    └── shared/          # Shared utilities
        ├── admin_client.py  # Admin API client
        ├── console.py       # Rich console output helpers
        ├── logging.py       # Logging configuration
        ├── prompts.py       # Interactive prompts (checkbox selector)
        ├── settings.py      # CLI settings (Pydantic)
        ├── token_cache.py   # Token caching
        └── auth/            # Pluggable authentication providers
            ├── base.py      # Auth provider interface
            └── azure.py     # Azure AD authentication

tests/
├── unit/                # Unit tests (no external dependencies)
└── integration/         # Integration tests (requires test DB containers)
```

### Agent Architecture

StatGPT uses a **tool-calling agent** approach:
- Main agent (`supreme_agent.py`) orchestrates all tools
- History consists of **static** (system prompt, predefined calls) and **dynamic** (user queries, tool calls, responses) blocks
- All data responses are grounded in actual query results to prevent hallucinations

### Available Agent Tools

| Category         | Tool                   | Purpose                                              |
|------------------|------------------------|------------------------------------------------------|
| **Data**         | Available Datasets     | List datasets with metadata                          |
| **Data**         | Data Query             | Build and execute SDMX queries from natural language |
| **Publications** | Available Publications | List publication types                               |
| **Publications** | Publications RAG       | Query publications using RAG                         |
| **Glossary**     | Glossary Terms         | List available terms                                 |
| **Glossary**     | Glossary Definitions   | Retrieve term definitions                            |
| **Web**          | Web Search             | Search and retrieve web content                      |

### Data Query Pipeline

The data query tool (`statgpt/app/chains/data_query/`) implements:
1. **Query Normalization** - Process user input
2. **Named Entity Recognition** - Extract countries, time periods
3. **Indicator Selection** - Semantic + keyword search for indicators
4. **Dataset Selection** - Identify appropriate datasets
5. **Availability Queries** - Verify data availability
6. **Query Execution** - Fetch and format SDMX data

### Key Design Patterns

- **DIAL SDK Integration**: Built on `aidial-sdk` for platform integration
- **Tool-Calling Architecture**: Structured tool calls instead of code generation
- **Multi-Layer Search**: Keyword + semantic + LLM reasoning
- **Grounded Responses**: All data responses cite sources and use exact values
- **Async Architecture**: Async/await throughout (asyncpg, aiohttp)
- **Pydantic v2**: All validation uses Pydantic models

## Code Style Requirements

### Python Style
- `snake_case` for functions/methods
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Protected visibility (`_`) by default
- Modern type hints: `list[str]`, `dict[str, int]`, `str | None`
- Import from `collections.abc` for abstract types
- Use `typing.Self` for factory methods

### Pydantic Best Practices
- Use Pydantic models for validation (not dicts)
- Use `Field(default_factory=list)` for mutable defaults
- Validate complex function arguments with Pydantic models

### Testing Guidelines
- Test database uses Docker containers (`vectordb-test`, `elasticsearch-test`)
- Integration tests require `TEST_DATABASE_*` environment variables
- Use pytest fixtures for common test setup

## Environment Setup

### Required Services
- PostgreSQL with pgvector extension
- ElasticSearch (optional, for hybrid search)
- AI DIAL Core deployment

### Key Environment Variables
- `DIAL_URL` - DIAL Core URL
- `DIAL_API_KEY` - API key for DIAL Core
- `PGVECTOR_*` - Database connection settings
- `ELASTIC_*` - ElasticSearch settings (optional)
- `STATGPT_CLI_*` - CLI-specific settings (see `statgpt/cli/README.md`)
- See `statgpt/common/README.md`, `statgpt/app/README.md`, `statgpt/admin/README.md`

## Development Workflow

1. Run `make format` before committing
2. Ensure `make lint` passes
3. Run relevant tests based on changes
4. Update alembic migrations when modifying models
5. Follow existing patterns in neighboring files
6. See `CODE_STYLE.md` for detailed style guidelines

## Related Repositories

- [StatGPT Admin Frontend](https://github.com/epam/statgpt-admin-frontend) - Admin UI (React/Next.js)
- [StatGPT Portal Frontend](https://github.com/epam/statgpt-portal-frontend) - User portal UI library
- [StatGPT Helm](https://github.com/epam/statgpt-helm) - Kubernetes deployment charts
