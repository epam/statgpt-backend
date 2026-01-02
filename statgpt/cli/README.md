# StatGPT CLI

Interactive command-line interface for StatGPT administration and content management.

## Quick Start

```bash
# 1. Install CLI dependencies
poetry install --with cli

# 2. Configure minimum required settings
export STATGPT_CLI_ADMIN_URL=http://localhost:8000

# 3. Start the interactive CLI
statgpt
```

## Installation

### Prerequisites

- Python 3.11+
- Poetry package manager
- Access to StatGPT Admin API

### Install CLI Dependencies

The CLI uses a separate Poetry dependency group to avoid mixing with production dependencies:

```bash
poetry install --with cli
```

### Verify Installation

```bash
# Using the entry point
statgpt

# Or as a Python module
python -m statgpt.cli
```

You should see:

```
╭─────────────────────────────────────────────────────╮
│                    StatGPT CLI                      │
│         Type 'help' for available commands          │
╰─────────────────────────────────────────────────────╯
statgpt>
```

## Usage

### Starting the CLI

```bash
statgpt
```

Or run as a Python module:

```bash
python -m statgpt.cli
```

### Interactive Features

- **Tab Completion**: Press Tab to see available commands and autocomplete
- **Command History**: Use Up/Down arrows to navigate command history
- **Interactive Prompts**: Commands prompt for missing required options
- **Progress Indicators**: Long-running operations show progress spinners

### Built-in Commands

| Command          | Description                                 |
|------------------|---------------------------------------------|
| `help`           | Show available commands                     |
| `help <command>` | Show detailed help for a command            |
| `settings`       | Show current CLI settings and their sources |
| `exit` / `quit`  | Exit the CLI                                |

## Commands

### Authentication

The CLI caches authentication tokens to avoid repeated logins. Tokens are stored in `~/.statgpt/token_cache.json` with restricted permissions.

#### `auth login`

Authenticate with the admin API and cache the token.

```
statgpt> auth login --method interactive
```

Options:

- `--method` - Authentication method: `interactive` (browser-based) or `system_user` (credentials)

#### `auth logout`

Clear the cached authentication token.

```
statgpt> auth logout
```

#### `auth status`

Show current authentication status.

```
statgpt> auth status
```

### Channel Management

#### `channel list`

List all available channels.

```
statgpt> channel list
```

#### `channel import`

Import a channel from a zip archive.

```
statgpt> channel import --file /path/to/channel.zip --clean
```

Options:

- `--file` - Path to the zip archive (prompts if not provided)
- `--clean` - Clean existing channel data before import
- `--update-datasets` - Update existing datasets
- `--update-data-sources` - Update existing data sources
- `--admin-url` - Override Admin API URL

#### `channel status`

Show preprocessing status of datasets for a specific channel.

```
statgpt> channel status -c my-deployment-id
statgpt> channel status --output-path report.csv
```

Options:

- `-c, --channel` - Channel deployment ID (prompts interactively if not provided)
- `-o, --output-path` - Export status report to CSV file

#### `channel reindex`

Reindex dataset embeddings for a channel.

```
statgpt> channel reindex --channel my-channel --mode all
```

Options:

- `-c, --channel` - Channel deployment ID (prompts if not provided)
- `--mode` - Reindex mode: `all`, `channel`, or `dataset`
- `--dataset-urn` - Dataset URN (required when mode=dataset)

#### `channel deduplicate`

Deduplicate embeddings for a channel.

```
statgpt> channel deduplicate -c my-channel
```

Options:

- `-c, --channel` - Channel deployment ID (prompts if not provided)

### Content Management

#### `content init`

Initialize channels, data sources, datasets, and glossaries from configuration files.

```
statgpt> content init --client-id my-client --skip-glossaries
```

Options:

- `--client-id` - Comma-separated list of client IDs to process
- `--datasets` - Comma-separated list of dataset URNs to process
- `--skip-data` - Skip data sources and datasets processing
- `--skip-glossaries` - Skip glossary terms processing
- `--skip-dial-files` - Skip uploading files to DIAL

### Configuration

#### `config generate`

Generate DIAL Core configuration from remote DIAL deployments.

```
statgpt> config generate --template template.json --config output.json
```

Options:

- `--template` - Path to the template JSON file (required)
- `--config` - Path to output configuration file (required)
- `--applications` - Comma-separated list of application IDs to include

### Utilities

#### `settings`

Display current CLI settings and their sources.

```
statgpt> settings
```

Shows all configured settings with indicators for:

- `(env)` - Value set via environment variable
- `(default)` - Using default value
- `(not set)` - Required but not configured

## Environment Variables

All CLI environment variables are prefixed with `STATGPT_CLI_` to avoid conflicts with application settings.

### Admin API

| Variable                | Required | Description                  | Default                 |
|-------------------------|:--------:|------------------------------|-------------------------|
| `STATGPT_CLI_ADMIN_URL` |    No    | URL of the StatGPT Admin API | `http://localhost:8000` |

### Content Initialization

| Variable                     | Required | Description                               | Default   |
|------------------------------|:--------:|-------------------------------------------|-----------|
| `STATGPT_CLI_CONFIG_DIR`     |   Yes*   | Path to the configuration directory       | -         |
| `STATGPT_CLI_MAX_EMBEDDINGS` |    No    | Maximum embeddings for reindex operations | unlimited |

\* Required for `content init` command

### DIAL Integration

| Variable                          | Required | Description                               | Default |
|-----------------------------------|:--------:|-------------------------------------------|---------|
| `STATGPT_CLI_DIAL_URL`            |    No    | DIAL URL for file uploads                 | -       |
| `STATGPT_CLI_DIAL_API_KEY`        |    No    | DIAL API key for file uploads             | -       |
| `STATGPT_CLI_REMOTE_DIAL_URL`     |   Yes*   | Remote DIAL URL for config generation     | -       |
| `STATGPT_CLI_REMOTE_DIAL_API_KEY` |   Yes*   | Remote DIAL API key for config generation | -       |

\* Required for `config generate` command

### Authentication

The CLI supports pluggable authentication providers. Set `STATGPT_CLI_AUTH_PROVIDER` to select the provider.

| Variable                    | Required | Description                             | Default |
|-----------------------------|:--------:|-----------------------------------------|---------|
| `STATGPT_CLI_AUTH_PROVIDER` |    No    | Authentication provider (`azure`, etc.) | `azure` |

#### Azure Entra ID Provider

| Variable                               | Required | Description                                                        | Default |
|----------------------------------------|:--------:|--------------------------------------------------------------------|---------|
| `STATGPT_CLI_AUTH_AZURE_CLIENT_ID`     |   Yes*   | Azure application/client ID                                        | -       |
| `STATGPT_CLI_AUTH_AZURE_AUTHORITY`     |   Yes*   | Authority URL (e.g., `https://login.microsoftonline.com/{tenant}`) | -       |
| `STATGPT_CLI_AUTH_AZURE_SCOPE`         |   Yes*   | Token scope                                                        | -       |
| `STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET` |  Yes**   | Client secret                                                      | -       |
| `STATGPT_CLI_AUTH_AZURE_USERNAME`      |  Yes**   | System user username                                               | -       |
| `STATGPT_CLI_AUTH_AZURE_PASSWORD`      |  Yes**   | System user password                                               | -       |

\* Required when using `auth login --method interactive` or `auth login --method system_user`
\** Required when using `auth login --method system_user`

#### Adding New Providers

To add a new authentication provider (e.g., Keycloak):

1. Create `statgpt/cli/shared/auth/keycloak.py` implementing `AuthProvider`
2. Add provider-specific settings to `CLISettings` with `auth_keycloak_*` prefix
3. Register the provider in `statgpt/cli/shared/auth/__init__.py`

```python
from statgpt.cli.shared.auth.base import AuthProvider


class KeycloakProvider(AuthProvider):
    @property
    def name(self) -> str:
        return "keycloak"

    def validate_config(self, settings, interactive: bool) -> None:
        # Check required settings
        ...

    def interactive_login(self, settings) -> str:
        # Return access token
        ...

    def system_user_login(self, settings) -> str:
        # Return access token
        ...
```

### General

| Variable                | Required | Description                                                     | Default |
|-------------------------|:--------:|-----------------------------------------------------------------|---------|
| `STATGPT_CLI_LOG_LEVEL` |    No    | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO`  |

## Configuration File

The CLI stores command history in `~/.statgpt/cli_history`.

## Example Workflows

### Initial Setup

```bash
# Set up environment
export STATGPT_CLI_ADMIN_URL=http://localhost:8000
export STATGPT_CLI_CONFIG_DIR=/path/to/config
export STATGPT_CLI_DIAL_URL=http://dial:8080
export STATGPT_CLI_DIAL_API_KEY=your-api-key

# For authentication (if using Azure)
export STATGPT_CLI_AUTH_AZURE_CLIENT_ID=your-client-id
export STATGPT_CLI_AUTH_AZURE_AUTHORITY=https://login.microsoftonline.com/your-tenant
export STATGPT_CLI_AUTH_AZURE_SCOPE=api://your-app/.default

# Start CLI
statgpt
```

### Authentication Workflow

```
statgpt> auth login --method interactive    # Login once (opens browser)
statgpt> auth status                        # Check token status
statgpt> channel import --file export.zip   # Uses cached token automatically
statgpt> channel status                     # Uses cached token automatically
statgpt> auth logout                        # Clear token when done
```

### Content Initialization Workflow

```
statgpt> auth login                         # Login first
statgpt> settings                           # Verify configuration
statgpt> content init --client-id my-client # Initialize all content
statgpt> channel status -c my-channel       # Check dataset status
statgpt> channel reindex -c my-channel --mode all  # Reindex datasets
statgpt> channel deduplicate -c my-channel  # Deduplicate if needed
```

### Channel Import Workflow

```
statgpt> auth login --method interactive
statgpt> channel import --file export.zip --clean
```

## Troubleshooting

### "Module not found" errors

Ensure CLI dependencies are installed:

```bash
poetry install --with cli
```

### Authentication failures

1. Check required environment variables are set:
   ```
   statgpt> settings
   ```

2. For Azure Entra ID, verify:
    - Client ID is correct
    - Authority URL includes your tenant ID
    - Scope matches your API registration

### Connection refused

Verify the Admin API is running and accessible:

```bash
curl http://localhost:8000/health
```

### Command not recognized

Use `help` to see available commands:

```
statgpt> help
```

## Architecture

```
statgpt/cli/
├── __init__.py          # Main entry point
├── __main__.py          # Module runner support
├── repl.py              # Interactive REPL with prompt_toolkit
├── completer.py         # Tab completion
├── commands/            # Command implementations
│   ├── base.py          # Command infrastructure
│   ├── auth.py          # auth login/logout/status
│   ├── channel.py       # channel list/import/status/deduplicate/reindex
│   ├── content.py       # content init
│   ├── config.py        # config generate
│   └── settings.py      # settings display
└── shared/              # Shared utilities
    ├── admin_client.py  # Admin API HTTP client
    ├── console.py       # Rich console helpers
    ├── logging.py       # Logging configuration
    ├── settings.py      # Pydantic settings
    ├── token_cache.py   # Token caching for auth
    └── auth/            # Authentication providers
        ├── base.py      # AuthProvider ABC, AuthResult
        ├── azure.py     # Azure Entra ID
        └── __init__.py  # Provider registry
```
