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

**Prerequisites:** Python 3.11+, Poetry, access to StatGPT Admin API

```bash
poetry install --with cli
```

The CLI uses a separate Poetry dependency group to avoid mixing with production dependencies.

### Interactive Features

- **Tab Completion**: Press Tab to see available commands and autocomplete
- **Command History**: Use Up/Down arrows to navigate command history
- **Interactive Prompts**: Commands prompt for missing required options
- **Progress Indicators**: Long-running operations show progress spinners

### Built-in Commands

| Command          | Description                      |
|------------------|----------------------------------|
| `help`           | Show available commands          |
| `help <command>` | Show detailed help for a command |
| `exit` / `quit`  | Exit the CLI                     |

## Commands

| Command               | Description                           |
|-----------------------|---------------------------------------|
| `auth login`          | Authenticate with admin API           |
| `auth logout`         | Clear cached authentication token     |
| `auth status`         | Show current authentication status    |
| `channel list`        | List all available channels           |
| `channel import`      | Import channel from zip archive       |
| `channel status`      | Show dataset preprocessing status     |
| `channel reindex`     | Reindex dataset embeddings            |
| `channel deduplicate` | Deduplicate embeddings for a channel  |
| `content init`        | Initialize content from config files  |
| `settings`            | Show current CLI settings and sources |

### auth login

```
statgpt> auth login --method interactive
```

- `--method` - `interactive` (browser-based) or `system_user` (credentials)

### channel import

```
statgpt> channel import --file /path/to/channel.zip --clean
```

| Option                  | Description                              |
|-------------------------|------------------------------------------|
| `--file`                | Path to zip archive (prompts if omitted) |
| `--clean`               | Clean existing data before import        |
| `--update-datasets`     | Update existing datasets                 |
| `--update-data-sources` | Update existing data sources             |

### channel status

```
statgpt> channel status -c my-deployment-id -o report.csv
```

| Option              | Description                                    |
|---------------------|------------------------------------------------|
| `-c, --channel`     | Channel deployment ID (interactive if omitted) |
| `-o, --output-path` | Export status report to CSV file               |

### channel reindex

```
statgpt> channel reindex -c my-channel --mode all
```

| Option          | Description                              |
|-----------------|------------------------------------------|
| `-c, --channel` | Channel deployment ID                    |
| `--mode`        | `all`, `channel`, or `dataset`           |
| `--dataset-urn` | Dataset URN (required when mode=dataset) |

### content init

```
statgpt> content init                             # interactive client selection
statgpt> content init --client-id my-client       # specific client
statgpt> content init --only channels,glossaries  # specific components
statgpt> content init -y                          # skip all prompts
```

| Option        | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| `--client-id` | Comma-separated client IDs (interactive selection if omitted)            |
| `--datasets`  | Comma-separated dataset URNs to process                                  |
| `-o, --only`  | Components: `channels`, `datasources`, `datasets`, `glossaries`, `files` |
| `-y, --yes`   | Skip all confirmation prompts                                            |

**Notes:**

- Specifying `datasets` automatically includes `datasources` (dependency)
- Interactive selectors: arrow keys to navigate, space to toggle, enter to confirm

## Environment Variables

All variables are prefixed with `STATGPT_CLI_`.

| Variable                   | Required | Description                         | Default                 |
|----------------------------|:--------:|-------------------------------------|-------------------------|
| **Admin API**              |          |                                     |                         |
| `ADMIN_URL`                |    No    | StatGPT Admin API URL               | `http://localhost:8000` |
| **Content Init**           |          |                                     |                         |
| `CONFIG_DIR`               |   Yes*   | Configuration directory path        | -                       |
| `MAX_EMBEDDINGS`           |    No    | Max embeddings for reindex          | unlimited               |
| **DIAL Integration**       |          |                                     |                         |
| `DIAL_URL`                 |    No    | DIAL URL for file uploads           | -                       |
| `DIAL_API_KEY`             |    No    | DIAL API key                        | -                       |
| **Authentication**         |          |                                     |                         |
| `AUTH_PROVIDER`            |    No    | Auth provider (`azure`, etc.)       | `azure`                 |
| `AUTH_AZURE_CLIENT_ID`     |  Yes**   | Azure application/client ID         | -                       |
| `AUTH_AZURE_AUTHORITY`     |  Yes**   | Authority URL (includes tenant)     | -                       |
| `AUTH_AZURE_SCOPE`         |  Yes**   | Token scope                         | -                       |
| `AUTH_AZURE_CLIENT_SECRET` |  Yes***  | Client secret (system_user only)    | -                       |
| `AUTH_AZURE_USERNAME`      |  Yes***  | Username (system_user only)         | -                       |
| `AUTH_AZURE_PASSWORD`      |  Yes***  | Password (system_user only)         | -                       |
| **General**                |          |                                     |                         |
| `LOG_LEVEL`                |    No    | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO`                  |
| `DATA_DIR`                 |    No    | CLI data directory                  | `~/.statgpt`            |

\* Required for `content init` command
\** Required for `auth login`
\*** Required for `auth login --method system_user`

## Data Directory

The CLI stores persistent data in `~/.statgpt/` (configurable via `STATGPT_CLI_DATA_DIR`):

| File               | Description                                           |
|--------------------|-------------------------------------------------------|
| `cli_history`      | Command history for up/down arrow navigation          |
| `token_cache.json` | Cached authentication tokens (restricted permissions) |

## Example Workflows

### Initial Setup

```bash
export STATGPT_CLI_ADMIN_URL=http://localhost:8000
export STATGPT_CLI_CONFIG_DIR=/path/to/config
export STATGPT_CLI_AUTH_AZURE_CLIENT_ID=your-client-id
export STATGPT_CLI_AUTH_AZURE_AUTHORITY=https://login.microsoftonline.com/your-tenant
export STATGPT_CLI_AUTH_AZURE_SCOPE=api://your-app/.default

statgpt
```

### Content Initialization

```
statgpt> auth login
statgpt> settings                           # verify configuration
statgpt> content init --client-id my-client
statgpt> channel status -c my-channel
statgpt> channel reindex -c my-channel --mode all
statgpt> channel deduplicate -c my-channel
```

### Channel Import

```
statgpt> auth login --method interactive
statgpt> channel import --file export.zip --clean
```

## Troubleshooting

| Problem                 | Solution                                              |
|-------------------------|-------------------------------------------------------|
| Module not found        | Run `poetry install --with cli`                       |
| Authentication failures | Check `settings` command, verify Azure config         |
| Connection refused      | Verify Admin API: `curl http://localhost:8000/health` |
| Command not recognized  | Run `help` to see available commands                  |
