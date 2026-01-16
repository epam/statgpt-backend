# StatGPT CLI

Interactive command-line interface for StatGPT administration and content management. Alternative to using the Admin
API directly and to [Admin web interface](https://github.com/epam/statgpt-admin-frontend).

## Prerequisites

Before using the CLI:

1. **Admin backend running** - Start with `make statgpt_admin`
2. **Environment variables** - Set required variables (see [Environment Variables](#environment-variables))

## Quick Start

```bash
make statgpt_cli
```

## Global Flags

| Flag                | Description                                          | Modes        |
|---------------------|------------------------------------------------------|--------------|
| `--debug`           | Show full stack trace on errors                      | Direct, REPL |
| `--non-interactive` | Disable interactive prompts (fail if input required) | Direct only  |

### Non-Interactive Mode

Use `--non-interactive` for CI/CD pipelines or scripts where interactive prompts are not possible:

```bash
# Will fail with helpful message if required parameters are missing
statgpt --non-interactive content init --client-id my-client --datasets urn1,urn2 -y

# Combined with -y to skip all confirmations
statgpt --non-interactive channel import --file archive.zip
```

**Note:** In non-interactive mode:

- Commands fail fast with helpful error messages showing required parameters
- Use `-y/--yes` to skip confirmation prompts
- All required parameters must be provided via command-line arguments

### Debug Mode

Use `--debug` to see full stack traces when errors occur:

```bash
# Direct mode
statgpt --debug content init --client-id my-client

# REPL mode - start REPL with debug enabled
statgpt --debug
```

This is useful for troubleshooting and reporting bugs.

## Interactive Features

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
statgpt> content init -y                          # skip all prompts, process ALL datasets
statgpt> content init --datasets urn1,urn2        # specific datasets only
```

| Option        | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| `--client-id` | Comma-separated client IDs (interactive selection if omitted)            |
| `--datasets`  | Comma-separated dataset URNs to process                                  |
| `-o, --only`  | Components: `channels`, `datasources`, `datasets`, `glossaries`, `files` |
| `-y, --yes`   | Skip all confirmation prompts and process ALL available content          |

**Notes:**

- Specifying `datasets` automatically includes `datasources` (dependency)
- Interactive selectors: arrow keys to navigate, space to toggle, enter to confirm
- **Important:** Using `-y` without `--datasets` processes ALL available datasets for selected clients
- For non-interactive use in CI/CD, always specify `--client-id` and optionally `--datasets`:
  ```bash
  statgpt --non-interactive content init --client-id my-client --datasets urn1,urn2 -y
  ```

## Environment Variables

All variables are prefixed with `STATGPT_CLI_`.

| Variable                      | Required | Description                                       | Default                               |
|-------------------------------|:--------:|---------------------------------------------------|---------------------------------------|
| **Admin API**                 |          |                                                   |                                       |
| `ADMIN_URL`                   |    No    | StatGPT Admin API URL                             | `http://localhost:8000`               |
| **Content Init**              |          |                                                   |                                       |
| `CONFIG_DIR`                  |   Yes*   | Configuration directory path                      | -                                     |
| `MAX_EMBEDDINGS`              |    No    | Max embeddings for reindex                        | unlimited                             |
| **DIAL Integration**          |          |                                                   |                                       |
| `DIAL_URL`                    |    No    | DIAL URL for file uploads                         | -                                     |
| `DIAL_API_KEY`                |    No    | DIAL API key                                      | -                                     |
| **Authentication**            |          |                                                   |                                       |
| `AUTH_PROVIDER`               |    No    | Auth provider (`azure`, `keycloak`, `auth0`)      | -                                     |
| `AUTH_CALLBACK_PORT`          | Yes****  | Fixed port for OAuth callback (e.g., `8142`)      | dynamic                               |
| `AUTH_AZURE_CLIENT_ID`        |  Yes**   | Azure application/client ID                       | -                                     |
| `AUTH_AZURE_AUTHORITY`        |  Yes**   | Authority URL (includes tenant)                   | -                                     |
| `AUTH_AZURE_SCOPE`            |  Yes**   | Token scope                                       | -                                     |
| `AUTH_AZURE_CLIENT_SECRET`    |  Yes***  | Client secret (M2M/CI only)                       | -                                     |
| `AUTH_KEYCLOAK_SERVER_URL`    |  Yes**   | Keycloak server URL                               | -                                     |
| `AUTH_KEYCLOAK_REALM`         |  Yes**   | Keycloak realm name                               | -                                     |
| `AUTH_KEYCLOAK_CLIENT_ID`     |  Yes**   | Keycloak client ID                                | -                                     |
| `AUTH_KEYCLOAK_CLIENT_SECRET` |  Yes***  | Client secret (M2M/CI only)                       | -                                     |
| `AUTH_KEYCLOAK_SCOPE`         |    No    | OAuth scope                                       | `openid`                              |
| `AUTH_AUTH0_DOMAIN`           |  Yes**   | Auth0 tenant domain (e.g., mytenant.us.auth0.com) | -                                     |
| `AUTH_AUTH0_CLIENT_ID`        |  Yes**   | Auth0 application client ID                       | -                                     |
| `AUTH_AUTH0_AUDIENCE`         |  Yes**   | Auth0 API identifier (audience)                   | -                                     |
| `AUTH_AUTH0_CLIENT_SECRET`    |  Yes***  | Client secret (M2M/CI only)                       | -                                     |
| `AUTH_AUTH0_SCOPE`            |    No    | OAuth scope                                       | `openid profile email offline_access` |
| **General**                   |          |                                                   |                                       |
| `LOG_LEVEL`                   |    No    | `DEBUG`, `INFO`, `WARNING`, `ERROR`               | `INFO`                                |
| `DATA_DIR`                    |    No    | CLI data directory                                | `~/.statgpt`                          |

\* Required for `content init` command
\** Required for `auth login` (Azure, Keycloak, or Auth0, depending on `AUTH_PROVIDER`)
\*** Required for `auth login --method system_user`
\**** Required for Auth0 interactive login (Auth0 doesn't support wildcard ports)

## Data Directory

The CLI stores persistent data in `~/.statgpt/` (configurable via `STATGPT_CLI_DATA_DIR`):

| File               | Description                                           |
|--------------------|-------------------------------------------------------|
| `cli_history`      | Command history for up/down arrow navigation          |
| `token_cache.json` | Cached authentication tokens (restricted permissions) |

## Authentication Provider Setup

All providers support two authentication flows:
- **Interactive login** - Browser-based authentication using PKCE (for developers)
- **System user login** - Client Credentials Grant (for CI/CD pipelines)

### Azure Entra ID

**For Interactive Login (Public Client):**

1. Go to Azure Portal → Microsoft Entra ID → App registrations
2. Create new registration (e.g., "StatGPT CLI")
3. Under Authentication:
   - Add platform: Mobile and desktop applications
   - Add redirect URI: `http://localhost` (MSAL handles port internally)
   - Enable "Allow public client flows"
4. Note the Application (client) ID and Directory (tenant) ID
5. Configure environment variables:
   ```bash
   STATGPT_CLI_AUTH_PROVIDER=azure
   STATGPT_CLI_AUTH_AZURE_CLIENT_ID={application-id}
   STATGPT_CLI_AUTH_AZURE_AUTHORITY=https://login.microsoftonline.com/{tenant-id}
   STATGPT_CLI_AUTH_AZURE_SCOPE=api://{api-client-id}/.default
   ```

**For M2M/CI (Service Principal):**

1. In the same or different app registration, go to Certificates & secrets
2. Create new client secret, note the value
3. Ensure the app has required API permissions (Application permissions, not Delegated)
4. Grant admin consent for the permissions
5. Add to environment:
   ```bash
   STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET={secret-value}
   ```

### Keycloak

**For Interactive Login (Public Client):**

1. Go to Keycloak Admin Console → Clients → Create client
2. Client type: OpenID Connect
3. Client ID: e.g., `statgpt-cli`
4. Client authentication: OFF (public client)
5. Standard flow: Enabled
6. Valid redirect URIs: `http://localhost:*/callback` or fixed port (e.g., `http://localhost:8142/callback`)
7. Configure environment variables:
   ```bash
   STATGPT_CLI_AUTH_PROVIDER=keycloak
   STATGPT_CLI_AUTH_KEYCLOAK_SERVER_URL=https://keycloak.example.com
   STATGPT_CLI_AUTH_KEYCLOAK_REALM={realm-name}
   STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_ID=statgpt-cli
   ```

**For M2M/CI (Service Account):**

1. Create a new client (can be same or separate from interactive client)
2. Client authentication: ON (confidential client)
3. Enable "Service accounts roles" in capability config
4. In Credentials tab, copy the client secret
5. Assign required roles to the service account user
6. Add to environment:
   ```bash
   STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_SECRET={client-secret}
   ```

### Auth0

**For Interactive Login (Native Application):**

1. Go to Auth0 Dashboard → Applications → Create Application
2. Choose "Native" application type
3. In Settings:
   - Allowed Callback URLs: `http://localhost:8142/callback` (must match `AUTH_CALLBACK_PORT`)
4. Note the Domain and Client ID
5. Configure environment variables:
   ```bash
   STATGPT_CLI_AUTH_PROVIDER=auth0
   STATGPT_CLI_AUTH_AUTH0_DOMAIN={tenant}.us.auth0.com
   STATGPT_CLI_AUTH_AUTH0_CLIENT_ID={client-id}
   STATGPT_CLI_AUTH_AUTH0_AUDIENCE={api-identifier}
   STATGPT_CLI_AUTH_CALLBACK_PORT=8142
   ```

**For M2M/CI (Machine-to-Machine):**

1. Create NEW Application → Machine to Machine
2. Select the API to authorize
3. Copy Client ID and Client Secret
4. Configure environment (use M2M app credentials):
   ```bash
   STATGPT_CLI_AUTH_AUTH0_CLIENT_ID={m2m-client-id}
   STATGPT_CLI_AUTH_AUTH0_CLIENT_SECRET={m2m-client-secret}
   ```

**Note:** Auth0 requires a fixed callback port (doesn't support wildcard ports in redirect URIs).

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

| Problem                 | Solution                                                     |
|-------------------------|--------------------------------------------------------------|
| Module not found        | Run `poetry install -E cli` or `pip install statgpt[cli]`    |
| Authentication failures | Check `settings` command, verify Azure/Keycloak/Auth0 config |
| Connection refused      | Verify Admin API: `curl http://localhost:8000/health`        |
| Command not recognized  | Run `help` to see available commands                         |
