# StatGPT Admin MCP Setup

MCP server for coding agents such as Cursor or Claude Code. Used for dataset onboarding.

> **Note:** This is an optional beta feature. The available tools and prompts are subject to change.

## Prerequisites

1. Install dependencies:
   ```bash
   poetry install -E beta-mcp
   ```
   or with make:
   ```bash
   make install_dev
   ```

2. Enable MCP by setting the environment variable:
   ```bash
   BETA_MCP_ENABLED=true
   ```

## Start MCP Server

Start the Admin app to run the MCP server:

```bash
make statgpt_admin
```

## Coding Agent Configuration

### Cursor

1. Go to **Settings** → **Tools and MCP** → **Add New MCP Server**
2. Add the following configuration:
   ```json
   {
     "mcpServers": {
       "statgpt-mcp": {
         "type": "http",
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }
   ```
3. Verify the MCP status is green. If needed, toggle the MCP button to refresh.

### Claude Code

1) Add the MCP server via terminal:
   ```bash
   claude mcp add --transport http statgpt-mcp http://127.0.0.1:8000/mcp
   ```

2) Verify it’s connected
   ```bash
   claude mcp list
   claude mcp get statgpt-mcp
   ```

## Usage

### Datasets onboarding

To add dataset configuration:

- open corresponding `yaml` file with datasets configuration
- mention MCP prompt respondible for dataset onboarding and pass requied context:
  ```
  /statgpt-mcp/add_config_for_dataset <query>
  ```
  for example:
  ```
  /statgpt-mcp/add_config_for_dataset please add CPI dataset
  ```
  alternatively, you can avoid mentioning prompt explicitly -
  coding agent will need find it on its own
- after coding agent finishes onboarding dataset,
  it's **strongly advised** to check if MCP called dataset validation.
  if not, ask coding agent to validate newly added datasets
