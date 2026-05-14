# StatGPT Admin MCP Setup

MCP server for coding agents such as Cursor or Claude Code. Used for dataset onboarding.

> **Note:** This is an optional beta feature. The available tools and prompts are subject to change.

## Prerequisites

1. Install dependencies:
   ```bash
   poetry install -E mcp
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
3. Verify the MCP was successfully loaded and its status is green.
If not, ensure admin app is running and try to reload the MCP.

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

The recommended way of adding new dataset configuration is by using specific prompt provided by MCP:

- Open corresponding `yaml` file with datasets configuration
- Call MCP prompt and add your query afterwards. For example:
  ```
  /statgpt-mcp/add_dataset_config please add CPI dataset
  ```
- Alternatively, you can send your query directly to the coding agent without calling
  the MCP prompt. However, this will likely lead to subtle mistakes in the generated
  dataset config.
- After coding agent finishes onboarding dataset, it's **strongly advised** to:
  - Check if coding agent validated generated configs using validation MCP tool.
    If not, ask coding agent to validate newly added datasets
  - Check generated config manually for any mistakes, redundancies, inefficiencies
    (like not re-using existing yaml anchors)
  - Review any new Named Entity types added to channel config
