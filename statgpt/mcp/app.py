from fastmcp import FastMCP

from statgpt.mcp.prompts.config_creation import add_config_for_dataset  # noqa: E402

# register tools
from statgpt.mcp.tools.dataset_exploration import (  # noqa: E402
    generate_id,
    generate_title_and_urn,
    get_data_sources,
    get_dataset_combinations,
    get_dataset_dimensions,
    get_datasets,
    validate_dataset_config,
)

mcp = FastMCP(name="statgpt_dataset_exploration")

mcp.tool()(get_data_sources)
mcp.tool()(get_datasets)
mcp.tool()(get_dataset_dimensions)
mcp.tool()(get_dataset_combinations)
mcp.tool()(validate_dataset_config)
mcp.tool()(generate_id)
mcp.tool()(generate_title_and_urn)
mcp.prompt()(add_config_for_dataset)
# ds struct
# attributes
# dimension values
# simplify/generalize prompt

mcp_app = mcp.http_app(path="/mcp", transport="streamable-http", stateless_http=True)
