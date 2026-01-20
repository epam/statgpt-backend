from fastmcp import FastMCP

from .dataset_exploration import (
    get_data_sources,
    get_dataset_dimensions_and_attributes,
    get_dataset_structure,
    get_datasets,
    validate_dataset_config,
    generate_id
)


def register_tools(mcp: FastMCP):
    mcp.tool()(get_data_sources)
    mcp.tool()(get_dataset_dimensions_and_attributes)
    mcp.tool()(get_dataset_structure)
    mcp.tool()(validate_dataset_config)
    mcp.tool()(get_datasets)
    mcp.tool()(generate_id)