from fastmcp import FastMCP

from .dataset_config import add_config_for_dataset, validate_generated_config


def register_prompts(mcp: FastMCP):
    mcp.prompt()(add_config_for_dataset)
    mcp.prompt()(validate_generated_config)
