from fastmcp import FastMCP

from .dataset_config import add_config_for_dataset, update_anchors_for_datasets


def register_prompts(mcp: FastMCP):
    mcp.prompt()(add_config_for_dataset)
    mcp.prompt()(update_anchors_for_datasets)
