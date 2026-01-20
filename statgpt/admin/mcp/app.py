from fastmcp import FastMCP

from statgpt.admin.mcp.prompts import register_prompts
from statgpt.admin.mcp.services import register_tools

mcp = FastMCP(name="statgpt_dataset_exploration")

register_tools(mcp=mcp)
register_prompts(mcp=mcp)
