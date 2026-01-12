from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.mcp.prompts.config_creation import add_config_for_dataset  # noqa: E402

# register tools
from statgpt.admin.mcp.services.dataset_exploration import (  # noqa: E402
    generate_id,
    get_data_sources_list,
    get_dataset_combinations,
    get_dataset_dimensions,
    get_datasets,
    validate_dataset_config,
)
from statgpt.common.models import get_session_contex_manager


mcp = FastMCP(name="statgpt_dataset_exploration")


@mcp.tool
async def get_data_sources(
    session: AsyncSession = Depends(get_session_contex_manager),
) -> list[...]:
    """Returns ..."""
    return await get_data_sources_list(session)


mcp.tool()(get_datasets)
mcp.tool()(get_dataset_dimensions)
mcp.tool()(get_dataset_combinations)
mcp.tool()(validate_dataset_config)
mcp.tool()(generate_id)
mcp.prompt()(add_config_for_dataset)
# ds struct
# attributes
# dimension values
# simplify/generalize prompt
